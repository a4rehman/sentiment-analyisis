# ============================================================
#  app.py – Streamlit Frontend for Emotion Detection System
# ============================================================

import os, sys, time, logging, threading, tempfile
from pathlib import Path
import numpy as np
import cv2
import streamlit as st
import pandas as pd

# ── Path fix so modules resolve correctly ──────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    LOGS_DIR, REPORTS_DIR, SNAPSHOTS_DIR,
    PROCESS_EVERY_N, RESIZE_FOR_DETECT,
    MAX_DISAPPEARED, IOU_THRESHOLD,
    EMOTION_COLORS, EMOTION_EMOJIS, EMOTIONS,
)
from modules.detector   import EmotionDetector
from modules.tracker    import FaceTracker
from modules.logger     import EmotionLogger, load_all_logs
from modules.alerts     import AlertSystem
from modules.analytics  import (
    SessionStats, build_bar_chart, build_timeline_chart,
    build_face_count_chart, fig_to_bytes,
    export_csv_report, export_summary_txt,
)
from modules.video_processor import VideoProcessor
from utils.helpers import save_uploaded_video, frame_to_jpeg_bytes, list_log_files

logging.basicConfig(level=logging.INFO)

# ═══════════════════════════════════════════════════════════════════════════════
#  Page config
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title  = "EmotiSense AI | Real-time Emotion Detection",
    page_icon   = "😊",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── Load CSS ──────────────────────────────────────────────────────────────────
css_path = Path(__file__).parent / "assets" / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  Session State Init
# ═══════════════════════════════════════════════════════════════════════════════
def _init_state():
    defaults = {
        "running":        False,
        "alerts":         [],
        "stats":          None,
        "logger_obj":     None,
        "vp":             None,
        "alert_sys":      None,
        "model_choice":   "fer",
        "source_type":    "webcam",
        "frame_placeholder": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ═══════════════════════════════════════════════════════════════════════════════
#  Sidebar
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:10px 0 20px'>
      <span style='font-size:2.5rem'>🧠</span>
      <h2 style='margin:0;color:#00d4ff;font-weight:700'>EmotiSense AI</h2>
      <p style='color:#8b949e;font-size:0.8rem;margin:0'>Real-time Emotion Intelligence</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Model selection ──────────────────────────────────────────────────────
    st.markdown("#### ⚙️ Model Settings")
    model_choice = st.selectbox(
        "Detection Model",
        ["fer", "deepface"],
        index=0,
        help="FER is faster; DeepFace is more accurate.",
    )
    if model_choice == "deepface":
        df_backend = st.selectbox(
            "DeepFace Backend",
            ["opencv", "ssd", "mtcnn", "retinaface"],
            index=0,
        )
    else:
        df_backend = "opencv"

    skip_n = st.slider("Process Every N Frames", 1, 5, PROCESS_EVERY_N,
                       help="Higher = faster, fewer detections.")
    resize_w = st.slider("Detection Resize Width", 320, 1280, RESIZE_FOR_DETECT, 64)

    st.divider()

    # ── Alert settings ───────────────────────────────────────────────────────
    st.markdown("#### 🔔 Alert Settings")
    anger_thresh = st.slider("Anger Confidence Threshold", 0.3, 1.0, 0.70, 0.05)
    crowd_thresh = st.slider("Crowd Anger Count", 1, 10, 2)

    st.divider()

    # ── Source selection ─────────────────────────────────────────────────────
    st.markdown("#### 📹 Input Source")
    source_type = st.radio(
        "Source",
        ["📷 Webcam", "🎬 Video File", "📡 IP/RTSP Camera"],
        index=0,
    )

    webcam_idx  = 0
    video_file  = None
    rtsp_url    = ""

    if "Webcam" in source_type:
        webcam_idx = st.number_input("Camera Index", 0, 10, 0)
    elif "Video File" in source_type:
        video_file = st.file_uploader(
            "Upload Video", type=["mp4", "avi", "mov", "mkv", "webm"]
        )
    else:
        rtsp_url = st.text_input("Stream URL", placeholder="rtsp://user:pass@ip:port/stream")

    st.divider()
    st.markdown(
        "<p style='color:#8b949e;font-size:0.75rem;text-align:center'>"
        "EmotiSense AI v1.0 · Built with OpenCV + FER/DeepFace"
        "</p>",
        unsafe_allow_html=True,
    )

# ═══════════════════════════════════════════════════════════════════════════════
#  Navigation tabs
# ═══════════════════════════════════════════════════════════════════════════════
tab_live, tab_analytics, tab_logs, tab_reports = st.tabs([
    "🎥 Live Detection", "📊 Analytics Dashboard", "📋 Emotion Logs", "📁 Reports"
])

# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 1 – Live Detection
# ═══════════════════════════════════════════════════════════════════════════════
with tab_live:
    st.markdown("## 🎥 Real-time Emotion Detection")

    # KPI row
    k1, k2, k3, k4, k5 = st.columns(5)
    kpi_fps    = k1.empty()
    kpi_faces  = k2.empty()
    kpi_mood   = k3.empty()
    kpi_events = k4.empty()
    kpi_time   = k5.empty()

    # Alert zone
    alert_zone = st.empty()

    # Video frame
    frame_placeholder = st.empty()

    # Control buttons
    col_start, col_stop, col_snap = st.columns([1, 1, 2])
    start_btn = col_start.button("▶ Start", use_container_width=True, type="primary")
    stop_btn  = col_stop.button("⏹ Stop",  use_container_width=True)

    # ── Helper to build pipeline ──────────────────────────────────────────
    def build_pipeline():
        detector  = EmotionDetector(
            model            = model_choice,
            deepface_backend = df_backend,
            process_every_n  = skip_n,
            resize_width     = resize_w,
        )
        tracker   = FaceTracker(max_disappeared=MAX_DISAPPEARED, iou_threshold=IOU_THRESHOLD)
        logger_   = EmotionLogger(session_name=model_choice)
        alert_sys = AlertSystem(
            anger_conf_thresh  = anger_thresh,
            anger_count_thresh = crowd_thresh,
        )
        stats     = SessionStats()
        return detector, tracker, logger_, alert_sys, stats

    def get_source():
        if "Webcam" in source_type:
            return webcam_idx, "webcam"
        elif "Video File" in source_type:
            if video_file is None:
                st.error("Please upload a video file first.")
                return None, None
            path = save_uploaded_video(video_file)
            return path, "video_file"
        else:
            if not rtsp_url:
                st.error("Please enter a stream URL.")
                return None, None
            return rtsp_url, "rtsp"

    # ── Start logic ──────────────────────────────────────────────────────
    if start_btn and not st.session_state.running:
        src, src_label = get_source() if not (
            "Webcam" in source_type and source_type
        ) else (webcam_idx, "webcam")

        # Re-resolve for all source types
        if "Webcam" in source_type:
            src, src_label = webcam_idx, "webcam"
        elif "Video File" in source_type:
            if video_file is None:
                st.error("Upload a video file first.")
                src = None
            else:
                src = save_uploaded_video(video_file)
                src_label = "video_file"
        else:
            src = rtsp_url.strip()
            src_label = "rtsp"
            if not src:
                st.error("Enter a stream URL.")
                src = None

        if src is not None:
            detector, tracker, logger_, alert_sys, stats = build_pipeline()
            vp = VideoProcessor(detector, tracker, logger_, alert_sys, stats,
                                source=src_label)
            if vp.open(src):
                st.session_state.running    = True
                st.session_state.vp         = vp
                st.session_state.stats      = stats
                st.session_state.logger_obj = logger_
                st.session_state.alert_sys  = alert_sys
                st.session_state.alerts     = []
            else:
                st.error(f"❌ Cannot open source: {src}")

    if stop_btn:
        st.session_state.running = False
        if st.session_state.vp:
            st.session_state.vp.release()

    # ── Streaming loop ───────────────────────────────────────────────────
    if st.session_state.running and st.session_state.vp:
        vp    = st.session_state.vp
        stats = st.session_state.stats

        EMOTION_HEX = {
            "angry":"#ef4444","disgust":"#f97316","fear":"#a855f7",
            "happy":"#22c55e","neutral":"#94a3b8","sad":"#3b82f6","surprise":"#06b6d4",
        }

        for annotated, results, new_alerts in vp.stream():
            if not st.session_state.running:
                break

            # KPIs
            dom   = stats.dominant_emotion() or "—"
            emoji = EMOTION_EMOJIS.get(dom, "")
            kpi_fps.metric("⚡ FPS",          f"{vp.fps:.1f}")
            kpi_faces.metric("👥 Faces",       len(results))
            kpi_mood.metric("🎭 Mood",         f"{emoji} {dom}")
            kpi_events.metric("📌 Events",     stats.total_events())
            kpi_time.metric("⏱ Duration",      stats.session_duration())

            # Alerts
            st.session_state.alerts.extend(new_alerts)
            recent = st.session_state.alerts[-3:]
            if recent:
                html = ""
                for a in reversed(recent):
                    cls = {"CRITICAL":"alert-critical","WARNING":"alert-warning","INFO":"alert-info"}.get(a.level,"alert-info")
                    html += f'<div class="{cls}"><b>[{a.level}]</b> {a.message}</div>'
                alert_zone.markdown(html, unsafe_allow_html=True)

            # Video frame
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(rgb, channels="RGB", use_container_width=True)

        st.session_state.running = False
        if vp:
            vp.release()
        st.success("✅ Session complete. Check the Analytics tab for results.")

# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 2 – Analytics Dashboard
# ═══════════════════════════════════════════════════════════════════════════════
with tab_analytics:
    st.markdown("## 📊 Analytics Dashboard")

    stats = st.session_state.get("stats")
    if stats is None or stats.total_events() == 0:
        st.info("ℹ️ Run a detection session first to see analytics.")
    else:
        # ── Summary metrics ──────────────────────────────────────────────
        dom  = stats.dominant_emotion() or "—"
        dist = stats.emotion_distribution()
        conf = stats.avg_confidence()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Events",    stats.total_events())
        m2.metric("Session Duration", stats.session_duration())
        m3.metric("Dominant Emotion", f"{EMOTION_EMOJIS.get(dom,'')} {dom.title()}")
        m4.metric("Crowd Mood",       stats.crowd_mood())

        st.divider()

        # ── Charts ──────────────────────────────────────────────────────
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 🎭 Emotion Distribution")
            fig_bar = build_bar_chart(stats)
            st.image(fig_to_bytes(fig_bar), use_container_width=True)

        with c2:
            st.markdown("#### 📈 Confidence Timeline")
            fig_tl = build_timeline_chart(stats)
            st.image(fig_to_bytes(fig_tl), use_container_width=True)

        st.markdown("#### 👥 Face Count Over Time")
        fig_fc = build_face_count_chart(stats)
        st.image(fig_to_bytes(fig_fc), use_container_width=True)

        st.divider()

        # ── Per-emotion stats table ──────────────────────────────────────
        st.markdown("#### 📋 Per-Emotion Breakdown")
        rows = []
        for e in EMOTIONS:
            rows.append({
                "Emotion":       f"{EMOTION_EMOJIS.get(e,'')} {e.title()}",
                "Count":         int(dist[e] * stats.total_events()),
                "Share (%)":     f"{dist[e]*100:.1f}",
                "Avg Confidence":f"{conf[e]:.0%}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # ── Alert history ────────────────────────────────────────────────
        alert_sys = st.session_state.get("alert_sys")
        if alert_sys and alert_sys.history:
            st.divider()
            st.markdown("#### 🔔 Alert History")
            for a in reversed(alert_sys.history[-20:]):
                ts_str = time.strftime("%H:%M:%S", time.localtime(a.timestamp))
                cls    = {"CRITICAL":"alert-critical","WARNING":"alert-warning","INFO":"alert-info"}.get(a.level,"alert-info")
                st.markdown(
                    f'<div class="{cls}"><small>{ts_str}</small> <b>[{a.level}]</b> {a.message}</div>',
                    unsafe_allow_html=True,
                )

# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 3 – Emotion Logs
# ═══════════════════════════════════════════════════════════════════════════════
with tab_logs:
    st.markdown("## 📋 Emotion Logs")

    log_files = list_log_files(LOGS_DIR)
    if not log_files:
        st.info("ℹ️ No log files yet. Run a detection session to generate logs.")
    else:
        selected_log = st.selectbox("Select Log File", log_files)
        log_path     = os.path.join(LOGS_DIR, selected_log)

        try:
            df_log = pd.read_csv(log_path)
            st.markdown(f"**{len(df_log)} events** · `{selected_log}`")

            # Filter controls
            fc1, fc2 = st.columns(2)
            sel_emotions = fc1.multiselect("Filter by Emotion", EMOTIONS, default=EMOTIONS)
            min_conf     = fc2.slider("Min Confidence", 0.0, 1.0, 0.0, 0.05)

            filtered = df_log[
                df_log["emotion"].isin(sel_emotions) &
                (df_log["confidence"] >= min_conf)
            ]
            st.dataframe(filtered, use_container_width=True, height=400)

            st.download_button(
                "⬇ Download Filtered CSV",
                filtered.to_csv(index=False).encode(),
                file_name=f"filtered_{selected_log}",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"Could not read log: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 4 – Reports
# ═══════════════════════════════════════════════════════════════════════════════
with tab_reports:
    st.markdown("## 📁 Export Reports")

    stats = st.session_state.get("stats")
    if stats is None or stats.total_events() == 0:
        st.info("ℹ️ No session data to export yet.")
    else:
        st.markdown("### Generate Reports")
        rc1, rc2 = st.columns(2)

        with rc1:
            if st.button("📄 Export Summary TXT", use_container_width=True):
                path = export_summary_txt(stats, session_name=model_choice)
                with open(path, "r") as f:
                    content = f.read()
                st.download_button(
                    "⬇ Download Summary",
                    content.encode(),
                    file_name=os.path.basename(path),
                    mime="text/plain",
                )
                st.success(f"✅ Saved: `{path}`")

        with rc2:
            if st.button("📊 Export All Logs CSV", use_container_width=True):
                df_all = load_all_logs()
                if df_all.empty:
                    st.warning("No log data found.")
                else:
                    path = export_csv_report(df_all, session_name="all")
                    st.download_button(
                        "⬇ Download All Logs",
                        df_all.to_csv(index=False).encode(),
                        file_name=os.path.basename(path),
                        mime="text/csv",
                    )
                    st.success(f"✅ Saved: `{path}`")

        st.divider()
        st.markdown("### 📂 Saved Reports")
        report_files = [f for f in os.listdir(REPORTS_DIR) if f.endswith((".csv", ".txt"))]
        if report_files:
            for rf in sorted(report_files, reverse=True):
                rpath = os.path.join(REPORTS_DIR, rf)
                size  = os.path.getsize(rpath)
                st.markdown(f"- `{rf}` ({size/1024:.1f} KB)")
        else:
            st.info("No reports generated yet.")

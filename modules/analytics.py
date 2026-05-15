# ============================================================
#  modules/analytics.py – Analytics & Report Generator
# ============================================================
"""
Provides:
  - Session statistics
  - Crowd mood summary
  - Emotion timeline (trend)
  - Matplotlib/Plotly chart generation
  - PDF/CSV report export
"""

from __future__ import annotations
import io
import logging
import os
from collections import Counter, defaultdict
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import numpy as np

from config import EMOTIONS, EMOTION_EMOJIS, REPORTS_DIR

_logger = logging.getLogger(__name__)


# ─── Session Statistics ───────────────────────────────────────────────────────
class SessionStats:
    """Accumulates per-session emotion statistics."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._counts: Counter            = Counter()
        self._conf_sum: Dict[str, float] = defaultdict(float)
        self._timeline: List[dict]       = []   # {time, emotion, confidence}
        self._face_count_history: List[int] = []
        self._start = datetime.now()

    # ------------------------------------------------------------------
    def update(self, results, source: str = "webcam"):
        from modules.detector import FaceResult
        ts = datetime.now().isoformat(timespec="seconds")
        self._face_count_history.append(len(results))
        for r in results:
            self._counts[r.emotion]   += 1
            self._conf_sum[r.emotion] += r.confidence
            self._timeline.append({
                "timestamp":  ts,
                "emotion":    r.emotion,
                "confidence": r.confidence,
                "track_id":   r.track_id,
            })

    # ------------------------------------------------------------------
    def dominant_emotion(self) -> Optional[str]:
        if not self._counts:
            return None
        return self._counts.most_common(1)[0][0]

    def crowd_mood(self) -> str:
        dom = self.dominant_emotion()
        if dom is None:
            return "No data"
        emoji = EMOTION_EMOJIS.get(dom, "")
        return f"{emoji} {dom.title()}"

    def emotion_distribution(self) -> Dict[str, float]:
        total = sum(self._counts.values()) or 1
        return {e: self._counts.get(e, 0) / total for e in EMOTIONS}

    def avg_confidence(self) -> Dict[str, float]:
        result = {}
        for e in EMOTIONS:
            cnt = self._counts.get(e, 0)
            result[e] = self._conf_sum[e] / cnt if cnt else 0.0
        return result

    def total_events(self) -> int:
        return sum(self._counts.values())

    def session_duration(self) -> str:
        delta = datetime.now() - self._start
        m, s  = divmod(int(delta.total_seconds()), 60)
        return f"{m:02d}:{s:02d}"

    def timeline_df(self) -> "pd.DataFrame":
        import pandas as pd
        return pd.DataFrame(self._timeline)

    def face_count_series(self) -> List[int]:
        return list(self._face_count_history)


# ─── Chart Builders ───────────────────────────────────────────────────────────
def build_bar_chart(stats: SessionStats) -> "plt.Figure":
    """Emotion distribution horizontal bar chart."""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")

    dist = stats.emotion_distribution()
    emotions  = list(dist.keys())
    values    = [dist[e] * 100 for e in emotions]
    colors    = [
        "#FF4444", "#FF8800", "#AA44FF",
        "#44DD44", "#AAAAAA", "#4488FF", "#00DDDD"
    ]

    fig, ax = plt.subplots(figsize=(6, 3), facecolor="#1a1a2e")
    ax.set_facecolor("#16213e")
    bars = ax.barh(emotions, values, color=colors, edgecolor="none", height=0.6)
    ax.set_xlabel("Percentage (%)", color="white")
    ax.set_title("Emotion Distribution", color="white", fontsize=13, fontweight="bold")
    ax.tick_params(colors="white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#444")
    ax.spines["left"].set_color("#444")
    for bar, val in zip(bars, values):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", color="white", fontsize=9)
    plt.tight_layout()
    return fig


def build_timeline_chart(stats: SessionStats, last_n: int = 200) -> "plt.Figure":
    """Emotion confidence timeline line chart."""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")

    df = stats.timeline_df()
    if df.empty:
        fig, ax = plt.subplots(facecolor="#1a1a2e")
        ax.set_facecolor("#16213e")
        ax.text(0.5, 0.5, "No data yet", transform=ax.transAxes,
                ha="center", color="white")
        return fig

    df = df.tail(last_n)
    palette = {
        "angry": "#FF4444", "disgust": "#FF8800", "fear": "#AA44FF",
        "happy": "#44DD44", "neutral": "#AAAAAA", "sad": "#4488FF",
        "surprise": "#00DDDD",
    }

    fig, ax = plt.subplots(figsize=(8, 3), facecolor="#1a1a2e")
    ax.set_facecolor("#16213e")
    for emo in EMOTIONS:
        sub = df[df["emotion"] == emo]
        if sub.empty:
            continue
        ax.plot(
            range(len(sub)), sub["confidence"] * 100,
            label=emo, color=palette.get(emo, "white"),
            linewidth=1.5, alpha=0.85,
        )

    ax.set_xlabel("Event #", color="white")
    ax.set_ylabel("Confidence %", color="white")
    ax.set_title("Emotion Confidence Timeline", color="white", fontsize=13, fontweight="bold")
    ax.tick_params(colors="white")
    ax.legend(loc="upper right", fontsize=7, facecolor="#222", labelcolor="white",
              framealpha=0.6)
    for sp in ax.spines.values():
        sp.set_color("#444")
    plt.tight_layout()
    return fig


def build_face_count_chart(stats: SessionStats) -> "plt.Figure":
    """Faces detected per frame sparkline."""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")

    hist = stats.face_count_series()[-300:]
    fig, ax = plt.subplots(figsize=(6, 2), facecolor="#1a1a2e")
    ax.set_facecolor("#16213e")
    if hist:
        ax.fill_between(range(len(hist)), hist, alpha=0.5, color="#00DDDD")
        ax.plot(hist, color="#00DDDD", linewidth=1.5)
    ax.set_title("Faces Detected (recent)", color="white", fontsize=11, fontweight="bold")
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_color("#444")
    plt.tight_layout()
    return fig


def fig_to_bytes(fig) -> bytes:
    """Convert Matplotlib figure to PNG bytes."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


# ─── Report Export ────────────────────────────────────────────────────────────
def export_csv_report(df: "pd.DataFrame", session_name: str = "") -> str:
    """Save a copy of the log DataFrame as a report CSV."""
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"report_{ts}_{session_name}.csv" if session_name else f"report_{ts}.csv"
    path = os.path.join(REPORTS_DIR, name)
    df.to_csv(path, index=False)
    return path


def export_summary_txt(stats: SessionStats, session_name: str = "") -> str:
    """Write a plain-text analytics summary."""
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"summary_{ts}_{session_name}.txt" if session_name else f"summary_{ts}.txt"
    path = os.path.join(REPORTS_DIR, name)
    dist = stats.emotion_distribution()
    conf = stats.avg_confidence()
    lines = [
        "=" * 50,
        "  EMOTION DETECTION SESSION SUMMARY",
        "=" * 50,
        f"Session duration : {stats.session_duration()}",
        f"Total events     : {stats.total_events()}",
        f"Dominant emotion : {stats.dominant_emotion()}",
        f"Crowd mood       : {stats.crowd_mood()}",
        "",
        "─── Emotion Distribution ─────────────────",
    ]
    for e in EMOTIONS:
        lines.append(f"  {e:<10}: {dist[e]*100:5.1f}%  (avg conf {conf[e]:.0%})")
    lines += ["", "=" * 50]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path

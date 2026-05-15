# ============================================================
#  modules/video_processor.py – OpenCV Video Processing Pipeline
# ============================================================
"""
Handles three input sources:
  1. Webcam  (device index)
  2. Video file upload (path)
  3. RTSP / IP camera stream (URL)

Yields annotated BGR frames + results for each frame.
"""

from __future__ import annotations
import logging
import time
from typing import Generator, List, Optional, Tuple, Union

import cv2
import numpy as np

from config import WEBCAM_WIDTH, WEBCAM_HEIGHT, PROCESS_EVERY_N
from modules.detector import EmotionDetector, FaceResult
from modules.tracker import FaceTracker
from modules.logger import EmotionLogger
from modules.alerts import AlertSystem
from modules.analytics import SessionStats
from utils.drawing import draw_results, draw_hud

_logger = logging.getLogger(__name__)


class VideoProcessor:
    """
    Orchestrates the full pipeline:

      VideoCapture → EmotionDetector → FaceTracker
          → EmotionLogger → AlertSystem → Annotated Frame
    """

    def __init__(
        self,
        detector:   EmotionDetector,
        tracker:    FaceTracker,
        logger:     EmotionLogger,
        alert_sys:  AlertSystem,
        stats:      SessionStats,
        source:     str = "webcam",
    ):
        self.detector   = detector
        self.tracker    = tracker
        self.logger_obj = logger
        self.alert_sys  = alert_sys
        self.stats      = stats
        self.source     = source
        self._cap: Optional[cv2.VideoCapture] = None
        self._fps   = 0.0
        self._frame_n = 0

    # ------------------------------------------------------------------
    def open(self, src: Union[int, str]) -> bool:
        """Open a capture source.  Returns True on success."""
        self._cap = cv2.VideoCapture(src)
        if isinstance(src, int):
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WEBCAM_WIDTH)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_HEIGHT)
            self._cap.set(cv2.CAP_PROP_FPS, 30)
        ok = self._cap.isOpened()
        if not ok:
            _logger.error("Cannot open capture: %s", src)
        return ok

    def release(self):
        if self._cap:
            self._cap.release()
            self._cap = None

    # ------------------------------------------------------------------
    def stream(
        self,
        max_frames: Optional[int] = None,
    ) -> Generator[Tuple[np.ndarray, List[FaceResult], List], None, None]:
        """
        Generator that yields (annotated_frame, results, new_alerts)
        until the source is exhausted or release() is called.
        """
        if self._cap is None or not self._cap.isOpened():
            _logger.error("VideoProcessor.stream() called without open().")
            return

        t_prev = time.perf_counter()
        while True:
            ok, frame = self._cap.read()
            if not ok:
                break

            self._frame_n += 1
            if max_frames and self._frame_n > max_frames:
                break

            # ── Detection ──────────────────────────────────────────
            results = self.detector.process_frame(frame)

            # ── Tracking ──────────────────────────────────────────
            results = self.tracker.update(results)

            # ── Logging + Stats ───────────────────────────────────
            if results:
                self.logger_obj.log(results, source=self.source)
                self.stats.update(results, source=self.source)

            # ── Alerts ────────────────────────────────────────────
            new_alerts = self.alert_sys.check(results)

            # ── FPS calculation ───────────────────────────────────
            t_now      = time.perf_counter()
            self._fps  = 1.0 / max(t_now - t_prev, 1e-6)
            t_prev     = t_now

            # ── Draw annotations ──────────────────────────────────
            annotated = draw_results(frame.copy(), results)
            annotated = draw_hud(annotated, self._fps, self._frame_n, results, self.stats)

            yield annotated, results, new_alerts

        self.logger_obj.flush()

    # ------------------------------------------------------------------
    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_count(self) -> int:
        return self._frame_n

    # ------------------------------------------------------------------
    @staticmethod
    def get_video_info(path: str) -> dict:
        """Return metadata for a video file."""
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return {}
        info = {
            "width":    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height":   int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps":      cap.get(cv2.CAP_PROP_FPS),
            "frames":   int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "duration": cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1),
        }
        cap.release()
        return info


# ─── Quick CLI helper ─────────────────────────────────────────────────────────
def run_cli(src: Union[int, str], model: str = "fer"):
    """Run the pipeline in a plain OpenCV window (no Streamlit)."""
    from config import (DETECTION_BACKEND, PROCESS_EVERY_N, RESIZE_FOR_DETECT,
                        MAX_DISAPPEARED, IOU_THRESHOLD)

    detector  = EmotionDetector(model=model, process_every_n=PROCESS_EVERY_N,
                                 resize_width=RESIZE_FOR_DETECT)
    tracker   = FaceTracker(max_disappeared=MAX_DISAPPEARED, iou_threshold=IOU_THRESHOLD)
    logger_   = EmotionLogger()
    alerts    = AlertSystem()
    stats     = SessionStats()

    vp = VideoProcessor(detector, tracker, logger_, alerts, stats,
                        source=str(src))
    if not vp.open(src):
        print("ERROR: Could not open source:", src)
        return

    print("Press 'q' to quit.")
    for frame, results, new_alerts in vp.stream():
        for a in new_alerts:
            print(a)
        cv2.imshow("Emotion Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    vp.release()
    cv2.destroyAllWindows()
    print("Session ended.  Log:", logger_.log_path)

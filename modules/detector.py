# ============================================================
#  modules/detector.py – Face & Emotion Detection Engine
# ============================================================
"""
Unified detector that supports two backends:
  - FER  : lightweight, fast, CPU-friendly
  - DeepFace : accurate, GPU-optional
"""

from __future__ import annotations
import time
import logging
import numpy as np
import cv2
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


# ─── Result DataClass ─────────────────────────────────────────────────────────
class FaceResult:
    """Holds result for one detected face."""
    __slots__ = ("bbox", "emotion", "confidence", "all_scores", "track_id", "timestamp")

    def __init__(
        self,
        bbox: tuple,          # (x, y, w, h)
        emotion: str,
        confidence: float,
        all_scores: Dict[str, float],
        track_id: int = -1,
    ):
        self.bbox        = bbox
        self.emotion     = emotion
        self.confidence  = round(confidence, 4)
        self.all_scores  = all_scores
        self.track_id    = track_id
        self.timestamp   = time.time()

    def to_dict(self) -> Dict[str, Any]:
        x, y, w, h = self.bbox
        return {
            "track_id":   self.track_id,
            "emotion":    self.emotion,
            "confidence": self.confidence,
            "face_x": x, "face_y": y,
            "face_w": w, "face_h": h,
            "all_scores": self.all_scores,
            "timestamp":  self.timestamp,
        }


# ─── FER Backend ─────────────────────────────────────────────────────────────
class FERDetector:
    """Wraps the `fer` library for real-time detection."""

    def __init__(self):
        try:
            from fer import FER
            self._fer = FER(mtcnn=False)   # mtcnn=True is slower but more accurate
            logger.info("FER detector initialised.")
        except ImportError:
            raise RuntimeError("FER not installed. Run: pip install fer")

    def detect(self, frame_rgb: np.ndarray) -> List[FaceResult]:
        results: List[FaceResult] = []
        try:
            detections = self._fer.detect_emotions(frame_rgb)
            for det in detections:
                box    = det.get("box", (0, 0, 0, 0))   # (x, y, w, h)
                scores = det.get("emotions", {})
                if not scores:
                    continue
                emotion    = max(scores, key=scores.get)
                confidence = float(scores[emotion])
                results.append(FaceResult(
                    bbox       = tuple(int(v) for v in box),
                    emotion    = emotion,
                    confidence = confidence,
                    all_scores = {k: float(v) for k, v in scores.items()},
                ))
        except Exception as exc:
            logger.warning("FER detection error: %s", exc)
        return results


# ─── DeepFace Backend ─────────────────────────────────────────────────────────
class DeepFaceDetector:
    """Wraps DeepFace for more accurate (but slower) emotion analysis."""

    def __init__(self, backend: str = "opencv"):
        self._backend = backend
        try:
            import deepface  # noqa: F401
            logger.info("DeepFace detector initialised (backend=%s).", backend)
        except ImportError:
            raise RuntimeError("DeepFace not installed. Run: pip install deepface")

    def detect(self, frame_rgb: np.ndarray) -> List[FaceResult]:
        from deepface import DeepFace
        results: List[FaceResult] = []
        try:
            analyses = DeepFace.analyze(
                frame_rgb,
                actions        = ["emotion"],
                detector_backend = self._backend,
                enforce_detection = False,
                silent         = True,
            )
            # DeepFace may return dict or list
            if isinstance(analyses, dict):
                analyses = [analyses]
            for ana in analyses:
                region = ana.get("region", {})
                x = int(region.get("x", 0))
                y = int(region.get("y", 0))
                w = int(region.get("w", 0))
                h = int(region.get("h", 0))
                scores  = ana.get("emotion", {})
                emotion = ana.get("dominant_emotion", "neutral")
                # Normalise scores to 0-1
                total   = sum(scores.values()) or 1
                norm    = {k: v / total for k, v in scores.items()}
                conf    = float(norm.get(emotion, 0.0))
                results.append(FaceResult(
                    bbox       = (x, y, w, h),
                    emotion    = emotion,
                    confidence = conf,
                    all_scores = norm,
                ))
        except Exception as exc:
            logger.warning("DeepFace detection error: %s", exc)
        return results


# ─── Unified Detector ─────────────────────────────────────────────────────────
class EmotionDetector:
    """
    High-level detector that:
      - chooses the backend (FER / DeepFace)
      - pre-processes frames for speed
      - provides frame-skip optimisation
    """

    def __init__(self, model: str = "fer", deepface_backend: str = "opencv",
                 process_every_n: int = 1, resize_width: int = 640):
        self._model         = model.lower()
        self._process_every = process_every_n
        self._resize_w      = resize_width
        self._frame_count   = 0
        self._last_results: List[FaceResult] = []

        if self._model == "fer":
            self._backend = FERDetector()
        elif self._model == "deepface":
            self._backend = DeepFaceDetector(backend=deepface_backend)
        else:
            raise ValueError(f"Unknown model: {model}. Choose 'fer' or 'deepface'.")

        logger.info("EmotionDetector ready (model=%s, every=%d frames).",
                    model, process_every_n)

    # ------------------------------------------------------------------
    def process_frame(self, frame_bgr: np.ndarray) -> List[FaceResult]:
        """
        Main entry point. Takes a BGR frame (from OpenCV), returns list
        of FaceResult objects.  Uses frame-skip for performance.
        """
        self._frame_count += 1

        # Return cached results on skipped frames
        if self._frame_count % self._process_every != 0:
            return self._last_results

        h, w = frame_bgr.shape[:2]
        scale = 1.0

        # Optionally downscale for detection
        if self._resize_w and w > self._resize_w:
            scale      = self._resize_w / w
            small      = cv2.resize(frame_bgr, (self._resize_w, int(h * scale)))
            frame_proc = small
        else:
            frame_proc = frame_bgr

        # Convert BGR → RGB (both FER and DeepFace expect RGB)
        frame_rgb = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2RGB)

        results = self._backend.detect(frame_rgb)

        # Scale bboxes back to original size if we downscaled
        if scale != 1.0:
            for r in results:
                x, y, ww, hh = r.bbox
                r.bbox = (
                    int(x / scale), int(y / scale),
                    int(ww / scale), int(hh / scale),
                )

        self._last_results = results
        return results

    # ------------------------------------------------------------------
    @property
    def model_name(self) -> str:
        return self._model

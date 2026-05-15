# ============================================================
#  modules/alerts.py – Real-time Alert System
# ============================================================
"""
Monitors detected emotions and raises:
  - Per-face anger alert  (confidence > threshold)
  - Crowd anger alert     (multiple angry faces simultaneously)
"""

from __future__ import annotations
import logging
import time
from typing import List, Optional

from config import (
    ANGER_ALERT_THRESHOLD,
    ANGER_COUNT_THRESHOLD,
    ALERT_COOLDOWN_SECONDS,
    EMOTION_EMOJIS,
)
from modules.detector import FaceResult

_logger = logging.getLogger(__name__)


class Alert:
    """Represents a single alert event."""
    __slots__ = ("level", "message", "timestamp")

    LEVELS = ("INFO", "WARNING", "CRITICAL")

    def __init__(self, level: str, message: str):
        assert level in self.LEVELS
        self.level     = level
        self.message   = message
        self.timestamp = time.time()

    def __repr__(self):
        return f"[{self.level}] {self.message}"


class AlertSystem:
    """
    Evaluates each frame's results and emits alert objects.

    Designed to be polled (call `check()` every frame) and returns
    a list of new Alert objects to display/log.
    """

    def __init__(
        self,
        anger_conf_thresh: float = ANGER_ALERT_THRESHOLD,
        anger_count_thresh: int  = ANGER_COUNT_THRESHOLD,
        cooldown_seconds: float  = ALERT_COOLDOWN_SECONDS,
    ):
        self._anger_conf   = anger_conf_thresh
        self._anger_count  = anger_count_thresh
        self._cooldown     = cooldown_seconds
        self._last_fire: dict[str, float] = {}   # key → last fire time
        self._history: List[Alert] = []

    # ------------------------------------------------------------------
    def check(self, results: List[FaceResult]) -> List[Alert]:
        """
        Analyse current frame results.
        Returns list of new Alert objects (may be empty).
        """
        new_alerts: List[Alert] = []
        now = time.time()

        angry_faces = [
            r for r in results
            if r.emotion == "angry" and r.confidence >= self._anger_conf
        ]

        # ── Per-face anger alert ──────────────────────────────────────
        for r in angry_faces:
            key = f"face_anger_{r.track_id}"
            if self._cooled_down(key, now):
                msg = (
                    f"{EMOTION_EMOJIS['angry']} High anger detected "
                    f"(Face #{r.track_id}, conf={r.confidence:.0%})"
                )
                alert = Alert("WARNING", msg)
                new_alerts.append(alert)
                self._history.append(alert)
                self._last_fire[key] = now

        # ── Crowd anger alert ────────────────────────────────────────
        if len(angry_faces) >= self._anger_count:
            key = "crowd_anger"
            if self._cooled_down(key, now):
                msg = (
                    f"🚨 CROWD ANGER ALERT: {len(angry_faces)} angry "
                    f"face(s) detected simultaneously!"
                )
                alert = Alert("CRITICAL", msg)
                new_alerts.append(alert)
                self._history.append(alert)
                self._last_fire[key] = now

        # ── Happy crowd note ─────────────────────────────────────────
        happy_faces = [r for r in results if r.emotion == "happy"]
        if len(happy_faces) >= 3:
            key = "crowd_happy"
            if self._cooled_down(key, now):
                msg = f"{EMOTION_EMOJIS['happy']} Positive crowd mood: {len(happy_faces)} happy faces!"
                alert = Alert("INFO", msg)
                new_alerts.append(alert)
                self._history.append(alert)
                self._last_fire[key] = now

        return new_alerts

    # ------------------------------------------------------------------
    def _cooled_down(self, key: str, now: float) -> bool:
        last = self._last_fire.get(key, 0.0)
        return (now - last) >= self._cooldown

    @property
    def history(self) -> List[Alert]:
        """All alerts ever raised in this session."""
        return list(self._history)

    def last_n(self, n: int = 5) -> List[Alert]:
        return list(self._history[-n:])

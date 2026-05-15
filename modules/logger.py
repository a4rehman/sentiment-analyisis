# ============================================================
#  modules/logger.py – Emotion Event Logger (CSV)
# ============================================================
"""
Thread-safe CSV logger.  Every detected face-emotion event is appended
to a timestamped CSV file in data/logs/.
"""

from __future__ import annotations
import csv
import logging
import os
import threading
import time
from datetime import datetime
from typing import List

from config import LOGS_DIR, LOG_FILENAME, LOG_COLUMNS
from modules.detector import FaceResult

_logger = logging.getLogger(__name__)


class EmotionLogger:
    """Thread-safe emotion event CSV logger."""

    def __init__(self, session_name: str = ""):
        ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"{ts}_{session_name}_{LOG_FILENAME}" if session_name else f"{ts}_{LOG_FILENAME}"
        self._path   = os.path.join(LOGS_DIR, fname)
        self._lock   = threading.Lock()
        self._buffer: List[dict] = []
        self._flush_interval = 2.0   # seconds
        self._last_flush     = time.time()
        self._initialised    = False
        self._init_csv()

    # ------------------------------------------------------------------
    def _init_csv(self):
        with open(self._path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
            writer.writeheader()
        self._initialised = True
        _logger.info("CSV log: %s", self._path)

    # ------------------------------------------------------------------
    def log(self, results: List[FaceResult], source: str = "webcam"):
        """Buffer emotion results for writing."""
        now = datetime.now().isoformat(timespec="milliseconds")
        rows = []
        for r in results:
            x, y, w, h = r.bbox
            rows.append({
                "timestamp":  now,
                "track_id":   r.track_id,
                "emotion":    r.emotion,
                "confidence": r.confidence,
                "face_x": x, "face_y": y,
                "face_w": w, "face_h": h,
                "source":     source,
            })
        with self._lock:
            self._buffer.extend(rows)

        # Periodic flush
        if time.time() - self._last_flush >= self._flush_interval:
            self.flush()

    # ------------------------------------------------------------------
    def flush(self):
        """Write buffered rows to disk."""
        with self._lock:
            if not self._buffer:
                return
            rows          = self._buffer[:]
            self._buffer  = []
            self._last_flush = time.time()

        try:
            with open(self._path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
                writer.writerows(rows)
        except Exception as exc:
            _logger.error("CSV flush error: %s", exc)

    # ------------------------------------------------------------------
    @property
    def log_path(self) -> str:
        return self._path

    def __del__(self):
        try:
            self.flush()
        except Exception:
            pass


# ─── Helper: load all log files ──────────────────────────────────────────────
def load_all_logs() -> "pd.DataFrame":
    """Return a combined DataFrame from all CSV log files."""
    import pandas as pd
    files = [
        os.path.join(LOGS_DIR, f)
        for f in os.listdir(LOGS_DIR)
        if f.endswith(".csv")
    ]
    if not files:
        return pd.DataFrame(columns=LOG_COLUMNS)
    dfs = []
    for fp in files:
        try:
            df = pd.read_csv(fp, parse_dates=["timestamp"])
            df["log_file"] = os.path.basename(fp)
            dfs.append(df)
        except Exception as exc:
            _logger.warning("Could not read %s: %s", fp, exc)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=LOG_COLUMNS)

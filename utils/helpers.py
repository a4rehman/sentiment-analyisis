# ============================================================
#  utils/helpers.py – Miscellaneous Helpers
# ============================================================

from __future__ import annotations
import os
import tempfile
import numpy as np
import cv2
import logging
from typing import Optional

_logger = logging.getLogger(__name__)


def save_snapshot(frame_bgr: np.ndarray, out_dir: str,
                  prefix: str = "snapshot") -> str:
    """Save a single frame as JPEG and return the path."""
    from datetime import datetime
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    name = f"{prefix}_{ts}.jpg"
    path = os.path.join(out_dir, name)
    cv2.imwrite(path, frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return path


def bytes_to_frame(data: bytes) -> Optional[np.ndarray]:
    """Decode image bytes to a BGR numpy array."""
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def frame_to_jpeg_bytes(frame_bgr: np.ndarray, quality: int = 85) -> bytes:
    """Encode a BGR frame to JPEG bytes."""
    _, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


def save_uploaded_video(uploaded_file) -> str:
    """Save a Streamlit UploadedFile to a temp file and return the path."""
    suffix = os.path.splitext(uploaded_file.name)[-1] or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.read())
    tmp.flush()
    tmp.close()
    return tmp.name


def list_log_files(logs_dir: str):
    """Return sorted list of CSV log files."""
    files = [
        f for f in os.listdir(logs_dir)
        if f.endswith(".csv")
    ]
    return sorted(files, reverse=True)


def sanitize_rtsp_url(url: str) -> str:
    """Basic RTSP URL sanitisation."""
    url = url.strip()
    if not url.startswith(("rtsp://", "http://", "https://", "rtsps://")):
        _logger.warning("Unusual stream URL: %s", url)
    return url

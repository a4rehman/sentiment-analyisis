# ============================================================
#  utils/drawing.py – OpenCV Annotation Utilities
# ============================================================
"""
Drawing primitives for overlaying face boxes, emotion labels,
confidence bars, and the stats HUD on video frames.
"""

from __future__ import annotations
import cv2
import numpy as np
from typing import List

from config import EMOTION_COLORS, EMOTION_EMOJIS, EMOTIONS


# ─── Colour helpers ───────────────────────────────────────────────────────────
def _alpha_rect(img: np.ndarray, x1, y1, x2, y2,
                color, alpha: float = 0.35) -> np.ndarray:
    """Draw a semi-transparent filled rectangle."""
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


def _put_text_with_shadow(img, text, pos, font_scale=0.55, color=(255,255,255),
                           thickness=1):
    """Text with a dark drop-shadow for readability."""
    x, y = pos
    # Shadow
    cv2.putText(img, text, (x + 1, y + 1), cv2.FONT_HERSHEY_DUPLEX,
                font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    # Main text
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_DUPLEX,
                font_scale, color, thickness, cv2.LINE_AA)


# ─── Per-face annotation ──────────────────────────────────────────────────────
def draw_face_box(img: np.ndarray, result) -> np.ndarray:
    """Draw bounding box, label, confidence bar for one face."""
    from modules.detector import FaceResult
    x, y, w, h = result.bbox
    if w <= 0 or h <= 0:
        return img

    color   = EMOTION_COLORS.get(result.emotion, (200, 200, 200))
    emoji   = EMOTION_EMOJIS.get(result.emotion, "")
    label   = f"#{result.track_id} {result.emotion.upper()} {result.confidence:.0%}"
    conf    = result.confidence

    # ── Bounding box ─────────────────────────────────────────────
    # Corner decorations (L-shaped corners instead of full rectangle)
    clen = min(w, h) // 5
    pts  = [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]
    dirs = [(1, 1), (-1, 1), (1, -1), (-1, -1)]
    for (px, py), (dx, dy) in zip(pts, dirs):
        cv2.line(img, (px, py), (px + dx * clen, py), color, 2, cv2.LINE_AA)
        cv2.line(img, (px, py), (px, py + dy * clen), color, 2, cv2.LINE_AA)

    # Light rectangle outline
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 1, cv2.LINE_AA)

    # ── Label background ─────────────────────────────────────────
    label_h = 26
    label_y = max(y - label_h - 2, 0)
    img = _alpha_rect(img, x, label_y, x + w, label_y + label_h, (20, 20, 40), 0.6)

    _put_text_with_shadow(img, label, (x + 4, label_y + 18), 0.45, color)

    # ── Confidence bar ───────────────────────────────────────────
    bar_y  = y + h + 4
    bar_h  = 6
    if bar_y + bar_h < img.shape[0]:
        cv2.rectangle(img, (x, bar_y), (x + w, bar_y + bar_h), (50, 50, 50), -1)
        fill_w = int(w * conf)
        cv2.rectangle(img, (x, bar_y), (x + fill_w, bar_y + bar_h), color, -1)

    # ── Mini score bars ──────────────────────────────────────────
    scores = result.all_scores
    if scores:
        _draw_mini_scores(img, x, bar_y + bar_h + 4, w, scores)

    return img


def _draw_mini_scores(img, x, y_start, width, scores):
    """Small per-emotion probability bars next to the face box."""
    bar_h  = 4
    gap    = 2
    avail  = img.shape[0] - y_start
    for i, emo in enumerate(EMOTIONS):
        vy = y_start + i * (bar_h + gap)
        if vy + bar_h >= img.shape[0]:
            break
        s   = float(scores.get(emo, 0.0))
        col = EMOTION_COLORS.get(emo, (180, 180, 180))
        cv2.rectangle(img, (x, vy), (x + width, vy + bar_h), (40, 40, 40), -1)
        fw  = int(width * s)
        cv2.rectangle(img, (x, vy), (x + fw, vy + bar_h), col, -1)


# ─── Multi-face draw ──────────────────────────────────────────────────────────
def draw_results(img: np.ndarray, results) -> np.ndarray:
    for r in results:
        img = draw_face_box(img, r)
    return img


# ─── HUD overlay ─────────────────────────────────────────────────────────────
def draw_hud(img: np.ndarray, fps: float, frame_n: int, results, stats) -> np.ndarray:
    """Top-right stats HUD."""
    h_img, w_img = img.shape[:2]

    # HUD panel
    panel_w = 240
    panel_h = 130
    px      = w_img - panel_w - 10
    py      = 10
    img = _alpha_rect(img, px, py, px + panel_w, py + panel_h, (10, 10, 30), 0.55)

    # FPS
    _put_text_with_shadow(img, f"FPS : {fps:5.1f}", (px + 8, py + 22), 0.5, (0, 220, 255))
    # Frame count
    _put_text_with_shadow(img, f"Frame: {frame_n}", (px + 8, py + 42), 0.45, (180, 180, 180))
    # Face count
    _put_text_with_shadow(img, f"Faces: {len(results)}", (px + 8, py + 62), 0.5, (100, 255, 180))
    # Dominant emotion
    dom = stats.dominant_emotion() or "—"
    _put_text_with_shadow(img, f"Mood : {dom}", (px + 8, py + 82), 0.5, (255, 200, 0))
    # Total events
    _put_text_with_shadow(img, f"Events: {stats.total_events()}", (px + 8, py + 102), 0.45, (160, 160, 160))
    # Duration
    _put_text_with_shadow(img, f"Time : {stats.session_duration()}", (px + 8, py + 120), 0.45, (160, 160, 160))

    return img

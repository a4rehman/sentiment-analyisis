# ============================================================
#  config.py – Central configuration for Emotion Detection
# ============================================================

import os

# ─── Paths ──────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")
LOGS_DIR        = os.path.join(DATA_DIR, "logs")
SNAPSHOTS_DIR   = os.path.join(DATA_DIR, "snapshots")
REPORTS_DIR     = os.path.join(BASE_DIR, "reports")
ASSETS_DIR      = os.path.join(BASE_DIR, "assets")

for d in [LOGS_DIR, SNAPSHOTS_DIR, REPORTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ─── Detection Settings ──────────────────────────────────────
DETECTION_BACKEND   = "opencv"          # opencv | ssd | dlib | mtcnn | retinaface
EMOTION_MODEL       = "fer"             # "fer" | "deepface"
DEEPFACE_MODEL      = "Emotion"         # Used when EMOTION_MODEL="deepface"
MIN_FACE_SIZE       = 30               # Minimum face size in pixels
SCALE_FACTOR        = 1.1
MIN_NEIGHBORS       = 5
ENFORCE_DETECTION   = False            # DeepFace strict mode

# ─── Emotion Labels & Colors ─────────────────────────────────
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

EMOTION_COLORS = {
    "angry":    (0,   30,  255),   # Red
    "disgust":  (0,  140,  255),   # Orange
    "fear":     (180,  0,  180),   # Purple
    "happy":    (0,  220,   90),   # Green
    "neutral":  (200, 200, 200),   # Gray
    "sad":      (255, 140,   0),   # Blue-Orange
    "surprise": (0,  220,  255),   # Cyan
}

EMOTION_EMOJIS = {
    "angry":    "😡",
    "disgust":  "🤢",
    "fear":     "😨",
    "happy":    "😊",
    "neutral":  "😐",
    "sad":      "😢",
    "surprise": "😲",
}

# ─── Alert Settings ──────────────────────────────────────────
ANGER_ALERT_THRESHOLD   = 0.70   # Confidence threshold for anger alert
ANGER_COUNT_THRESHOLD   = 2      # Number of angry faces to trigger crowd alert
ALERT_COOLDOWN_SECONDS  = 5      # Seconds between repeated alerts

# ─── Tracking Settings ───────────────────────────────────────
MAX_DISAPPEARED         = 30     # Frames before a track is dropped
IOU_THRESHOLD           = 0.3    # IoU threshold for assignment

# ─── Performance Settings ────────────────────────────────────
WEBCAM_WIDTH        = 1280
WEBCAM_HEIGHT       = 720
PROCESS_EVERY_N     = 2          # Run detection every N frames (1 = every frame)
RESIZE_FOR_DETECT   = 640        # Downscale width for faster detection

# ─── Logging ─────────────────────────────────────────────────
LOG_FILENAME        = "emotion_log.csv"
LOG_COLUMNS         = ["timestamp", "track_id", "emotion", "confidence",
                        "face_x", "face_y", "face_w", "face_h", "source"]

# ─── Report Settings ─────────────────────────────────────────
REPORT_CHART_DPI    = 150
REPORT_FONT_SIZE    = 10

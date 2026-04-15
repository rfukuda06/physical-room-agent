"""
Central configuration for the Newton-for-a-Room agent.

Loads API keys and Kasa credentials from .env. Defines device IPs, thresholds,
and zone definitions.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root
load_dotenv(Path(__file__).parent / ".env")

# -- API keys --
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
CLAUDE_API_KEY = os.environ["CLAUDE_API_KEY"]

# -- Kasa plug credentials (for KLAP auth on KP125M) --
KASA_USERNAME = os.environ["KASA_USERNAME"]
KASA_PASSWORD = os.environ["KASA_PASSWORD"]

# -- Kasa plug IPs (captured from `kasa discover`) --
# NOTE: IPs can change when your hotspot restarts. We'll re-discover at startup
# and only fall back to these if discovery fails.
LAMP_PLUG_ALIAS = "light"
FAN_PLUG_ALIAS = "fan"
LAMP_PLUG_IP_HINT = "172.20.10.2"
FAN_PLUG_IP_HINT = "172.20.10.3"

# -- Camera / audio --
CAMERA_INDEX = 0
AUDIO_DEVICE_INDEX = 0  # MacBook Air Microphone

# -- Camera capture vs. buffer rates --
# We capture live at full res/fps so YOLO's tracker gets crisp input,
# but store a downsampled copy in the ring buffer (much cheaper RAM-wise,
# and Gemini/Claude downsample internally anyway).
CAMERA_CAPTURE_WIDTH = 1280
CAMERA_CAPTURE_HEIGHT = 720
CAMERA_CAPTURE_FPS = 30

BUFFER_FRAME_WIDTH = 1280
BUFFER_FRAME_HEIGHT = 720
BUFFER_FPS = 10  # every 3rd capture frame goes into the buffer

# -- Perception thresholds --
FRAME_BUFFER_SECONDS = 10
AUDIO_SAMPLE_RATE = 16000  # YAMNet expects 16 kHz
AUDIO_WINDOW_SECONDS = 1.0
YAMNET_CLASSIFY_INTERVAL_SECONDS = 0.5
YAMNET_MIN_CONFIDENCE = 0.3
YAMNET_PERSISTENCE_WINDOWS = 2  # class must persist N windows to report

# -- YOLO engine --
# yolo26n-pose handles person detection + 17-keypoint pose in one forward pass.
# Chosen over yolo11n-pose because on CPU/MPS (no NVIDIA GPU here) it is ~30%
# faster AND slightly more accurate, with an improved pose head (RLE) that
# stabilizes keypoints under occlusion (e.g. sitting behind a desk).
YOLO_MODEL = "yolo26n-pose.pt"
YOLO_IMGSZ = 640            # inference resolution the model was trained for
YOLO_CONF = 0.35            # min detection confidence to keep a box
YOLO_IOU = 0.5              # NMS IoU threshold (ignored by e2e head; set for safety)
YOLO_TRACKER = "botsort.yaml"   # Ultralytics' shipped BoT-SORT config
YOLO_DEVICE = "mps"         # Apple Silicon GPU; falls back to "cpu" automatically
YOLO_INFER_EVERY_N_FRAMES = 1   # bump to 2 if CPU is saturated

# -- Baselines / calibration --
CALIBRATION_SECONDS = 300  # 5 min

# -- Zones (pixel polygons in the camera frame) --
# Each zone is a list of (x, y) vertices in pixel space, in order around the
# polygon. Zones are drawn on the FLOOR, not on the object — e.g., the
# walkable region in front of the desk, not the desk surface itself. See the
# module docstring in perception/zone_map.py for why the ground-plane
# assumption makes foot-pixel queries work from a single camera.
#
# Populate by running the click-to-define tool against your live camera:
#     python -m perception.zone_map
# It prints a ready-to-paste ZONES block. See SETUP.md §4c for the walkthrough.
ZONES: dict[str, list[tuple[int, int]]] = {
    
    # "desk":  [(412, 603), (887, 598), (901, 842), (388, 847)],
    # "door":  [(1055, 420), (1260, 418), (1258, 880), (1050, 885)],
    # "couch": [(120, 700), (610, 695), (605, 940), (115, 945)],
}

# -- LLM toggles (useful during dev) --
OBSERVER_ENABLED = True   # Gemini Flash (Beat 1)
REASONER_ENABLED = True   # Claude Sonnet (Beat 2) — disable to save cost

# -- Reasoner routing policy (hybrid) --
# Event types in this set always escalate to the Reasoner regardless of what
# the Observer says. Everything else is Observer-triaged via the escalate flag.
REASONER_ALWAYS: set[str] = {
    "new_person",
    "lost_person",
    "unusual_sound_class",    # glass_break, alarm, scream, etc.
    "power_anomaly",
    "security_event",
    "periodic_refresh_hourly",  # guaranteed hourly full-reasoning pass
}

# -- Output toggles --
TTS_ENABLED = True        # speak narrations aloud; turn off during development
TEXT_FEED_ENABLED = True  # always show narrations on the dashboard reasoning feed

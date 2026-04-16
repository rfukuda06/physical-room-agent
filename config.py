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
AUDIO_DEVICE_INDEX = 1  # MacBook Air Microphone

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
YAMNET_PERSISTENCE_WINDOWS = 1  # class must persist N windows to report

# -- Audio spike detection --
AUDIO_SPIKE_DB_THRESHOLD = 25.0     # dB above rolling mean to trigger audio_spike
AUDIO_SPIKE_COOLDOWN_SECONDS = 1.5  # suppress re-fire for this long after a spike
AUDIO_DB_ROLLING_WINDOW_SECONDS = 30  # seconds of dB history for rolling baseline

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

# -- Event detector (Day 1 Block 5) --
# These govern how raw YOLO per-frame output is condensed into discrete events
# (new_person, lost_person, pose_change, zone_transition). The detector runs
# on every YoloResult; these thresholds decide what counts as a "real" change
# vs. frame-to-frame noise. Tuned against a 30 FPS capture pipeline — bump
# the *_FRAMES constants if you lower YOLO_INFER_EVERY_N_FRAMES.
#
# Why each one exists:
#   * KP_MIN_CONF      — keypoints below this are unreliable; pose classifier
#                        returns "unknown" and we hold last_pose rather than
#                        flicker on a bad frame.
#   * POSE_HYSTERESIS  — sitting/standing/walking must persist N frames before
#                        we emit pose_change. Cheap hysteresis, prevents one
#                        bad keypoint frame from firing a spurious transition.
#   * ZONE_DWELL       — a person stepping briefly across a zone boundary
#                        shouldn't count; zone must hold for N frames.
#   * LOST_GRACE       — BoT-SORT occasionally drops a track for 1–2 frames
#                        mid-stream (occlusion, motion blur). Waiting ~1s
#                        avoids false lost_person/new_person churn for the
#                        *same* physical person within one track lifetime.
#   * WALK_MIN_DX_PX   — center-x displacement per frame above which we call
#                        a standing person "walking". ~15px at 30 FPS ≈ 0.5
#                        body-widths/second of lateral motion, which matches
#                        a normal indoor walking pace for a 1280-wide frame.
EVENT_POSE_KP_MIN_CONF = 0.5
EVENT_POSE_HYSTERESIS_FRAMES = 5     # ~0.17s at 30 FPS
EVENT_ZONE_DWELL_FRAMES = 5          # ~0.17s — ignore drive-by boundary crosses
EVENT_LOST_PERSON_GRACE_FRAMES = 30  # ~1.0s — tracker occlusion tolerance
EVENT_WALK_MIN_DX_PX = 15            # per-frame center-x delta threshold

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
    "door": [(1052, 419), (953, 454), (1063, 541), (1220, 514), (1236, 494), (1087, 414)],
    "desks": [(624, 713), (491, 492), (271, 511), (291, 714)],
    "bed": [(165, 342), (177, 471), (688, 506), (613, 391)],
    "closet": [(737, 381), (822, 494), (1062, 439), (1030, 387)],
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

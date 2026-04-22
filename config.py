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
AUDIO_DEVICE_NAME = "MacBook Air Microphone"  # matched by substring; index resolved at startup

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
#   * WALK_MIN_DIST_PX — 2D center-point displacement per frame above which
#                        we call a standing person "walking". Uses both x and
#                        y so walking toward/away from the camera registers.
#                        8px at 30 FPS on a 1280-wide frame catches normal
#                        indoor walking in any direction.
#   * WALK_HOLD_FRAMES — once walking is confirmed, hold the "walking" label
#                        for this many frames even if motion drops below
#                        threshold. Walking is naturally uneven (stride pauses,
#                        turns) so without this, walking flickers to standing
#                        and back on every slow frame.
EVENT_POSE_KP_MIN_CONF = 0.5
EVENT_POSE_HYSTERESIS_FRAMES = 8     # ~0.27s at 30 FPS
EVENT_ZONE_DWELL_FRAMES = 5          # ~0.17s — ignore drive-by boundary crosses
EVENT_NEW_PERSON_CONFIRM_FRAMES = 10 # ~0.33s — track must persist before new_person fires
EVENT_LOST_PERSON_GRACE_FRAMES = 60  # ~2.0s — tracker occlusion tolerance
EVENT_WALK_MIN_DIST_PX = 8           # per-frame 2D center-point distance threshold
EVENT_WALK_HOLD_FRAMES = 15          # ~0.5s — stay "walking" through brief pauses
EVENT_SIT_HOLD_FRAMES = 5            # ~0.17s — stay "sitting" through brief jitter
EVENT_POSE_COOLDOWN_S = 3.0          # max 1 pose_change per 3s per track (catch-up on expiry)
EVENT_ZONE_COOLDOWN_S = 2.0          # max 1 zone_transition per 2s per track (catch-up on expiry)

# -- Baselines / calibration --
CALIBRATION_SECONDS = 30               # ~30s gives enough samples (see LEARNING.md)
CALIBRATION_AMBIENT_CLASS_MIN_RATIO = 0.3   # class must appear in >=30% of windows
CALIBRATION_EXCLUDE_SPEECH_FROM_FLOOR = True  # skip speech dB from noise floor calc

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
    "door": [(964, 338), (875, 373), (1008, 469), (1112, 398)],
}

# -- LLM toggles (useful during dev) --
OBSERVER_ENABLED = True   # Gemini Flash (Beat 1)
REASONER_ENABLED = True   # Claude Sonnet (Beat 2) — disable to save cost

# -- Observer (Gemini) settings --
GEMINI_MODEL = "gemini-2.5-flash"       # swap to "gemini-2.5-flash-lite" for cheaper/faster
OBSERVER_THINKING_BUDGET = 0            # 0 = disable thinking for low-latency factual descriptions
OBSERVER_REFRESH_INTERVAL_S = 30        # background refresh during quiet periods
OBSERVER_DEBOUNCE_S = 0.5             # batch events within this window before calling Gemini
OBSERVER_MIN_CALL_INTERVAL_S = 2.0    # minimum seconds between Gemini calls (rate limiting)
OBSERVER_TIMEOUT_S = 8.0               # max wait for Gemini response before giving up
OBSERVER_MAX_FRAMES = 3                # how many frames to send (current + N-1 prior)
OBSERVER_FRAME_QUALITY = 70            # JPEG quality for frame encoding (lower = smaller = cheaper)
OBSERVER_FRAME_MAX_DIM = 1280          # raw camera res (1280×720 = 2 tiles = 516 tokens/image)

# -- Reasoner routing policy (hybrid) --
# Event types in this set always escalate to the Reasoner regardless of what
# the Observer says. Everything else is Observer-triaged via the escalate flag.
REASONER_ALWAYS: set[str] = {
    "new_person",
    "lost_person",
    "power_anomaly",
    "security_event",
    "periodic_refresh_minutely",  # guaranteed minutely session summary
    "periodic_refresh_hourly",    # guaranteed hourly full-reasoning pass
}

# -- Reasoner (Claude Sonnet 4.6) settings --
CLAUDE_MODEL = "claude-sonnet-4-6"
REASONER_MAX_TOKENS = 1024              # generous for reasoning + narration + actions
REASONER_INCLUDE_FRAME_ALWAYS = False   # if True, always send a camera frame
REASONER_SUMMARY_INTERVAL_S = 60.0     # push a session summary to Reasoner every ~60s

# -- TTS settings --
TTS_VOICE = "en-US-AriaNeural"      # edge-tts voice; swap for demo preference

# -- Output toggles --
TTS_ENABLED = False       # speak narrations aloud; turn off during development
TEXT_FEED_ENABLED = True  # always show narrations on the dashboard reasoning feed


def resolve_audio_device(name: str = AUDIO_DEVICE_NAME) -> int:
    """
    Find the first input device whose name contains `name` (case-insensitive).
    Returns its integer index for sounddevice.
    Raises RuntimeError if not found — better to fail loudly at startup than
    silently record from the wrong mic.
    """
    import sounddevice as sd
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0 and name.lower() in d["name"].lower():
            return i
    available = [d["name"] for d in sd.query_devices() if d["max_input_channels"] > 0]
    raise RuntimeError(
        f"Audio device '{name}' not found. Available input devices: {available}"
    )

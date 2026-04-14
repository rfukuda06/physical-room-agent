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

# -- Perception thresholds --
FRAME_BUFFER_SECONDS = 30
AUDIO_SAMPLE_RATE = 16000  # YAMNet expects 16 kHz
AUDIO_WINDOW_SECONDS = 1.0
YAMNET_CLASSIFY_INTERVAL_SECONDS = 0.5
YAMNET_MIN_CONFIDENCE = 0.3
YAMNET_PERSISTENCE_WINDOWS = 2  # class must persist N windows to report

# -- Baselines / calibration --
CALIBRATION_SECONDS = 300  # 5 min

# -- Zones (pixel regions in the camera frame). Fill in after you see the feed. --
# Format: {"zone_name": (x, y, w, h)}
ZONES: dict[str, tuple[int, int, int, int]] = {
    # "desk": (100, 200, 400, 300),
    # "door": (800, 100, 200, 500),
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

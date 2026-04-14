"""
Orchestrator — top-level entry point.

Starts each layer in the right order:
  1. Load config
  2. Start camera + frame buffer
  3. Start YOLO engine
  4. Start audio + YAMNet
  5. Discover smart plugs
  6. Run calibration phase (~5 min)
  7. Enter monitoring loop: Layer 0 events → Observer → Reasoner → actions
  8. (Optionally) start the FastAPI/WebSocket server for the dashboard

Run with:  python main.py
"""

import config


def main() -> None:
    print("Newton-for-a-Room — starting...")
    print(f"Observer enabled: {config.OBSERVER_ENABLED}")
    print(f"Reasoner enabled: {config.REASONER_ENABLED}")
    print(f"Camera index: {config.CAMERA_INDEX}")
    print(f"Audio device index: {config.AUDIO_DEVICE_INDEX}")
    # TODO: wire up perception → agents → actuators


if __name__ == "__main__":
    main()

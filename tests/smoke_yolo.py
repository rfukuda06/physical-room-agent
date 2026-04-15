"""
Smoke test: webcam + YOLO detection/tracking/pose.

Run:
    python -m tests.smoke_yolo

Shows the annotated camera feed (bounding boxes, track IDs, pose skeleton)
and logs one summary line per second to the console. Press 'q' to quit.

Delegates to perception.yolo_engine._preview_main.
"""

from perception.yolo_engine import _preview_main


if __name__ == "__main__":
    _preview_main()

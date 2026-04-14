"""
Event detector — consumes YOLO output, emits structured events.

Watches per-frame YOLO state transitions and emits events like:
  new_person, lost_person, pose_change, zone_transition, object_moved.
These events trigger Layer 1 (Observer).
"""

# TODO: implement on Day 1 Block 5

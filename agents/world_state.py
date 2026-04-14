"""
Shared in-memory world model.

Structured state (entities, counts, audio, devices) updated every frame
by Layer 0. Semantic state (scene_description, activity_summary, mood)
updated by Layer 1 on each call. Baselines learned during calibration.
Ring buffer of last 50 events.

This is the single source of truth. No database — lives in RAM.
"""

# TODO: implement on Day 2 Block 1

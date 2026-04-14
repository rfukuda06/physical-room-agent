"""
Layer 1 — Observer agent (Gemini 2.0 Flash). Beat 1 of the two-beat rhythm.

Event-driven + periodic background refresh. Fast factual description of
what just happened. Receives: current frame + 2-3 buffered frames + YOLO
signals + audio (level + YAMNet top-k) + device state.

Output contract (JSON):
  {
    "narration": "short factual description, spoken aloud as Beat 1",
    "world_state_update": {...},   # semantic fields to merge into WorldState
    "escalate": true | false,       # should the Reasoner be called?
    "escalate_reason": "short string — why escalate / why not"
  }

The `escalate` flag drives the hybrid routing policy — see agents/routing.py
and ARCHITECTURE_AND_BUILD_PLAN copy.md §2.5.
"""

# TODO: implement on Day 2 Block 3

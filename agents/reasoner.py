"""
Layer 2 — Reasoner agent (Claude Sonnet). Beat 2 of the two-beat rhythm.

Runs only when the routing policy escalates (see agents/routing.py). Receives
the Observer's description + full world state + recent event history + all
sensor channels. Produces deep reasoning (intent, safety, wellness), a
thoughtful spoken follow-up, and concrete action decisions (lamp on/off,
fan on/off, alerts).

Output contract (JSON):
  {
    "narration": "thoughtful follow-up, spoken aloud as Beat 2",
    "actions": [
      {"type": "lamp", "state": "on" | "off"},
      {"type": "fan",  "state": "on" | "off"},
      {"type": "alert", "severity": "...", "message": "..."},
      ...
    ],
    "world_state_update": {...}  # deeper semantic interpretation
  }
"""

# TODO: implement on Day 2 Block 4

"""
Audio perception: mic streaming + dB monitoring + YAMNet classification.

Streams from sounddevice, computes rolling dB level, detects spikes, and
runs YAMNet on rolling 1s windows every ~500ms. Filters to a room-relevant
whitelist of ~30-50 AudioSet classes (speech, clap, door, glass_break, etc.)
with temporal smoothing (class must persist >= N windows to report).

AudioClassifier is an interface — YAMNet is the current implementation, but
CLAP/BEATs could swap in later.
"""

# TODO: implement on Day 1 Block 6

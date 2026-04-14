"""
Webcam capture + rolling 30s frame buffer.

Owns the OpenCV VideoCapture. Pushes every frame into a rolling in-memory
buffer so the Observer/Reasoner can grab prior frames when reconstructing
what led up to an event.
"""

# TODO: implement on Day 1 Block 2

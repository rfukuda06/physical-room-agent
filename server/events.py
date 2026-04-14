"""
Event bus connecting all three layers.

Layer 0 publishes events → Layer 1 subscribes → Layer 2 subscribes after.
Also fans out to the dashboard WebSocket for the reasoning feed.
"""

# TODO: implement alongside the Observer on Day 2

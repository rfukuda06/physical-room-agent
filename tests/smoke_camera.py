"""
Smoke test: live webcam + rolling buffer.

Run:
    python -m tests.smoke_camera

Controls (see preview window title):
    q = quit (prints buffer stats on exit)
    s = dump current 10s buffer to ./_debug_frames/ as JPEGs

Delegates to perception.camera._preview_main so the actual loop stays next
to the code it exercises. This file just gives it a stable, discoverable
test entrypoint under tests/.
"""

from perception.camera import _preview_main


if __name__ == "__main__":
    _preview_main()

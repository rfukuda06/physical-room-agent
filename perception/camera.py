"""
Webcam capture + rolling frame buffer.

This is the very first layer of the system: we open the webcam with OpenCV,
continuously read frames in a background thread, and keep the last ~N seconds
of frames in memory so later layers (Observer/Reasoner) can look back at what
just happened when an event fires.

--- Quick primer on the concepts in here (for first-time readers) ---

* `cv2.VideoCapture(index)` — OpenCV's handle to a camera. `.read()` returns
  `(ok, frame)`. `frame` is a NumPy array of shape (H, W, 3) in BGR order
  (yes, BGR — a historical OpenCV quirk). We'll convert to RGB only when we
  hand frames off to something that expects RGB (YOLO handles BGR natively).

* `collections.deque(maxlen=N)` — a list-like "ring buffer". When it's full
  and you append to the right, the oldest item on the left is dropped
  automatically. Perfect for "keep only the last N frames".

* Threading — `cap.read()` is a blocking I/O call (it waits for the next
  frame from the camera). If we did it inline with, say, YOLO inference,
  we'd stall the whole pipeline. Instead we run the read loop in its own
  thread so consumers (YOLO, dashboard) can pull the latest frame whenever
  they're ready, without blocking the camera. Two threads touching the same
  mutable state (`_latest_frame`, the deque) need a `threading.Lock` so we
  never read a half-updated value.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


class CameraCapture:
    """Background-thread webcam reader with a rolling downsampled buffer.

    Two separate "streams" come out of the same hardware read:
      - `latest_frame()`: the most recent raw capture frame, at full
        capture resolution/fps. This is what YOLO will consume live.
      - `buffer_snapshot()`: the last N seconds of frames at a lower
        resolution and framerate. This is what the Observer/Reasoner
        will use to reconstruct "what led up to this event".
    """

    def __init__(
        self,
        camera_index: int,
        capture_size: tuple[int, int],   # (width, height)
        capture_fps: int,
        buffer_size: tuple[int, int],    # (width, height)
        buffer_fps: int,
        buffer_seconds: int,
    ) -> None:
        self._camera_index = camera_index
        self._capture_w, self._capture_h = capture_size
        self._capture_fps = capture_fps
        self._buffer_w, self._buffer_h = buffer_size
        self._buffer_fps = buffer_fps
        self._buffer_seconds = buffer_seconds

        # How many capture frames per one buffer frame. Example: capture=30fps,
        # buffer=10fps -> subsample every 3rd frame.
        self._subsample_every = max(1, capture_fps // buffer_fps)

        # Ring buffer: holds (timestamp_monotonic, downsampled_frame).
        self._buffer: deque[tuple[float, np.ndarray]] = deque(
            maxlen=buffer_fps * buffer_seconds
        )

        self._latest_frame: Optional[np.ndarray] = None
        self._latest_ts: float = 0.0

        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Separate locks so a slow buffer reader can't block latest-frame reads.
        self._latest_lock = threading.Lock()
        self._buffer_lock = threading.Lock()

    # ---- lifecycle ----

    def start(self) -> None:
        """Open the camera and spawn the background read thread."""
        if self._thread is not None:
            raise RuntimeError("CameraCapture already started")

        # On macOS, specifying CAP_AVFOUNDATION avoids a slow fallback search
        # through other backends. Harmless if OpenCV picks it anyway.
        self._cap = cv2.VideoCapture(self._camera_index, cv2.CAP_AVFOUNDATION)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Could not open camera index {self._camera_index}. "
                "On macOS, make sure your terminal/IDE has Camera permission "
                "(System Settings → Privacy & Security → Camera)."
            )

        # These are *requests* — the driver may pick the closest supported
        # mode. We'll log what we actually got after the first frame.
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._capture_w)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._capture_h)
        self._cap.set(cv2.CAP_PROP_FPS, self._capture_fps)

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._read_loop, name="camera-capture", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the read thread to exit, join it, and release the camera."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "CameraCapture":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    # ---- consumer API ----

    def latest_frame(self) -> Optional[np.ndarray]:
        """Return the most recent raw capture frame (or None if none yet).

        Returns a *copy* so the caller is free to mutate/annotate it without
        racing the background writer.
        """
        with self._latest_lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def buffer_snapshot(
        self, seconds: Optional[float] = None
    ) -> list[tuple[float, np.ndarray]]:
        """Return a list of (timestamp, frame) pairs from the ring buffer.

        If `seconds` is None, returns everything currently held. Otherwise
        returns only frames newer than `now - seconds`. Frames are copied so
        callers can't mutate the live buffer.
        """
        cutoff = None
        if seconds is not None:
            cutoff = time.monotonic() - seconds

        with self._buffer_lock:
            if cutoff is None:
                items = list(self._buffer)
            else:
                items = [(t, f) for (t, f) in self._buffer if t >= cutoff]

        return [(t, f.copy()) for (t, f) in items]

    def buffer_stats(self) -> dict:
        """Cheap diagnostics for logging / the dev preview."""
        with self._buffer_lock:
            n = len(self._buffer)
            span = (self._buffer[-1][0] - self._buffer[0][0]) if n >= 2 else 0.0
            # Frames are uint8 HxWx3 -> nbytes ~= H*W*3
            bytes_per = (
                self._buffer[0][1].nbytes if n >= 1 else 0
            )
        return {
            "frames": n,
            "span_seconds": round(span, 2),
            "approx_mb": round(n * bytes_per / (1024 * 1024), 1),
        }

    # ---- internals ----

    def _read_loop(self) -> None:
        """Background thread body: pull frames, update latest, fill buffer."""
        assert self._cap is not None
        frame_ix = 0

        while not self._stop_event.is_set():
            ok, frame = self._cap.read()
            if not ok or frame is None:
                # Transient glitch — back off briefly and try again.
                time.sleep(0.01)
                continue

            ts = time.monotonic()

            # Always publish the newest raw frame for YOLO.
            with self._latest_lock:
                self._latest_frame = frame
                self._latest_ts = ts

            # Subsample into the ring buffer, downsized to the buffer size.
            if frame_ix % self._subsample_every == 0:
                resized = cv2.resize(
                    frame,
                    (self._buffer_w, self._buffer_h),
                    interpolation=cv2.INTER_AREA,
                )
                with self._buffer_lock:
                    self._buffer.append((ts, resized))

            frame_ix += 1


# ---------------------------------------------------------------------------
# Standalone preview mode: `python -m perception.camera`
#
# Lets us eyeball the capture before YOLO exists. Controls:
#   q  -> quit (prints buffer stats on exit)
#   s  -> save current buffer snapshot as JPEGs in ./_debug_frames/
# ---------------------------------------------------------------------------

def _preview_main() -> None:
    import config  # imported lazily so unit tests can import CameraCapture alone

    cap = CameraCapture(
        camera_index=config.CAMERA_INDEX,
        capture_size=(config.CAMERA_CAPTURE_WIDTH, config.CAMERA_CAPTURE_HEIGHT),
        capture_fps=config.CAMERA_CAPTURE_FPS,
        buffer_size=(config.BUFFER_FRAME_WIDTH, config.BUFFER_FRAME_HEIGHT),
        buffer_fps=config.BUFFER_FPS,
        buffer_seconds=config.FRAME_BUFFER_SECONDS,
    )

    print("Starting camera preview — press 'q' to quit, 's' to dump buffer.")
    cap.start()

    # Wait briefly for the first frame so the window doesn't flicker.
    for _ in range(50):
        if cap.latest_frame() is not None:
            break
        time.sleep(0.05)

    try:
        while True:
            frame = cap.latest_frame()
            if frame is not None:
                cv2.imshow("camera preview (q=quit, s=save buffer)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                snap = cap.buffer_snapshot()
                out_dir = Path("_debug_frames")
                out_dir.mkdir(exist_ok=True)
                # Clear prior dumps so numbering is fresh.
                for p in out_dir.glob("buffer_*.jpg"):
                    p.unlink()
                t0 = snap[0][0] if snap else 0.0
                for i, (t, f) in enumerate(snap):
                    fname = out_dir / f"buffer_{i:03d}_t{(t - t0):05.2f}s.jpg"
                    cv2.imwrite(str(fname), f)
                print(f"Saved {len(snap)} frames to {out_dir}/")
    finally:
        stats = cap.buffer_stats()
        cap.stop()
        cv2.destroyAllWindows()
        print(f"Buffer stats on exit: {stats}")


if __name__ == "__main__":
    _preview_main()

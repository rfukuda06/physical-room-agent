"""
Smoke test: event_detector end-to-end with real camera + YOLO + zones.

Run:
    python -m tests.smoke_event_detector

What you'll see:
    * Camera feed with YOLO boxes/IDs/skeleton AND zone polygons AND
      a rolling event log in the top-left corner.
    * One console line per event (`[event] ...`) AND a one-per-second
      status line showing active track IDs, each with its current zones
      and confirmed pose.

How to exercise every event type:
    new_person       — walk into frame.
    zone_transition  — cross from one zone polygon into another.
                       (Only fires when ZONES is populated — run
                        `python -m perception.zone_map` first if empty.)
    pose_change      — sit down, then stand back up (wait ~0.2s each way
                       for hysteresis to settle).
    lost_person      — walk fully out of frame and hold still for ~1s.
                       Walking back in will emit a *new* new_person with
                       a fresh track_id — BoT-SORT doesn't re-identify
                       across full exits. See event_detector.py docstring
                       note #3.

Keys:
    q    quit and print summary stats
"""

from __future__ import annotations

import time
from collections import deque

import cv2
import numpy as np

import config
from perception.camera import CameraCapture
from perception.yolo_engine import YoloEngine
from perception.event_detector import Event, EventDetector

# --- Overlay colors (BGR) ---
ZONE_COLOR = (0, 255, 0)        # green
LOG_BG = (0, 0, 0)              # black rectangle behind event log
LOG_FG = {                      # per-event-type colors for the log list
    "new_person":      (  0, 255,   0),
    "lost_person":     (  0,   0, 255),
    "pose_change":     (  0, 200, 255),   # amber
    "zone_transition": (255, 200,   0),   # cyan-ish
}
STATUS_FG = (255, 255, 255)     # white

# How many recent events to show on the overlay.
LOG_MAX_LINES = 8


def _fmt_event(ev: Event) -> str:
    """One-line human-readable event string for console + overlay."""
    p = ev.payload
    if ev.type == "new_person":
        return f"new_person   id={ev.track_id} zones={ev.zones} pose={p.get('initial_pose')}"
    if ev.type == "lost_person":
        return (f"lost_person  id={ev.track_id} "
                f"last_zones={p.get('last_zones')} "
                f"missed={p.get('frames_missing')}f")
    if ev.type == "pose_change":
        return (f"pose_change  id={ev.track_id} "
                f"{p.get('from_pose')} -> {p.get('to_pose')} zones={ev.zones}")
    if ev.type == "zone_transition":
        return (f"zone_trans   id={ev.track_id} "
                f"{p.get('from_zones')} -> {p.get('to_zones')}")
    return f"{ev.type} id={ev.track_id} payload={p}"


def _draw_overlay(
    frame: np.ndarray,
    recent_events: deque[tuple[float, Event]],
    detector: EventDetector,
) -> np.ndarray:
    """Draw zones + per-track status + the recent event log onto `frame`."""
    out = frame

    # Zone polygons (read directly from config so new zones picked up
    # without restart — same pattern as smoke_zone_map).
    for name, pts in config.ZONES.items():
        if len(pts) < 3:
            continue
        poly = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [poly], isClosed=True, color=ZONE_COLOR, thickness=2)
        cv2.putText(out, name, pts[0], cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, ZONE_COLOR, 2, cv2.LINE_AA)

    # Rolling event log — top-left corner, translucent black backing so
    # it's readable against a bright frame.
    if recent_events:
        h_line = 22
        n = min(LOG_MAX_LINES, len(recent_events))
        box_h = n * h_line + 14
        overlay = out.copy()
        cv2.rectangle(overlay, (5, 5), (620, 5 + box_h), LOG_BG, -1)
        cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)

        # Show newest-at-top.
        for i, (t, ev) in enumerate(list(recent_events)[-n:][::-1]):
            color = LOG_FG.get(ev.type, STATUS_FG)
            text = f"{t:6.1f}s  {_fmt_event(ev)}"
            y = 25 + i * h_line
            cv2.putText(out, text, (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # Per-track status strip along the bottom.
    tids = detector.active_track_ids()
    strip = "tracks: " + (
        "  ".join(f"id={tid}:{detector.pose_for(tid)}" for tid in tids) if tids
        else "(none)"
    )
    cv2.putText(out, strip, (10, out.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, STATUS_FG, 1, cv2.LINE_AA)

    return out


def main() -> None:
    camera = CameraCapture(
        camera_index=config.CAMERA_INDEX,
        capture_size=(config.CAMERA_CAPTURE_WIDTH, config.CAMERA_CAPTURE_HEIGHT),
        capture_fps=config.CAMERA_CAPTURE_FPS,
        buffer_size=(config.BUFFER_FRAME_WIDTH, config.BUFFER_FRAME_HEIGHT),
        buffer_fps=config.BUFFER_FPS,
        buffer_seconds=config.FRAME_BUFFER_SECONDS,
    )
    engine = YoloEngine(
        camera=camera,
        model_path=config.YOLO_MODEL,
        imgsz=config.YOLO_IMGSZ,
        conf=config.YOLO_CONF,
        iou=config.YOLO_IOU,
        tracker=config.YOLO_TRACKER,
        device=config.YOLO_DEVICE,
        infer_every_n_frames=config.YOLO_INFER_EVERY_N_FRAMES,
    )
    detector = EventDetector()

    n_zones = len(config.ZONES)
    zone_summary = (
        list(config.ZONES.keys()) if n_zones
        else "(none — zone_transition events will never fire; "
             "run `python -m perception.zone_map` first if you want them)"
    )
    print(f"event_detector smoke test — {n_zones} zone(s) loaded: {zone_summary}")
    print("Press 'q' in the window to quit.")
    print("Try: walk in, cross zones, sit, stand, walk out, walk back in.\n")

    camera.start()
    for _ in range(100):
        if camera.latest_frame() is not None:
            break
        time.sleep(0.05)
    engine.start()

    # Rolling log of (elapsed_seconds, Event) for the on-screen overlay.
    recent: deque[tuple[float, Event]] = deque(maxlen=40)
    event_counts: dict[str, int] = {}
    t_start = time.monotonic()
    last_status = 0.0

    try:
        while True:
            result = engine.latest_result()
            new_events = detector.tick(result)
            for ev in new_events:
                elapsed = ev.ts - t_start
                recent.append((elapsed, ev))
                event_counts[ev.type] = event_counts.get(ev.type, 0) + 1
                print(f"[event] t={elapsed:6.2f}s  {_fmt_event(ev)}")

            if result is not None and result.annotated_frame is not None:
                frame = _draw_overlay(result.annotated_frame, recent, detector)
                cv2.imshow("event_detector smoke (q=quit)", frame)

                # One status line per second — parallel to smoke_zone_map.
                now = time.monotonic()
                if now - last_status >= 1.0:
                    tids = detector.active_track_ids()
                    tid_strs = [
                        f"id={tid}({detector.pose_for(tid)})" for tid in tids
                    ]
                    print(f"[status] t={now - t_start:6.2f}s "
                          f"tracks={tid_strs or '(none)'}")
                    last_status = now

            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        engine.stop()
        camera.stop()
        cv2.destroyAllWindows()
        total = sum(event_counts.values())
        print(f"\nEvents emitted this run ({total} total): {event_counts}")


if __name__ == "__main__":
    main()

"""
Orchestrator — top-level entry point.

Starts each layer in the right order:
  1. Load config
  2. Start camera + frame buffer
  3. Start YOLO engine
  4. Start audio + YAMNet
  5. Discover smart plugs (TODO — Block 7)
  6. Run calibration phase (TODO — Day 2 Block 2)
  7. Enter monitoring loop: Layer 0 events → Observer → Reasoner → actions
  8. (Optionally) start the FastAPI/WebSocket server for the dashboard

Run with:  python main.py

Block 8 (Day 1): Smoke test — all Layer 0 signals flowing, events printing
to console. Camera + YOLO + audio events merge into a single stream. No
Observer/Reasoner yet; this loop is the skeleton for the full pipeline.
"""

from __future__ import annotations

import logging
import signal
import sys
import time

import cv2
import numpy as np

import config
from perception.camera import CameraCapture
from perception.yolo_engine import YoloEngine
from perception.event_detector import Event, EventDetector
from perception.audio import AudioMonitor

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Formatting helpers (reused from the individual smoke tests, but now unified)
# ---------------------------------------------------------------------------

def _fmt_event(ev: Event) -> str:
    """One-line human-readable event string for the console."""
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
    # Audio events
    if ev.type == "unusual_sound_class":
        return (f"sound_class  {p.get('class_name')} "
                f"conf={p.get('confidence', 0):.2f} "
                f"db={p.get('db_level', 0):.1f}")
    if ev.type == "audio_spike":
        return (f"audio_spike  current={p.get('current_db', 0):.1f}dB "
                f"baseline={p.get('baseline_db', 0):.1f}dB "
                f"delta={p.get('delta_db', 0):.1f}dB")
    if ev.type == "speech_start":
        return f"speech_start conf={p.get('confidence', 0):.2f} db={p.get('db_level', 0):.1f}"
    if ev.type == "speech_end":
        return f"speech_end   duration={p.get('duration_seconds', 0):.1f}s"
    return f"{ev.type} id={ev.track_id} payload={p}"


# ---------------------------------------------------------------------------
# CV2 overlay — unified version showing YOLO + zones + event log + audio
# ---------------------------------------------------------------------------

_ZONE_COLOR = (0, 255, 0)
_LOG_BG = (0, 0, 0)
_LOG_FG = {
    "new_person":          (  0, 255,   0),
    "lost_person":         (  0,   0, 255),
    "pose_change":         (  0, 200, 255),
    "zone_transition":     (255, 200,   0),
    "unusual_sound_class": (255, 100, 255),
    "audio_spike":         (  0, 180, 255),
    "speech_start":        (200, 200,   0),
    "speech_end":          (200, 200,   0),
}
_STATUS_FG = (255, 255, 255)
_LOG_MAX_LINES = 10


def _draw_overlay(
    frame: np.ndarray,
    recent_events: list[tuple[float, Event]],
    detector: EventDetector,
    audio: AudioMonitor,
) -> np.ndarray:
    """Draw zones, event log, track status, and audio state onto the frame."""
    out = frame

    # Zone polygons
    for name, pts in config.ZONES.items():
        if len(pts) < 3:
            continue
        poly = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [poly], isClosed=True, color=_ZONE_COLOR, thickness=2)
        cv2.putText(out, name, pts[0], cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, _ZONE_COLOR, 2, cv2.LINE_AA)

    # Rolling event log — top-left, translucent black background
    if recent_events:
        h_line = 22
        n = min(_LOG_MAX_LINES, len(recent_events))
        box_h = n * h_line + 14
        overlay = out.copy()
        cv2.rectangle(overlay, (5, 5), (700, 5 + box_h), _LOG_BG, -1)
        cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)
        for i, (t, ev) in enumerate(recent_events[-n:][::-1]):
            color = _LOG_FG.get(ev.type, _STATUS_FG)
            text = f"{t:6.1f}s  {_fmt_event(ev)}"
            y = 25 + i * h_line
            cv2.putText(out, text, (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    # Bottom bar: track status (left) + audio status (right)
    h, w = out.shape[:2]

    # Track status
    tids = detector.active_track_ids()
    track_str = "tracks: " + (
        "  ".join(f"id={tid}:{detector.pose_for(tid)}" for tid in tids)
        if tids else "(none)"
    )
    cv2.putText(out, track_str, (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, _STATUS_FG, 1, cv2.LINE_AA)

    # Audio status
    audio_st = audio.latest_state()
    if audio_st is not None:
        audio_str = (
            f"audio: {audio_st.audio_level_db:.0f}dB  "
            f"{'SPEECH' if audio_st.speech_active else audio_st.dominant_class}"
        )
        # Right-align
        (tw, _), _ = cv2.getTextSize(audio_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(out, audio_str, (w - tw - 10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, _STATUS_FG, 1, cv2.LINE_AA)

    return out


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-24s  %(message)s",
        datefmt="%H:%M:%S",
    )

    print("=" * 60)
    print("  Newton-for-a-Room — Layer 0 smoke test (Block 8)")
    print("=" * 60)
    print(f"  Camera index:       {config.CAMERA_INDEX}")
    print(f"  Audio device index: {config.AUDIO_DEVICE_INDEX}")
    print(f"  Zones loaded:       {list(config.ZONES.keys()) or '(none)'}")
    print(f"  Observer enabled:   {config.OBSERVER_ENABLED}  (not wired yet)")
    print(f"  Reasoner enabled:   {config.REASONER_ENABLED}  (not wired yet)")
    print()
    print("  Press 'q' in the video window to quit.")
    print("=" * 60)
    print()

    # -- Step 1: Start camera --
    camera = CameraCapture(
        camera_index=config.CAMERA_INDEX,
        capture_size=(config.CAMERA_CAPTURE_WIDTH, config.CAMERA_CAPTURE_HEIGHT),
        capture_fps=config.CAMERA_CAPTURE_FPS,
        buffer_size=(config.BUFFER_FRAME_WIDTH, config.BUFFER_FRAME_HEIGHT),
        buffer_fps=config.BUFFER_FPS,
        buffer_seconds=config.FRAME_BUFFER_SECONDS,
    )
    camera.start()
    log.info("Camera started — waiting for first frame...")

    # Wait until camera produces at least one frame
    for _ in range(100):
        if camera.latest_frame() is not None:
            break
        time.sleep(0.05)
    else:
        log.error("Camera did not produce a frame within 5 seconds — aborting")
        camera.stop()
        sys.exit(1)
    log.info("Camera ready")

    # -- Step 2: Start YOLO engine --
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
    engine.start()
    log.info("YOLO engine started (model=%s, device=%s)", config.YOLO_MODEL, config.YOLO_DEVICE)

    # -- Step 3: Start audio monitor --
    audio = AudioMonitor(
        device_index=config.AUDIO_DEVICE_INDEX,
        sample_rate=config.AUDIO_SAMPLE_RATE,
        window_seconds=config.AUDIO_WINDOW_SECONDS,
        classify_interval=config.YAMNET_CLASSIFY_INTERVAL_SECONDS,
        min_confidence=config.YAMNET_MIN_CONFIDENCE,
        persistence_windows=config.YAMNET_PERSISTENCE_WINDOWS,
        spike_db_threshold=config.AUDIO_SPIKE_DB_THRESHOLD,
        spike_cooldown=config.AUDIO_SPIKE_COOLDOWN_SECONDS,
        db_rolling_window=config.AUDIO_DB_ROLLING_WINDOW_SECONDS,
    )
    audio.start()
    log.info("Audio monitor started (device=%d)", config.AUDIO_DEVICE_INDEX)

    # -- Step 4: Create event detector --
    detector = EventDetector()

    # -- Step 5 (skipped): Smart plugs -- Block 7 deferred
    # -- Step 6 (skipped): Calibration -- Day 2 Block 2

    # -- Step 7: Main monitoring loop --
    recent_events: list[tuple[float, Event]] = []  # (elapsed_sec, event)
    event_counts: dict[str, int] = {}
    t_start = time.monotonic()
    last_status_print = 0.0

    # Graceful Ctrl+C
    shutdown = False

    def _signal_handler(sig, frame):
        nonlocal shutdown
        shutdown = True

    signal.signal(signal.SIGINT, _signal_handler)

    log.info("Entering main loop — all Layer 0 systems active")
    print()

    try:
        while not shutdown:
            # --- Tick YOLO event detector ---
            result = engine.latest_result()
            yolo_events = detector.tick(result)

            # --- Tick audio monitor ---
            audio_events = audio.tick()

            # --- Merge and print all events ---
            all_events = yolo_events + audio_events
            for ev in all_events:
                elapsed = ev.ts - t_start
                recent_events.append((elapsed, ev))
                event_counts[ev.type] = event_counts.get(ev.type, 0) + 1
                print(f"  [event] t={elapsed:6.2f}s  {_fmt_event(ev)}")

            # Keep the recent list bounded
            if len(recent_events) > 50:
                recent_events = recent_events[-50:]

            # --- Render video overlay ---
            if result is not None and result.annotated_frame is not None:
                frame = _draw_overlay(
                    result.annotated_frame, recent_events, detector, audio
                )
                cv2.imshow("Newton-for-a-Room  (q=quit)", frame)

            # --- Periodic status line (every 2 seconds) ---
            now = time.monotonic()
            if now - last_status_print >= 2.0:
                tids = detector.active_track_ids()
                tid_strs = [
                    f"id={tid}({detector.pose_for(tid)})" for tid in tids
                ]
                audio_st = audio.latest_state()
                audio_info = ""
                if audio_st is not None:
                    audio_info = (
                        f"  audio={audio_st.audio_level_db:.0f}dB "
                        f"{'SPEECH' if audio_st.speech_active else audio_st.dominant_class}"
                    )
                print(
                    f"  [status] t={now - t_start:6.1f}s  "
                    f"tracks={tid_strs or '(none)'}"
                    f"{audio_info}"
                )
                last_status_print = now

            # Check for 'q' keypress
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    finally:
        log.info("Shutting down...")
        engine.stop()
        audio.stop()
        camera.stop()
        cv2.destroyAllWindows()

        # --- Summary ---
        elapsed_total = time.monotonic() - t_start
        total_events = sum(event_counts.values())
        print()
        print("=" * 60)
        print(f"  Session: {elapsed_total:.1f}s  |  Events: {total_events}")
        if event_counts:
            for etype, count in sorted(event_counts.items()):
                print(f"    {etype:25s}  {count}")
        print("=" * 60)


if __name__ == "__main__":
    main()

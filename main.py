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
import os
import signal
import sys
import threading
import time

from typing import Optional

import cv2
import numpy as np

import json

import config
from perception.camera import CameraCapture
from perception.yolo_engine import YoloEngine
from perception.event_detector import Event, EventDetector
from perception.audio import AudioMonitor
from perception.plugs import PlugManager
from agents.baselines import CalibrationCollector
from agents.decisions import DecisionEngine
from agents.empty_room_watcher import EmptyRoomWatcher
from agents.observer import Observer, ObserverWorker
from agents.reasoner import Reasoner, ReasonerWorker
from agents.routing import should_call_reasoner
from agents.world_state import WorldState
from actuators.speaker import Speaker
from server.app import run_server_in_thread
from server.broadcaster import broadcaster

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
    if ev.type == "beat_1":
        narration = p.get("narration", "")
        if len(narration) > 62:
            narration = narration[:59] + "..."
        return f"[Beat 1] {narration}"
    if ev.type == "beat_2":
        narration = p.get("narration", "")
        if len(narration) > 62:
            narration = narration[:59] + "..."
        return f"[Beat 2] {narration}"
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
    "plug_power":          (180, 255, 180),
    "beat_1":              (255, 255, 255),
    "beat_2":              (180, 200, 255),  # soft blue for Reasoner
}
_STATUS_FG = (255, 255, 255)
_LOG_MAX_LINES = 10


def _draw_overlay(
    frame: np.ndarray,
    recent_events: list[tuple[float, Event]],
    detector: EventDetector,
    audio: AudioMonitor,
    plugs: Optional[PlugManager] = None,
) -> np.ndarray:
    """Draw zone polygons onto the frame.

    Previously this also burned in an event log, track status, audio
    status, and plug status. Those all live on the dashboard now
    (EventLog, StatsPanel, etc.) so the cv2 window stays clean — just
    YOLO boxes/skeleton (already drawn into annotated_frame by the
    engine) plus zone polygons for spatial context.
    `recent_events`, `detector`, `audio`, and `plugs` are kept in the
    signature so callers don't break; they're unused for now.
    """
    del recent_events, detector, audio, plugs  # intentionally unused

    out = frame
    for name, pts in config.ZONES.items():
        if len(pts) < 3:
            continue
        poly = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [poly], isClosed=True, color=_ZONE_COLOR, thickness=2)
        cv2.putText(out, name, pts[0], cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, _ZONE_COLOR, 2, cv2.LINE_AA)
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

    # Start the dashboard server early so it can serve /config and a warm-up
    # MJPEG stream (just placeholder frames) while the rest of the pipeline
    # boots. Any publish_* calls before a client connects are no-ops.
    run_server_in_thread(host="127.0.0.1", port=8000)
    print("  Dashboard:          http://127.0.0.1:8000")
    print(f"  Camera index:       {config.CAMERA_INDEX}")
    print(f"  Audio device:       {config.AUDIO_DEVICE_NAME}")
    print(f"  Zones loaded:       {list(config.ZONES.keys()) or '(none)'}")
    print(f"  Calibration:        {config.CALIBRATION_SECONDS}s")
    print(f"  Observer enabled:   {config.OBSERVER_ENABLED}  (model={config.GEMINI_MODEL})")
    print(f"  Reasoner enabled:   {config.REASONER_ENABLED}  (model={config.CLAUDE_MODEL})")
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
    audio_device_index = config.resolve_audio_device()
    log.info("Audio device resolved: '%s' → index %d", config.AUDIO_DEVICE_NAME, audio_device_index)
    audio = AudioMonitor(
        device_index=audio_device_index,
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
    log.info("Audio monitor started (device=%d)", audio_device_index)

    # -- Step 4: Create event detector --
    detector = EventDetector()

    # -- Step 5: Smart plugs --
    plugs: Optional[PlugManager] = None
    if config.KASA_USERNAME and config.KASA_PASSWORD:
        plugs = PlugManager()
        plugs.start()
        log.info("Smart plug discovery starting (async, non-blocking)…")
        # discover() blocks up to 15 s — run in a thread so the main loop
        # can start immediately and plug states just appear once found.
        def _bg_discover():
            found = plugs.discover(timeout=15.0)
            if found:
                log.info("Smart plugs: both plugs discovered and polling started")
            else:
                log.warning("Smart plugs: discovery partial or failed — plug states may be unavailable")
        threading.Thread(target=_bg_discover, daemon=True, name="plug-discover").start()
    else:
        log.info("Smart plugs: KASA credentials missing — skipping plug setup")

    # -- Step 6: World state --
    world = WorldState()
    log.info("WorldState initialized")

    # -- Step 6b: Calibration phase --
    def _calibration_overlay(frame: np.ndarray, elapsed: float, duration: float) -> None:
        """Draw a progress bar and countdown on the video frame during calibration."""
        h, w = frame.shape[:2]

        # Semi-transparent bar across the top
        bar_h = 44
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        remaining = max(0, duration - elapsed)
        progress = min(1.0, elapsed / duration)

        # Progress bar (orange fill)
        bar_x0, bar_y0, bar_y1 = 10, 10, 32
        bar_w = int(progress * (w - 20))
        cv2.rectangle(frame, (bar_x0, bar_y0), (bar_x0 + bar_w, bar_y1), (0, 160, 255), -1)
        cv2.rectangle(frame, (bar_x0, bar_y0), (w - 10, bar_y1), (100, 100, 100), 1)

        # Countdown text
        text = f"Calibrating... {remaining:.0f}s remaining"
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.putText(
            frame, text, ((w - tw) // 2, 26),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
        )

    cal = CalibrationCollector(
        world=world,
        engine=engine,
        detector=detector,
        audio=audio,
        plugs=plugs,
        duration=config.CALIBRATION_SECONDS,
    )
    print()
    print(f"  Calibrating — learning this room ({config.CALIBRATION_SECONDS}s)...")
    print()
    baselines = cal.run(overlay_callback=_calibration_overlay)
    print()
    print(
        f"  Baseline established. "
        f"Audio: {baselines.audio_mean_db:.1f} +/- {baselines.audio_std_db:.1f} dB, "
        f"Occupancy: {baselines.typical_occupancy}, "
        f"Lamp: {baselines.power_idle_lamp_w:.1f}W, "
        f"Fan: {baselines.power_idle_fan_w:.1f}W"
    )
    if baselines.ambient_audio_classes:
        print(f"  Ambient classes: {baselines.ambient_audio_classes}")
    print()

    # -- Step 6c: Observer agent (Layer 1) --
    observer = Observer(world=world, camera=camera)
    observer_worker = ObserverWorker(observer)
    if config.OBSERVER_ENABLED:
        observer_worker.start()
        log.info(
            "Observer started (model=%s, thinking=%s, refresh=%ds)",
            config.GEMINI_MODEL,
            "off" if config.OBSERVER_THINKING_BUDGET == 0 else config.OBSERVER_THINKING_BUDGET,
            config.OBSERVER_REFRESH_INTERVAL_S,
        )

    # -- Step 6d: Speaker (TTS) --
    speaker = Speaker()
    if config.TTS_ENABLED:
        speaker.start()

    # -- Step 6e: Reasoner agent (Layer 2) + decision engine --
    reasoner = Reasoner(world=world, camera=camera)
    reasoner_worker = ReasonerWorker(reasoner=reasoner)
    decisions = DecisionEngine(plugs=plugs, speaker=speaker, world=world)

    def _on_room_empty() -> None:
        log.info("EmptyRoomWatcher: room_empty_confirmed (debounce elapsed)")
        if not config.REASONER_ENABLED:
            return
        reasoner_worker.push_work(
            obs_result={
                "narration": "",
                "escalate": False,
                "escalate_reason": "room empty for debounce window",
            },
            event_types=["room_empty_confirmed"],
            frame=camera.latest_frame(),
        )

    empty_room_watcher = EmptyRoomWatcher(
        on_empty=_on_room_empty,
        debounce_s=config.EMPTY_ROOM_DEBOUNCE_S,
    )

    if config.REASONER_ENABLED:
        reasoner_worker.start()
        log.info(
            "Reasoner started (model=%s, max_tokens=%d)",
            config.CLAUDE_MODEL,
            config.REASONER_MAX_TOKENS,
        )

    # -- Step 7: Main monitoring loop --
    recent_events: list[tuple[float, Event]] = []  # (elapsed_sec, event)
    event_counts: dict[str, int] = {}
    t_start = time.monotonic()
    last_status_print = 0.0
    last_summary_push = time.monotonic()
    last_display_frame: Optional[np.ndarray] = None  # cache so window always has content

    # Graceful Ctrl+C — first press sets the flag, second press force-exits.
    shutdown = False
    _ctrl_c_count = 0

    def _signal_handler(sig, frame):
        nonlocal shutdown, _ctrl_c_count
        _ctrl_c_count += 1
        if _ctrl_c_count >= 2:
            print("\nForce-quitting.")
            sys.exit(1)
        shutdown = True
        print("\n(Ctrl+C again to force-quit)")

    signal.signal(signal.SIGINT, _signal_handler)

    log.info("Entering main loop — all Layer 0 systems active")
    print()

    # Throttles for the dashboard publishers.
    # - Snapshots at 10 Hz is plenty for UI (entities, audio dB, devices).
    # - Frames at 15 FPS balances smoothness vs. encode cost.
    last_snapshot_publish = 0.0
    last_frame_publish = 0.0
    SNAPSHOT_INTERVAL_S = 0.1
    FRAME_INTERVAL_S = 1.0 / 15.0

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
                # Dashboard: one message per event, as they fire.
                broadcaster.publish_event({
                    "type": ev.type,
                    "ts": ev.ts,
                    "elapsed": round(elapsed, 2),
                    "track_id": ev.track_id,
                    "zones": list(ev.zones) if ev.zones else [],
                    "payload": dict(ev.payload) if ev.payload else {},
                })

            # Keep the recent list bounded
            if len(recent_events) > 50:
                recent_events = recent_events[-50:]

            # --- Update world state ---
            world.update_from_yolo(result, detector)
            audio_st = audio.latest_state()
            if audio_st:
                world.update_audio(audio_st)
            if plugs:
                world.update_devices(plugs)
            empty_room_watcher.update(world.people_count())
            for ev in all_events:
                world.push_event(ev)

            # --- Dashboard: publish WorldState snapshot (throttled) ---
            now = time.monotonic()
            if now - last_snapshot_publish >= SNAPSHOT_INTERVAL_S:
                broadcaster.publish_snapshot(world.snapshot())
                last_snapshot_publish = now

            # --- Feed events to Observer (Layer 1) ---
            if config.OBSERVER_ENABLED and all_events:
                serialized_events = []
                for ev in all_events:
                    serialized_events.append({
                        "type": ev.type,
                        "track_id": ev.track_id,
                        "zones": list(ev.zones) if ev.zones else [],
                        "payload": dict(ev.payload) if ev.payload else {},
                    })
                observer_worker.push_events(
                    serialized_events, frame=camera.latest_frame(),
                )

            # --- Check for Observer results (Beat 1) ---
            if config.OBSERVER_ENABLED:
                obs_poll = observer_worker.poll_result()
                if obs_poll is not None:
                    obs_result, obs_event_types = obs_poll
                    narration = obs_result.get("narration", "")
                    escalate = obs_result.get("escalate", False)

                    # Print Beat 1 narration
                    print(f"  [BEAT 1] {narration}")

                    # Dashboard: push Observer narration + JSON output.
                    broadcaster.publish_narration("observer", {
                        "narration": narration,
                        "escalate": escalate,
                        "escalate_reason": obs_result.get("escalate_reason", ""),
                        "world_state_update": obs_result.get("world_state_update", {}),
                        "trigger_events": list(obs_event_types),
                        "ts": time.time(),
                    })

                    # Append to the recent_events overlay as a special entry
                    beat1_elapsed = time.monotonic() - t_start
                    recent_events.append((beat1_elapsed, Event(
                        type="beat_1",
                        ts=time.monotonic(),
                        track_id=None,
                        zones=[],
                        confidence=1.0,
                        payload={"narration": narration[:80]},
                    )))

                    # Routing: check if Reasoner should fire
                    # Use the first trigger event type, or "periodic_refresh"
                    primary_event_type = (
                        obs_event_types[0] if obs_event_types
                        else "periodic_refresh"
                    )
                    fired = should_call_reasoner(primary_event_type, obs_result)
                    if fired:
                        log.info(
                            "Routing → Reasoner (trigger=%s, escalate=%s, "
                            "reason=%s)",
                            primary_event_type, escalate,
                            obs_result.get("escalate_reason", ""),
                        )
                        # Beat 1: speak Observer narration immediately
                        if narration:
                            speaker.enqueue_beat1(narration)
                        # Beat 2: push to Reasoner worker (non-blocking)
                        if config.REASONER_ENABLED:
                            reasoner_worker.push_work(
                                obs_result=obs_result,
                                event_types=obs_event_types,
                                frame=camera.latest_frame(),
                            )
                    else:
                        # Beat 1 only — Observer handled it, no Reasoner needed
                        if narration:
                            speaker.enqueue_beat1(narration)
                        log.debug(
                            "Routing → Beat 1 only (trigger=%s, escalate=%s)",
                            primary_event_type, escalate,
                        )
                    # Dashboard: show routing decision in the Reasoner panel
                    # even before Claude is wired, so the UI isn't dead.
                    broadcaster.publish_routing({
                        "trigger": primary_event_type,
                        "fired": fired,
                        "escalate": escalate,
                        "reason": obs_result.get("escalate_reason", ""),
                        "ts": time.time(),
                    })

            # --- Check for Reasoner results (Beat 2) ---
            if config.REASONER_ENABLED:
                r = reasoner_worker.poll_result()
                if r is not None:
                    # Dispatch lamp/fan/alert/TTS actions
                    decisions.execute(r)

                    # Dashboard: push Reasoner narration + actions
                    broadcaster.publish_narration("reasoner", {
                        "narration": r.narration,
                        "lamp": r.lamp,
                        "fan": r.fan,
                        "alert": r.alert,
                        "speak": r.speak,
                        "reasoning": r.reasoning,
                        "world_state_update": r.world_state_update.model_dump(),
                        "ts": time.time(),
                    })

                    # Add Beat 2 to the on-screen event log
                    if r.narration:
                        beat2_elapsed = time.monotonic() - t_start
                        recent_events.append((beat2_elapsed, Event(
                            type="beat_2",
                            ts=time.monotonic(),
                            track_id=None,
                            zones=[],
                            confidence=1.0,
                            payload={"narration": r.narration[:80]},
                        )))

                    if r.narration:
                        print(f"  [BEAT 2] {r.narration}")
                    if r.reasoning:
                        log.debug("Reasoner reasoning: %s", r.reasoning[:200])

            # --- Minutely session summary ---
            # Every REASONER_SUMMARY_INTERVAL_S seconds, push a synthetic event so
            # the Reasoner can update its session_narrative and activity_label even
            # during quiet periods when no real events are triggering it.
            if config.REASONER_ENABLED:
                now_s = time.monotonic()
                if now_s - last_summary_push >= config.REASONER_SUMMARY_INTERVAL_S:
                    reasoner_worker.push_work(
                        obs_result={
                            "narration": "",
                            "escalate": False,
                            "escalate_reason": "minutely session summary",
                        },
                        event_types=["periodic_refresh_minutely"],
                        frame=camera.latest_frame(),
                    )
                    last_summary_push = now_s
                    log.debug("Pushed minutely session summary to Reasoner")

            # --- Render video overlay ---
            # Always update last_display_frame when we have a new annotated result,
            # then always call imshow so the window exists and waitKey reliably
            # captures 'q' regardless of whether YOLO produced a result this tick.
            if result is not None and result.annotated_frame is not None:
                last_display_frame = _draw_overlay(
                    result.annotated_frame, recent_events, detector, audio, plugs
                )
            if last_display_frame is not None:
                cv2.imshow("Newton-for-a-Room  (q=quit)", last_display_frame)

                # Dashboard: encode + publish JPEG at ~15 FPS for MJPEG.
                # Not gated on has_clients() because MJPEG viewers connect
                # via HTTP (not WS) and aren't counted there.
                if now - last_frame_publish >= FRAME_INTERVAL_S:
                    ok, buf = cv2.imencode(
                        ".jpg", last_display_frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 70],
                    )
                    if ok:
                        broadcaster.publish_frame(buf.tobytes())
                        last_frame_publish = now

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
                plug_info = ""
                if plugs is not None:
                    parts = []
                    for alias in (config.LAMP_PLUG_ALIAS, config.FAN_PLUG_ALIAS):
                        st = plugs.state(alias)
                        if st is None:
                            parts.append(f"{alias}=discovering…")
                        else:
                            parts.append(f"{alias}={'ON' if st.is_on else 'off'} {st.power_w:.0f}W")
                    plug_info = "  plugs=[" + ", ".join(parts) + "]"
                print(
                    f"  [status] t={now - t_start:6.1f}s  "
                    f"tracks={tid_strs or '(none)'}"
                    f"{audio_info}"
                    f"{plug_info}"
                )
                last_status_print = now

            # Check for keypresses: 'q' = quit, 'd' = dump world state
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("d"):
                snap = world.snapshot()
                print("\n" + "=" * 60)
                print("  WORLD STATE SNAPSHOT")
                print("=" * 60)
                print(json.dumps(snap, indent=2, default=str))
                print("=" * 60 + "\n")

    finally:
        log.info("Shutting down...")

        # Stop LLM worker threads first (no hardware — just threads)
        if config.OBSERVER_ENABLED:
            observer_worker.stop()
        if config.REASONER_ENABLED:
            reasoner_worker.stop()
        if config.TTS_ENABLED:
            speaker.stop()

        # Print summary immediately — doesn't depend on hardware cleanup.
        elapsed_total = time.monotonic() - t_start
        total_events = sum(event_counts.values())
        print()
        print("=" * 60)
        print(f"  Session: {elapsed_total:.1f}s  |  Events: {total_events}")
        if event_counts:
            for etype, count in sorted(event_counts.items()):
                print(f"    {etype:25s}  {count}")
        print("=" * 60)

        # Run hardware cleanup in a daemon thread so a hung cap.release() or
        # stream.close() (both known to stall on macOS) can't trap the process.
        # os._exit(0) at the end kills everything; the OS reclaims the camera
        # handle and mic handle regardless, so the green light will go off.
        def _cleanup():
            engine.stop()
            audio.stop()
            camera.stop()
            if plugs is not None:
                plugs.stop()
            cv2.destroyAllWindows()

        t = threading.Thread(target=_cleanup, daemon=True)
        t.start()
        t.join(timeout=6.0)
        if t.is_alive():
            log.warning("Cleanup timed out — force-exiting")
        os._exit(0)


if __name__ == "__main__":
    main()

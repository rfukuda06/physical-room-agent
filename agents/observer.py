"""
Layer 1 — Observer agent (Gemini 2.5 Flash). Beat 1 of the two-beat rhythm.

How it works (for first-time readers):
--------------------------------------
The Observer is the system's "fast eye". When Layer 0 detects something
interesting (a person moved, a sound happened, a plug changed state), the
Observer gets called with camera frames + structured sensor data. It sends
all of that to Google's Gemini 2.5 Flash model — a fast multimodal LLM that
can understand both images and text — and asks it to produce a short factual
description of what just happened.

The key idea: Gemini sees the room visually (camera frames) AND gets structured
data (YOLO detections, audio levels, plug states) as context. It fuses both
to produce a narration like "Person stood up from desk and is walking toward
the door."  It also decides whether the event is interesting enough to escalate
to the Reasoner (Claude) for deeper thinking.

This module has two classes:
  - Observer: handles the actual Gemini API call, prompt building, response
    parsing, and fallback logic.
  - ObserverWorker: runs the Observer in a background thread so the main loop
    (which drives the video overlay at ~30 FPS) never freezes waiting for a
    ~1s API response.

SDK note: We use the `google-genai` SDK (not the deprecated `google-generativeai`).
The new SDK is client-based: you create a `genai.Client`, then call
`client.models.generate_content(...)`. Images are sent as `types.Part.from_bytes`.

Output contract (JSON):
  {
    "narration": "short factual description, spoken aloud as Beat 1",
    "world_state_update": {
      "scene_description": "one-line scene summary",
      "activity_summary": "what person(s) are doing",
      "mood": "quiet | active | transitional"
    },
    "escalate": true | false,
    "escalate_reason": "short string — why escalate / why not"
  }

The `escalate` flag drives the hybrid routing policy — see agents/routing.py
and ARCHITECTURE_AND_BUILD_PLAN copy.md Section 2.5.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Optional

import cv2
import numpy as np

from google import genai
from google.genai import types

import config
from agents.world_state import WorldState
from perception.camera import CameraCapture

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt — sent with every Gemini call as the system_instruction.
# Kept concise to minimize token cost (this gets cached by the SDK across
# calls, so the marginal cost after the first call is near zero).
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are the Observer — a fast, factual room monitoring agent. You receive \
camera frames and structured sensor data from a room. Your job is to describe \
what just happened in a short, natural sentence.

SIGNAL RELIABILITY — the structured events you receive are from computer vision \
and audio classifiers. They are useful but noisy. Always verify against the \
camera frames before narrating:
- zone_transition: most reliable. Occasionally false-positive at zone boundaries.
- pose_change: often correct but frequently fires false transitions, especially \
  at the sitting/standing boundary. If the camera shows no visible change in \
  posture, ignore the event.
- new_person / lost_person: pre-filtered for phantoms, but still occasionally \
  wrong. If new_person fires but the camera shows no one new, or lost_person \
  fires but the person is clearly still visible, trust the camera.
- Audio classes (YAMNet): conservative — many more false negatives than false \
  positives. When a class IS reported, it is usually correct. Speech detection \
  is highly reliable. Trust reported audio classes, but do not assume silence \
  means nothing happened — the classifier often misses quiet or brief sounds.
- Audio spikes: reliable loudness signal, but fires on mundane sounds too \
  (chair scrape, AC). If the camera shows nothing noteworthy, say so.

RULES:
1. Be factual. Describe what you SEE in the frames, using the sensor data as \
supporting context. Do not interpret intent, judge mood, or reason about why.
2. Be brief. One sentence, occasionally two. Written to be spoken aloud.
3. If nothing meaningful changed since the last observation, say so briefly: \
"Room unchanged. One person at desk."
4. Set escalate=true ONLY when the event likely needs deeper reasoning: \
occupancy change, unusual or ambiguous activity, potential security concern, \
or anything a thoughtful system should evaluate further. Routine pose \
adjustments and quiet-room refreshes do NOT need escalation.

OUTPUT FORMAT — respond with ONLY this JSON, no markdown fences, no extra text:
{
  "narration": "short factual description",
  "world_state_update": {
    "scene_description": "one-line scene summary",
    "activity_summary": "what the person(s) are doing"
  },
  "escalate": false,
  "escalate_reason": "brief reason"
}
"""


# ---------------------------------------------------------------------------
# Observer — handles a single Gemini call
# ---------------------------------------------------------------------------

class Observer:
    """Layer 1 Observer agent — Gemini 2.5 Flash.

    Call `call(trigger_events, frame)` to make a single Gemini request.
    The method is synchronous (blocks until Gemini responds or times out).
    Use ObserverWorker to run this off the main thread.
    """

    def __init__(self, world: WorldState, camera: CameraCapture) -> None:
        self._world = world
        self._camera = camera
        # The genai.Client handles connection pooling and retries internally.
        # api_key is read from the GEMINI_API_KEY environment variable via
        # config.py, which loads it from .env at import time.
        self._client = genai.Client(api_key=config.GEMINI_API_KEY)
        self._consecutive_failures: int = 0
        self._last_output: Optional[dict] = None  # cache for debugging

    # ----- public API -----

    def call(
        self,
        trigger_events: list[dict],
        frame: Optional[np.ndarray] = None,
    ) -> Optional[dict]:
        """Make one synchronous Gemini call and return the parsed output.

        Parameters
        ----------
        trigger_events : list[dict]
            Serialized Layer 0 events that triggered this call.  Each dict has
            keys: type, track_id, zones, payload.  Empty list means periodic
            refresh (no specific event).
        frame : np.ndarray | None
            Current BGR frame from the camera.  If None, we grab the latest
            from the CameraCapture instance.

        Returns
        -------
        dict | None
            Parsed Observer output matching the JSON contract, or None if
            the Observer is disabled.
        """
        if not config.OBSERVER_ENABLED:
            return None

        # 1. Snapshot the world state (thread-safe deep copy)
        snapshot = self._world.snapshot_for_observer()

        # 2. Gather image parts — images go FIRST in the contents array
        #    (Google best practice: put images before the text prompt).
        parts: list[types.Part] = []
        frames_sent = 0

        current_frame = frame if frame is not None else self._camera.latest_frame()
        if current_frame is not None:
            parts.append(types.Part.from_bytes(
                data=self._encode_frame(current_frame),
                mime_type="image/jpeg",
            ))
            frames_sent += 1

        # Grab prior frames from the rolling buffer for temporal context.
        # With 3 frames total: current, ~1s ago, ~2.5s ago — evenly spaced
        # so Gemini can see motion and scene changes leading up to the event.
        if config.OBSERVER_MAX_FRAMES > 1:
            buf = self._camera.buffer_snapshot(seconds=3)
            # How many prior frames do we still need?
            n_prior = config.OBSERVER_MAX_FRAMES - 1  # already have current
            if len(buf) >= 3 and n_prior > 0:
                # Space the picks evenly across the buffer
                for i in range(n_prior):
                    # e.g. n_prior=2 → pick at 1/3 and 2/3 of the buffer
                    idx = int(len(buf) * (i + 1) / (n_prior + 1))
                    prior_frame = buf[idx][1]
                    parts.append(types.Part.from_bytes(
                        data=self._encode_frame(prior_frame),
                        mime_type="image/jpeg",
                    ))
                    frames_sent += 1

        # 3. Build the text prompt (goes AFTER images)
        prompt_text = self._build_prompt(snapshot, trigger_events, frames_sent)
        parts.append(types.Part.from_text(text=prompt_text))

        # 4. Call Gemini
        try:
            response = self._client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=parts,
                config=types.GenerateContentConfig(
                    system_instruction=_SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    temperature=0.3,
                    max_output_tokens=300,
                    thinking_config=types.ThinkingConfig(
                        thinking_budget=config.OBSERVER_THINKING_BUDGET,
                    ),
                ),
            )
            raw = response.text
        except Exception as exc:
            log.warning("Observer Gemini call failed: %s", exc)
            self._consecutive_failures += 1
            self._backoff_if_needed()
            return self._fallback_output(trigger_events, snapshot)

        # 5. Parse the JSON response
        output = self._parse_response(raw)
        if output is None:
            log.warning("Observer: invalid JSON from Gemini: %.200s", raw)
            self._consecutive_failures += 1
            return self._fallback_output(trigger_events, snapshot)

        # 6. Success — reset failure counter, apply world state update
        self._consecutive_failures = 0
        self._last_output = output

        if "world_state_update" in output:
            self._world.apply_observer_update(output["world_state_update"])

        return output

    # ----- internal helpers -----

    def _encode_frame(self, frame: np.ndarray) -> bytes:
        """Resize + compress a BGR frame to JPEG bytes.

        Optionally resize so both dimensions are <= OBSERVER_FRAME_MAX_DIM.
        With max_dim=1280 (default), raw 1280x720 frames pass through
        untouched — 2 tiles = 516 tokens each in Gemini's tokenizer.
        Set max_dim=768 for 1 tile (258 tokens) or 384 for aggressive
        downsampling.

        The aspect ratio is preserved — we scale down to fit within a
        max_dim x max_dim box.
        """
        h, w = frame.shape[:2]
        max_dim = config.OBSERVER_FRAME_MAX_DIM

        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        ok, buf = cv2.imencode(
            ".jpg", frame,
            [cv2.IMWRITE_JPEG_QUALITY, config.OBSERVER_FRAME_QUALITY],
        )
        if not ok:
            raise RuntimeError("Failed to encode frame as JPEG")
        return buf.tobytes()

    def _build_prompt(
        self,
        snapshot: dict,
        trigger_events: list[dict],
        n_frames: int,
    ) -> str:
        """Assemble the per-call user prompt from sensor data and events."""
        lines: list[str] = []

        # What triggered this call
        if trigger_events:
            lines.append("TRIGGER: The following events just occurred:")
            for ev in trigger_events:
                payload_str = json.dumps(ev.get("payload", {}))
                lines.append(f"  - {ev['type']}: {payload_str}")
        else:
            lines.append("TRIGGER: Periodic background refresh (no specific event).")

        # Structured sensor data
        lines.append("")
        lines.append("SENSOR STATE:")
        lines.append(json.dumps(snapshot, indent=2, default=str))

        # Note about attached frames
        lines.append("")
        lines.append(f"ATTACHED: {n_frames} camera frame(s) — most recent first.")
        if n_frames == 2:
            lines.append(
                "The second frame is from ~1.5 seconds ago for temporal context."
            )
        elif n_frames >= 3:
            lines.append(
                "Frames span ~3 seconds: most recent, ~1s ago, ~2.5s ago. "
                "Use them to observe motion and changes."
            )

        return "\n".join(lines)

    def _parse_response(self, raw_text: str) -> Optional[dict]:
        """Parse Gemini's JSON response, with fallback for markdown fences.

        Returns the parsed dict if valid, or None if we can't parse it.
        """
        # First try direct parse
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError:
            # Gemini sometimes wraps JSON in ```json ... ``` fences
            stripped = raw_text.strip()
            if stripped.startswith("```"):
                stripped = stripped.split("\n", 1)[-1].rsplit("```", 1)[0]
                try:
                    data = json.loads(stripped)
                except json.JSONDecodeError:
                    return None
            else:
                return None

        # Validate the required key
        if "narration" not in data:
            return None

        # Fill in defaults for optional fields
        data.setdefault("world_state_update", {})
        data.setdefault("escalate", False)
        data.setdefault("escalate_reason", "")

        # Coerce escalate to bool (Gemini occasionally returns strings)
        data["escalate"] = bool(data["escalate"])

        return data

    def _fallback_output(
        self,
        trigger_events: list[dict],
        snapshot: dict,
    ) -> dict:
        """Generate a minimal deterministic narration from sensor data.

        Used when Gemini fails (timeout, bad JSON, API down).  Ensures the
        pipeline never stalls — the Reasoner routing still has something to
        evaluate, and a narration (even a basic one) is produced.
        """
        people = snapshot.get("people_count", 0)

        if trigger_events:
            event_types = [e["type"] for e in trigger_events]
            narration = (
                f"Event detected: {', '.join(event_types)}. "
                f"{people} person(s) in room."
            )
        else:
            narration = f"Room status: {people} person(s) detected."

        # Conservative escalation: only escalate for must-escalate event types
        should_escalate = any(
            e["type"] in config.REASONER_ALWAYS for e in trigger_events
        )

        return {
            "narration": narration,
            "world_state_update": {},
            "escalate": should_escalate,
            "escalate_reason": (
                "fallback — Gemini unavailable"
                if should_escalate
                else "fallback, no escalation needed"
            ),
        }

    def _backoff_if_needed(self) -> None:
        """Sleep briefly after consecutive failures (exponential backoff).

        Prevents hammering a failing API.  Capped at 30s.  After 10 consecutive
        failures we still call but the fallback output keeps the pipeline alive.
        """
        if self._consecutive_failures > 1:
            delay = min(2 ** (self._consecutive_failures - 1), 30)
            log.info(
                "Observer: backing off %.1fs after %d consecutive failures",
                delay, self._consecutive_failures,
            )
            time.sleep(delay)


# ---------------------------------------------------------------------------
# ObserverWorker — background thread that batches events and calls Observer
# ---------------------------------------------------------------------------

class ObserverWorker:
    """Runs the Observer in a daemon thread with event batching and periodic refresh.

    How it works:
    - The main loop calls `push_events(events, frame)` whenever Layer 0 events fire.
      This is non-blocking — it just appends events to a queue and wakes the worker.
    - The worker thread wakes up, waits briefly for more events to batch (debounce),
      then drains all pending events into a single Observer call.
    - Results are posted to `_result` and the main loop picks them up via
      `poll_result()` on its next tick.
    - When no events come for OBSERVER_REFRESH_INTERVAL_S seconds, the worker
      fires a periodic refresh (Observer call with empty event list).
    """

    def __init__(self, observer: Observer) -> None:
        self._observer = observer
        self._pending_events: list[dict] = []
        self._pending_frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._wake = threading.Event()
        # Result is a tuple: (observer_output_dict, list_of_trigger_event_types)
        self._result: Optional[tuple[dict, list[str]]] = None
        self._result_ready = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._last_refresh_time: float = time.monotonic()

    def start(self) -> None:
        """Start the background worker thread."""
        self._thread = threading.Thread(
            target=self._run, name="observer-worker", daemon=True,
        )
        self._thread.start()
        log.info("ObserverWorker thread started")

    def stop(self) -> None:
        """Signal the worker to stop and wait for it to exit."""
        self._stop.set()
        self._wake.set()  # unblock if sleeping on wait()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            log.info("ObserverWorker thread stopped")

    def push_events(
        self,
        events: list[dict],
        frame: Optional[np.ndarray] = None,
    ) -> None:
        """Push new events from the main loop (non-blocking).

        Called from the main thread whenever Layer 0 events fire.  The frame
        is the current camera capture — we keep only the most recent one.
        """
        with self._lock:
            self._pending_events.extend(events)
            if frame is not None:
                self._pending_frame = frame
        self._wake.set()

    def poll_result(self) -> Optional[tuple[dict, list[str]]]:
        """Non-blocking check for Observer results from the main loop.

        Returns (observer_output, event_types) if a result is ready, else None.
        The result is consumed (cleared) on read.
        """
        if self._result_ready.is_set():
            self._result_ready.clear()
            result = self._result
            self._result = None
            return result
        return None

    def _run(self) -> None:
        """Worker thread main loop."""
        while not self._stop.is_set():
            # Wait for either: events pushed, or periodic refresh timer
            remaining = config.OBSERVER_REFRESH_INTERVAL_S - (
                time.monotonic() - self._last_refresh_time
            )
            timeout = max(0.1, remaining)
            self._wake.wait(timeout=timeout)
            self._wake.clear()

            if self._stop.is_set():
                break

            # Debounce: wait briefly for more events to batch.
            # If 5 events fire within 200ms (e.g. pose_change + zone_transition
            # on the same person in the same frame), they all get batched into
            # a single Gemini call.
            time.sleep(config.OBSERVER_DEBOUNCE_S)

            # Drain pending events
            with self._lock:
                events = self._pending_events
                self._pending_events = []
                frame = self._pending_frame
                self._pending_frame = None

            # Check if this is a periodic refresh
            now = time.monotonic()
            is_refresh = (
                now - self._last_refresh_time
            ) >= config.OBSERVER_REFRESH_INTERVAL_S

            # If no events and refresh timer hasn't expired, go back to sleep
            if not events and not is_refresh:
                continue

            # Record the event types for routing (before calling Observer)
            event_types = [e["type"] for e in events]

            # Call Observer (synchronous — blocks until Gemini responds)
            result = self._observer.call(events, frame)

            if result is not None:
                self._result = (result, event_types)
                self._result_ready.set()

            self._last_refresh_time = time.monotonic()

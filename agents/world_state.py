"""
Shared in-memory world model.

WorldState is the single source of truth for "what the room looks like right
now."  It aggregates every Layer 0 signal — YOLO tracks, audio, smart plugs —
into one queryable, JSON-serializable object.

**How it works:**

The main loop calls the `update_*` methods on every tick (~30 fps):
  - update_from_yolo()  — rebuilds the entity list from YOLO + EventDetector
  - update_audio()      — copies the latest AudioState snapshot
  - update_devices()    — copies the latest plug states
  - push_event()        — appends a new event to the ring buffer

Separately, the Observer (Block 3) and Reasoner (Block 4) call
`apply_observer_update()` / `apply_reasoner_update()` to write semantic
fields like scene_description and activity_summary.

When Observer or Reasoner needs the current state, it calls `snapshot()` which
acquires the lock, deep-copies everything into a plain dict, and returns it.
The caller gets a frozen picture that won't change while it's building a prompt.

**Thread safety:**

A single `threading.Lock` protects all reads and writes.  The lock is held
briefly (no I/O inside it).  snapshot*() returns a deep copy, so callers
never hold a reference to live data.
"""

from __future__ import annotations

import copy
import datetime
import math
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

import config
from perception.audio import AudioState
from perception.event_detector import Event, EventDetector
from perception.plugs import PlugManager
from perception.yolo_engine import YoloResult


# ---------------------------------------------------------------------------
# Sub-structures — typed containers for each slice of world state
# ---------------------------------------------------------------------------

@dataclass
class EntityState:
    """One tracked person in the room."""
    id: int
    bbox_xywh: tuple[float, float, float, float]
    zones: list[str]
    pose: str                     # "standing" | "sitting" | "walking" | "unknown"
    first_seen: float             # monotonic timestamp
    last_seen: float              # monotonic timestamp
    velocity_px_per_s: float      # 2D center displacement per second


@dataclass
class AudioSnapshot:
    """Point-in-time copy of audio perception state."""
    level_db: float = -100.0
    top_classes: list[tuple[str, float]] = field(default_factory=list)
    dominant_class: str = "silence"
    speech_active: bool = False
    recent_spike: bool = False
    spike_magnitude_db: float = 0.0


@dataclass
class DeviceState:
    """One smart plug's state, including agent-command history and lockout."""
    alias: str = ""
    on: bool = False
    power_w: float = 0.0

    # Agent-command history (used to detect manual overrides)
    last_agent_command_at: float | None = None       # monotonic ts of last DE-issued toggle
    last_agent_command_intent: bool | None = None    # what the agent tried to set it to

    # Manual override / lockout tracking
    last_manual_override_at: float | None = None     # monotonic ts of last detected override
    lockout_until: float | None = None               # monotonic ts; if > now, no agent toggles

    # First-contact sentinel: set to True after the very first update_devices poll.
    # Override detection is skipped on the first observation to avoid a startup
    # false positive when the plug was already on before the agent started.
    _observed: bool = False


@dataclass
class Baselines:
    """Learned "normal" for this room.  Populated by calibration (Block 2)."""
    audio_mean_db: float = -50.0
    audio_std_db: float = 5.0
    typical_occupancy: int = 0
    power_idle_lamp_w: float = 0.0
    power_idle_fan_w: float = 0.0
    ambient_audio_classes: list[str] = field(default_factory=list)
    calibrated: bool = False


# ---------------------------------------------------------------------------
# Internal memo — tracks per-entity state that persists across ticks
# ---------------------------------------------------------------------------

@dataclass
class _EntityMemo:
    """Across-tick memory for one track_id (first_seen, velocity)."""
    first_seen: float
    prev_center_x: float
    prev_center_y: float
    prev_ts: float


# ---------------------------------------------------------------------------
# WorldState
# ---------------------------------------------------------------------------

_MAX_EVENTS = 50


class WorldState:
    """Thread-safe, in-memory world model.  No database — lives in RAM."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

        # Machine state (updated every tick by main loop)
        self._entities: list[EntityState] = []
        self._people_count: int = 0
        self._audio: AudioSnapshot = AudioSnapshot()
        self._devices: dict[str, DeviceState] = {}

        # Semantic state (updated by Observer / Reasoner)
        self._scene_description: str = ""
        self._activity_summary: str = ""

        # Reasoner session model — compounds in intelligence over time
        self._session_narrative: str = ""   # Reasoner's running interpretation of the session
        self._activity_label: str = "unknown"  # current activity classification
        self._session_start: float = time.monotonic()

        # Baselines (populated by calibration)
        self._baselines: Baselines = Baselines()

        # Event ring buffer
        self._recent_events: list[dict] = []

        # Internal: per-entity memory for velocity / first_seen tracking
        self._entity_memo: dict[int, _EntityMemo] = {}

    # ------------------------------------------------------------------
    # Machine state updates (called from main loop every tick)
    # ------------------------------------------------------------------

    def update_from_yolo(
        self,
        result: Optional[YoloResult],
        detector: EventDetector,
    ) -> None:
        """Rebuild entity list from current YOLO result + EventDetector state.

        For each active track:
          1. Match it to a YoloEntity by track_id to get bbox.
          2. Get pose and zones from the EventDetector.
          3. Compute velocity from center displacement vs. previous tick.
          4. Track first_seen using internal _entity_memo.
        """
        if result is None:
            return

        now = time.monotonic()

        # Build a lookup from track_id -> YoloEntity for O(1) matching
        yolo_by_tid: dict[int, tuple[float, float, float, float]] = {}
        for ent in result.entities:
            if ent.track_id is not None:
                yolo_by_tid[ent.track_id] = ent.bbox_xywh

        active_tids = detector.active_track_ids()
        new_entities: list[EntityState] = []

        with self._lock:
            for tid in active_tids:
                bbox = yolo_by_tid.get(tid)
                if bbox is None:
                    # Track is in the grace period (EventDetector holds it but
                    # YOLO didn't detect it this frame).  Skip — we'll keep
                    # the entity from the previous tick via the memo.
                    continue

                cx, cy = bbox[0], bbox[1]
                pose = detector.pose_for(tid) or "unknown"
                zones = detector.zones_for(tid)

                # Look up or create the memo for this track
                memo = self._entity_memo.get(tid)
                if memo is None:
                    # First time seeing this track — no velocity yet
                    memo = _EntityMemo(
                        first_seen=now,
                        prev_center_x=cx,
                        prev_center_y=cy,
                        prev_ts=now,
                    )
                    self._entity_memo[tid] = memo
                    velocity = 0.0
                else:
                    # Compute velocity (pixels per second)
                    dt = now - memo.prev_ts
                    if dt > 0.001:  # guard against division by zero
                        dx = cx - memo.prev_center_x
                        dy = cy - memo.prev_center_y
                        displacement = math.sqrt(dx * dx + dy * dy)
                        velocity = displacement / dt
                    else:
                        velocity = 0.0
                    # Update memo for next tick
                    memo.prev_center_x = cx
                    memo.prev_center_y = cy
                    memo.prev_ts = now

                new_entities.append(EntityState(
                    id=tid,
                    bbox_xywh=bbox,
                    zones=zones,
                    pose=pose,
                    first_seen=memo.first_seen,
                    last_seen=now,
                    velocity_px_per_s=round(velocity, 1),
                ))

            # Prune memos for tracks no longer active
            active_set = set(active_tids)
            stale = [k for k in self._entity_memo if k not in active_set]
            for k in stale:
                del self._entity_memo[k]

            self._entities = new_entities
            self._people_count = len(new_entities)

    def update_audio(self, audio_state: AudioState) -> None:
        """Copy latest AudioState into the world model."""
        with self._lock:
            self._audio = AudioSnapshot(
                level_db=float(audio_state.audio_level_db),
                top_classes=list(audio_state.top_classes),
                dominant_class=str(audio_state.dominant_class),
                speech_active=bool(audio_state.speech_active),
                recent_spike=bool(audio_state.recent_spike),
                spike_magnitude_db=float(audio_state.spike_magnitude_db),
            )

    def record_agent_command(self, alias: str, intent: bool, now: float | None = None) -> None:
        """Called by DecisionEngine immediately before issuing a turn_on/off.

        Stores the agent's intent and timestamp so update_devices() can later
        distinguish 'state still settling' from 'user just overrode us'.
        """
        if now is None:
            now = time.monotonic()
        with self._lock:
            ds = self._devices.get(alias)
            if ds is None:
                ds = DeviceState(alias=alias)
                self._devices[alias] = ds
            ds.last_agent_command_at = now
            ds.last_agent_command_intent = intent

    def device_state(self, alias: str) -> DeviceState | None:
        """Return a copy of the DeviceState for this plug, or None if unknown."""
        with self._lock:
            ds = self._devices.get(alias)
            if ds is None:
                return None
            # Return a copy so callers can read without holding the lock
            return DeviceState(
                alias=ds.alias, on=ds.on, power_w=ds.power_w,
                last_agent_command_at=ds.last_agent_command_at,
                last_agent_command_intent=ds.last_agent_command_intent,
                last_manual_override_at=ds.last_manual_override_at,
                lockout_until=ds.lockout_until,
                _observed=ds._observed,
            )

    def people_count(self) -> int:
        """Current number of tracked persons (thread-safe accessor)."""
        with self._lock:
            return self._people_count

    def update_devices(self, plugs: PlugManager, now: float | None = None) -> None:
        """Copy latest plug states into the world model, detecting manual overrides.

        Override detection: if the actual is_on differs from our last agent-command
        intent AND the command was issued more than AGENT_COMMAND_GRACE_S ago (or
        we have no record of commanding it at all), the user touched it.
        Set last_manual_override_at and lockout_until.
        """
        if now is None:
            now = time.monotonic()
        with self._lock:
            for alias in (config.LAMP_PLUG_ALIAS, config.FAN_PLUG_ALIAS):
                st = plugs.state(alias)
                if st is None:
                    continue
                ds = self._devices.get(alias)
                if ds is None:
                    ds = DeviceState(alias=alias)
                    self._devices[alias] = ds

                new_on = bool(st.is_on)
                # first_contact is True when update_devices has never polled
                # this plug before AND no prior agent command has been recorded.
                # If an agent command exists (record_agent_command was called
                # before the first poll), we still run override detection so
                # we can catch a plug that ignored the command.
                first_contact = (not ds._observed) and (ds.last_agent_command_intent is None)

                if not first_contact:
                    # Override detection — only when:
                    #   (a) the new state contradicts our last intent, AND
                    #   (b) we're outside the grace window OR we never commanded the device
                    intent = ds.last_agent_command_intent
                    cmd_at = ds.last_agent_command_at
                    outside_grace = cmd_at is None or (now - cmd_at) > config.AGENT_COMMAND_GRACE_S
                    state_contradicts_intent = (intent is not None) and (new_on != intent)
                    state_changed_unprompted = (intent is None) and (new_on != ds.on)

                    if outside_grace and (state_contradicts_intent or state_changed_unprompted):
                        ds.last_manual_override_at = now
                        ds.lockout_until = now + config.MANUAL_OVERRIDE_LOCKOUT_S

                ds.on = new_on
                ds.power_w = float(st.power_w)
                ds._observed = True

    def push_event(self, event: Event) -> None:
        """Serialize an Event and append to the ring buffer (max 50)."""
        entry = {
            "type": event.type,
            "ts": event.ts,
            "track_id": event.track_id,
            "zones": list(event.zones) if event.zones else [],
            "payload": dict(event.payload) if event.payload else {},
        }
        with self._lock:
            self._recent_events.append(entry)
            if len(self._recent_events) > _MAX_EVENTS:
                self._recent_events = self._recent_events[-_MAX_EVENTS:]

    # ------------------------------------------------------------------
    # Baselines
    # ------------------------------------------------------------------

    def set_baselines(self, baselines: Baselines) -> None:
        """Store calibration results (called once by CalibrationCollector)."""
        with self._lock:
            self._baselines = baselines

    # ------------------------------------------------------------------
    # Semantic state updates (called from Observer / Reasoner workers)
    # ------------------------------------------------------------------

    def apply_observer_update(self, update: dict) -> None:
        """Merge scene_description, activity_summary, mood from Observer."""
        with self._lock:
            if "scene_description" in update:
                self._scene_description = str(update["scene_description"])
            if "activity_summary" in update:
                self._activity_summary = str(update["activity_summary"])

    def apply_reasoner_update(self, update: dict) -> None:
        """Merge deeper semantic fields from Reasoner, including session model."""
        with self._lock:
            if "scene_description" in update:
                self._scene_description = str(update["scene_description"])
            if "activity_summary" in update:
                self._activity_summary = str(update["activity_summary"])
            if update.get("session_narrative"):
                self._session_narrative = str(update["session_narrative"])
            if update.get("activity_label"):
                self._activity_label = str(update["activity_label"])

    # ------------------------------------------------------------------
    # Snapshots — deep copies for LLM prompt building
    # ------------------------------------------------------------------

    def snapshot(self) -> dict:
        """Full world state as a JSON-serializable dict.

        Acquires the lock, copies everything, releases.  The returned dict
        contains only native Python types (no numpy, no dataclass instances).
        """
        with self._lock:
            return self._build_snapshot(include_semantic=True, include_events=True)

    def snapshot_for_observer(self) -> dict:
        """Lighter snapshot for the Observer: entities, audio, devices, baselines.

        Excludes semantic fields (scene_description, etc.) and event history
        because the Observer is producing those, not consuming them.
        """
        with self._lock:
            return self._build_snapshot(include_semantic=False, include_events=False)

    def snapshot_audio_only(self) -> dict:
        """Audio-only snapshot for the Observer. Gemini reads visual state from frames."""
        with self._lock:
            return {
                "level_db": round(self._audio.level_db, 1),
                "dominant_class": self._audio.dominant_class,
                "speech_active": self._audio.speech_active,
                "recent_spike": self._audio.recent_spike,
                "spike_magnitude_db": round(self._audio.spike_magnitude_db, 1),
                "top_classes": [
                    {"label": label, "confidence": round(conf, 2)}
                    for label, conf in self._audio.top_classes
                ],
            }

    def snapshot_for_reasoner(self, now: float | None = None) -> dict:
        """Full snapshot for the Reasoner, including session model + events.

        `now` is the monotonic reference time for lockout/age calculations.
        Defaults to time.monotonic().  Pass an explicit value in tests that
        use synthetic timestamps for update_devices/record_agent_command.
        """
        if now is None:
            now = time.monotonic()
        with self._lock:
            snap = self._build_snapshot(include_semantic=True, include_events=True, now=now)
            snap["session_elapsed_s"] = round(now - self._session_start, 1)
            snap["session_narrative"] = self._session_narrative
            snap["activity_label"] = self._activity_label
            return snap

    def _device_for_reasoner(self, d: DeviceState, now: float) -> dict:
        """Serialize one DeviceState for the Reasoner prompt, including lockout info."""
        lockout_remaining = 0.0
        if d.lockout_until is not None and d.lockout_until > now:
            lockout_remaining = d.lockout_until - now
        last_cmd_age = None
        if d.last_agent_command_at is not None:
            last_cmd_age = round(now - d.last_agent_command_at, 1)
        return {
            "on": d.on,
            "power_w": round(d.power_w, 1),
            "lockout_active": lockout_remaining > 0,
            "lockout_remaining_s": round(lockout_remaining, 1),
            "last_agent_command_age_s": last_cmd_age,
            "last_agent_command_intent": d.last_agent_command_intent,
        }

    def _build_snapshot(
        self,
        include_semantic: bool,
        include_events: bool,
        now: float | None = None,
    ) -> dict:
        """Internal: build the snapshot dict.  Caller must hold self._lock."""
        if now is None:
            now = time.monotonic()

        entities = []
        for e in self._entities:
            entities.append({
                "id": e.id,
                "bbox_xywh": list(e.bbox_xywh),
                "zones": list(e.zones),
                "pose": e.pose,
                "seconds_in_frame": round(now - e.first_seen, 1),
                "velocity_px_per_s": e.velocity_px_per_s,
            })

        snap: dict = {
            "timestamp": datetime.datetime.now().isoformat(),
            "entities": entities,
            "people_count": self._people_count,
            "audio": {
                "level_db": round(self._audio.level_db, 1),
                "top_classes": [
                    {"label": label, "confidence": round(conf, 2)}
                    for label, conf in self._audio.top_classes
                ],
                "dominant_class": self._audio.dominant_class,
                "speech_active": self._audio.speech_active,
                "recent_spike": self._audio.recent_spike,
                "spike_magnitude_db": round(self._audio.spike_magnitude_db, 1),
            },
            "devices": {
                alias: self._device_for_reasoner(d, now)
                for alias, d in self._devices.items()
            },
            "baselines": {
                "audio_mean_db": round(self._baselines.audio_mean_db, 1),
                "audio_std_db": round(self._baselines.audio_std_db, 1),
                "typical_occupancy": self._baselines.typical_occupancy,
                "power_idle_lamp_w": round(self._baselines.power_idle_lamp_w, 1),
                "power_idle_fan_w": round(self._baselines.power_idle_fan_w, 1),
                "ambient_audio_classes": list(self._baselines.ambient_audio_classes),
                "calibrated": self._baselines.calibrated,
            },
        }

        if include_semantic:
            snap["scene_description"] = self._scene_description
            snap["activity_summary"] = self._activity_summary

        if include_events:
            snap["recent_events"] = copy.deepcopy(self._recent_events)

        return snap

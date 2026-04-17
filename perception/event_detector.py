"""
Event detector — YOLO per-frame output → named, stable events.

This is the module that turns a stream of "here are N bounding boxes with
track IDs" into the structured events the rest of the system reasons about:
`new_person`, `lost_person`, `pose_change`, `zone_transition`. The Observer
(Layer 1) never sees raw YoloResult objects — it sees these events.

--- Why this module exists as a separate layer ---

YOLO gives us *continuous* machine state: 30 times a second it re-tells us
"there are 2 people, at (x,y) with this pose, with track IDs 7 and 12."
Most of that is redundant — nothing new happened in the last 33ms. The
Observer and Reasoner are expensive (API calls) and want *deltas*, not the
raw state. So we need a module that (a) holds state across frames, (b)
notices transitions, and (c) emits one event per real-world change.

That's all `EventDetector.tick()` does. You hand it a YoloResult each
tick; it returns a (possibly empty) list of Events.

--- The core problems and how we solve them ---

1. **Noise.** A person standing still has a bbox that jitters ±2-3 px; a
   pose keypoint drops below confidence for one frame; a tracker briefly
   loses an ID. If we treat every frame-to-frame difference as an event,
   we'd emit a flood. Fix: hysteresis. Every transition (pose, zone)
   needs to persist for N consecutive frames before we emit.

2. **Tracker instability.** BoT-SORT occasionally drops an ID for 1–2
   frames under occlusion, then picks the person back up with the *same*
   ID. We don't want lost_person + new_person events for that. Fix: a
   grace window (`EVENT_LOST_PERSON_GRACE_FRAMES`). A track is only
   "lost" after being absent for ~1s of frames.

3. **Identity across re-entry.** When a person walks fully out of frame
   and later returns, BoT-SORT assigns a *new* track ID — it can't tell
   this is the same human. We treat that as correct behavior: a
   `lost_person(id=N)` fires when the old track expires, and a
   `new_person(id=M)` fires when someone (possibly the same human, but
   we can't know from pixels alone) re-enters. Layer 0 does NOT attempt
   identity re-association; that's a semantic question the Observer or
   Reasoner decides. Downstream code must not assume track_id stability
   across exit/re-entry.

4. **Pose classification.** "Sitting" vs "standing" vs "walking" isn't
   in the YOLO output — it's derived from the 17 COCO keypoints' relative
   geometry (hip-knee-ankle verticals) plus center-x motion. See
   `_classify_pose` below.

--- Delivery model: pull, not push ---

The detector exposes a single synchronous method:

    events = detector.tick(yolo_engine.latest_result())

The orchestrator (`main.py`) calls this each loop iteration. No threads,
no queues, no async. Simple, testable, matches the threading story of
`camera.py` and `yolo_engine.py`: background workers for hardware I/O,
synchronous pull for everything else.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

import config
from perception.yolo_engine import YoloEntity, YoloResult
from perception.zone_map import zone_for_entity


# ---------------------------------------------------------------------------
# Public event schema
# ---------------------------------------------------------------------------

# Valid values for Event.type. Kept as a module-level tuple so downstream
# code (tests, the observer, routing) can import the canonical list rather
# than hard-coding strings.
EVENT_TYPES: tuple[str, ...] = (
    "new_person",
    "lost_person",
    "pose_change",
    "zone_transition",
)

# Valid values for a track's pose state. Keep in sync with `_classify_pose`.
POSE_STATES: tuple[str, ...] = ("standing", "sitting", "walking", "unknown")


@dataclass
class Event:
    """One structured event produced by the detector.

    All events share these common fields. The `payload` dict carries
    type-specific extras (documented per event type below).

    Payload contracts:
        new_person       {"bbox_xywh", "initial_pose"}
        lost_person      {"last_zones", "last_bbox_xywh", "frames_missing"}
        pose_change      {"from_pose", "to_pose"}
        zone_transition  {"from_zones", "to_zones"}
    """
    type: str                             # one of EVENT_TYPES
    ts: float                             # time.monotonic() when emitted
    track_id: Optional[int]               # None only for future non-person events
    zones: list[str]                      # zones at the moment of the event
    confidence: float                     # detection confidence of the underlying entity
    payload: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal per-track state
# ---------------------------------------------------------------------------

@dataclass
class _TrackState:
    """Everything we remember about one active track_id between ticks.

    `last_*` fields hold the *confirmed* value (what the detector currently
    treats as the truth). `pending_*` fields are the candidate for an
    in-progress transition; `*_streak` counts how many consecutive frames
    the candidate has held. When a streak reaches its threshold, we emit
    the corresponding event and promote pending -> last.
    """
    last_seen_ts: float
    last_conf: float
    last_bbox: tuple[float, float, float, float]
    last_center_x: float
    last_center_y: float

    last_zones: list[str]
    pending_zones: list[str]
    zone_streak: int

    last_pose: str
    pending_pose: str
    pose_streak: int
    walk_hold_remaining: int
    sit_hold_remaining: int

    frames_missing: int

    confirmed: bool             # False until track persists NEW_PERSON_CONFIRM_FRAMES
    frames_seen: int            # how many frames this track has been present


@dataclass
class _CooldownState:
    """Per-track cooldown for rate-limiting event emission.

    When an event fires, it starts a cooldown.  Subsequent transitions during
    the cooldown are suppressed but remembered (pending_value).  When the
    cooldown expires and the current state differs from the last emitted state,
    a catch-up event is emitted with the net change.  This compresses jitter
    (sitting→standing→walking→sitting becomes sitting→sitting = no event)
    while never losing genuine transitions.
    """
    last_emitted_ts: float      # monotonic time of last emitted event
    last_emitted_value: object  # pose string or zone list at emission
    pending_value: object       # latest suppressed value, or None
    pending_conf: float         # detection confidence at time of suppressed event


# ---------------------------------------------------------------------------
# Pose classifier
# ---------------------------------------------------------------------------

# COCO keypoint indices used here. YOLO-pose returns 17 keypoints in this
# order; we only need the four vertical landmarks of the torso and legs.
_LEFT_SHOULDER = 5
_RIGHT_SHOULDER = 6
_LEFT_HIP = 11
_RIGHT_HIP = 12
_LEFT_KNEE = 13
_RIGHT_KNEE = 14


def _mean_y(
    kp_xy: Optional[list[tuple[float, float]]],
    kp_cf: Optional[list[float]],
    ia: int,
    ib: int,
    min_conf: float,
) -> Optional[float]:
    """Return the mean y of two paired keypoints, or a single one, or None.

    "Confident" means the per-keypoint confidence is >= min_conf. If both
    of the pair pass, average them (more stable). If only one passes, use
    it (better than nothing). If neither, return None — the classifier
    will fall back to bbox geometry.
    """
    if kp_xy is None or kp_cf is None:
        return None
    if len(kp_xy) <= max(ia, ib):
        return None

    ya, yb = kp_xy[ia][1], kp_xy[ib][1]
    ca, cb = kp_cf[ia], kp_cf[ib]

    if ca >= min_conf and cb >= min_conf:
        return 0.5 * (ya + yb)
    if ca >= min_conf:
        return ya
    if cb >= min_conf:
        return yb
    return None


def _classify_pose(
    entity: YoloEntity,
    last_center_x: Optional[float],
    last_center_y: Optional[float],
    kp_min_conf: float,
    walk_min_dist_px: float,
) -> str:
    """Return one of POSE_STATES for this frame.

    Order matters: we check for ambiguous geometry first (return unknown),
    then sitting vs standing, then promote standing -> walking if the
    center moved enough since the previous frame (2D distance).

    Heuristics:
      * `sitting`  — thighs roughly horizontal: |knee_y − hip_y| is small
                     compared to torso length (|hip_y − shoulder_y|). Or
                     bbox aspect says short-and-wide when keypoints are
                     occluded.
      * `standing` — default upright pose; hip-to-knee span is a real
                     fraction of the torso.
      * `walking`  — `standing` + lateral center motion >= threshold.
      * `unknown`  — not enough signal, or ambiguous geometry (flat torso,
                     bbox wider than tall). Callers hold the last known
                     pose rather than flip.
    """
    cx, cy, w, h = entity.bbox_xywh

    shoulder_y = _mean_y(entity.keypoints_xy, entity.keypoints_conf,
                         _LEFT_SHOULDER, _RIGHT_SHOULDER, kp_min_conf)
    hip_y = _mean_y(entity.keypoints_xy, entity.keypoints_conf,
                    _LEFT_HIP, _RIGHT_HIP, kp_min_conf)
    knee_y = _mean_y(entity.keypoints_xy, entity.keypoints_conf,
                     _LEFT_KNEE, _RIGHT_KNEE, kp_min_conf)

    raw: str
    if shoulder_y is not None and hip_y is not None:
        torso_len = abs(hip_y - shoulder_y)
        # If the torso isn't meaningfully vertical (or the bbox is
        # flatter than it is tall), the geometry is ambiguous — treat
        # as unknown so we hold the last confirmed pose.
        if torso_len < w * 0.35 or w > h * 1.1:
            return "unknown"
        elif knee_y is not None:
            thigh_len = abs(knee_y - hip_y)
            # When sitting upright, knees are roughly level with hips
            # (thighs horizontal), so the vertical hip→knee span
            # collapses to well under half the torso length.
            raw = "sitting" if thigh_len < torso_len * 0.45 else "standing"
        else:
            # Knees occluded (classic: person sitting behind a desk).
            # Fall back to aspect ratio: a standing upright person's bbox
            # is usually ~2× taller than wide; a sitting-behind-desk
            # bbox collapses toward square.
            raw = "standing" if h > w * 1.6 else "sitting"
    elif w > 0 and h > 0:
        # No torso keypoints at all — use bbox aspect only.
        if w > h * 1.1:
            return "unknown"
        elif h < w * 1.6:
            raw = "sitting"
        else:
            raw = "standing"
    else:
        return "unknown"

    # Promote standing -> walking on real motion. Uses 2D distance so
    # walking toward/away from the camera registers, not just lateral.
    # Only checked when already classified as standing — a sitting person
    # doesn't become "walking" because their bbox shifted (YOLO jitter).
    if raw == "standing" and last_center_x is not None and last_center_y is not None:
        dx = cx - last_center_x
        dy = cy - last_center_y
        dist = math.sqrt(dx * dx + dy * dy)
        if dist >= walk_min_dist_px:
            raw = "walking"

    return raw


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class EventDetector:
    """Stateful event detector. Call `tick(yolo_result)` each frame.

    Not thread-safe — intended to be called from a single orchestrator
    loop. If you ever need it in a thread, wrap `tick` in a lock.
    """

    def __init__(
        self,
        *,
        kp_min_conf: float = config.EVENT_POSE_KP_MIN_CONF,
        pose_hysteresis_frames: int = config.EVENT_POSE_HYSTERESIS_FRAMES,
        zone_dwell_frames: int = config.EVENT_ZONE_DWELL_FRAMES,
        new_person_confirm_frames: int = config.EVENT_NEW_PERSON_CONFIRM_FRAMES,
        lost_person_grace_frames: int = config.EVENT_LOST_PERSON_GRACE_FRAMES,
        walk_min_dist_px: float = config.EVENT_WALK_MIN_DIST_PX,
        walk_hold_frames: int = config.EVENT_WALK_HOLD_FRAMES,
        sit_hold_frames: int = config.EVENT_SIT_HOLD_FRAMES,
        pose_cooldown_s: float = config.EVENT_POSE_COOLDOWN_S,
        zone_cooldown_s: float = config.EVENT_ZONE_COOLDOWN_S,
    ) -> None:
        # Thresholds are injected so tests can override them without
        # monkey-patching config.
        self._kp_min_conf = kp_min_conf
        self._pose_hyst = max(1, pose_hysteresis_frames)
        self._zone_dwell = max(1, zone_dwell_frames)
        self._new_confirm = max(1, new_person_confirm_frames)
        self._lost_grace = max(1, lost_person_grace_frames)
        self._walk_min_dist = walk_min_dist_px
        self._walk_hold = max(1, walk_hold_frames)
        self._sit_hold = max(1, sit_hold_frames)
        self._pose_cooldown = pose_cooldown_s
        self._zone_cooldown = zone_cooldown_s

        self._tracks: dict[int, _TrackState] = {}
        self._last_processed_ts: float = -1.0

        # Per-track cooldown state for rate-limiting events.
        self._pose_cd: dict[int, _CooldownState] = {}
        self._zone_cd: dict[int, _CooldownState] = {}

    # ---- stats / introspection (useful for the smoke test and future dashboard) ----

    def active_track_ids(self) -> list[int]:
        """Return track IDs currently being held in state (including grace-period)."""
        return sorted(self._tracks.keys())

    def pose_for(self, track_id: int) -> Optional[str]:
        """Return the confirmed pose state for a track, or None if unknown."""
        state = self._tracks.get(track_id)
        return state.last_pose if state else None

    def zones_for(self, track_id: int) -> list[str]:
        """Return the confirmed zone list for a track, or empty if unknown."""
        state = self._tracks.get(track_id)
        return list(state.last_zones) if state else []

    # ---- main API ----

    def tick(self, result: Optional[YoloResult]) -> list[Event]:
        """Consume one YoloResult; return any events it triggered.

        Safe to call with None (nothing happens). Safe to call twice
        with the same YoloResult timestamp (second call is a no-op) —
        useful if the orchestrator's tick runs faster than YOLO's.
        """
        if result is None:
            return []
        if result.timestamp == self._last_processed_ts:
            return []
        self._last_processed_ts = result.timestamp

        events: list[Event] = []
        ts = result.timestamp

        # Build a track_id -> entity view of this frame, filtered to persons
        # with a tracker ID. Non-persons and untracked persons can't
        # participate in our event model (we key everything on track_id).
        persons: dict[int, YoloEntity] = {}
        for ent in result.entities:
            if ent.cls_name == "person" and ent.track_id is not None:
                persons[ent.track_id] = ent

        seen_ids: set[int] = set()

        # Pass 1: walk every present track — either register it fresh
        # (new_person) or update its state and fire pose/zone transitions.
        for tid, ent in persons.items():
            seen_ids.add(tid)

            state = self._tracks.get(tid)
            if state is None:
                # Fresh track — start unconfirmed.  Don't emit new_person
                # yet; the track must persist for _new_confirm frames first
                # to filter out phantom YOLO detections (shadows, furniture
                # that briefly looks person-shaped).
                initial_pose = _classify_pose(
                    ent, last_center_x=None, last_center_y=None,
                    kp_min_conf=self._kp_min_conf,
                    walk_min_dist_px=self._walk_min_dist,
                )
                zones = zone_for_entity(ent, pose_hint=initial_pose)
                self._tracks[tid] = _TrackState(
                    last_seen_ts=ts,
                    last_conf=ent.conf,
                    last_bbox=ent.bbox_xywh,
                    last_center_x=ent.bbox_xywh[0],
                    last_center_y=ent.bbox_xywh[1],
                    last_zones=zones,
                    pending_zones=zones,
                    zone_streak=0,
                    last_pose=initial_pose,
                    pending_pose=initial_pose,
                    pose_streak=0,
                    walk_hold_remaining=0,
                    sit_hold_remaining=0,
                    frames_missing=0,
                    confirmed=False,
                    frames_seen=1,
                )
                continue

            # Use the last confirmed pose as a hint for zone lookup —
            # sitting people use bbox center instead of foot point.
            zones = zone_for_entity(ent, pose_hint=state.last_pose)

            # Existing track — update "seen" bookkeeping up front so the
            # pose/zone branches below can assume it's live.
            state.last_seen_ts = ts
            state.last_conf = ent.conf
            state.last_bbox = ent.bbox_xywh
            state.frames_missing = 0
            state.frames_seen += 1

            # Confirmation gate: track must persist _new_confirm frames
            # before we emit new_person and process pose/zone events.
            # This filters phantom detections that appear for a few frames
            # then vanish.
            if not state.confirmed:
                if state.frames_seen >= self._new_confirm:
                    state.confirmed = True
                    zones = zone_for_entity(ent, pose_hint=state.last_pose)
                    state.last_zones = zones
                    events.append(Event(
                        type="new_person",
                        ts=ts,
                        track_id=tid,
                        zones=zones,
                        confidence=ent.conf,
                        payload={
                            "bbox_xywh": ent.bbox_xywh,
                            "initial_pose": state.last_pose,
                        },
                    ))
                # Update center for motion tracking even while unconfirmed
                state.last_center_x = ent.bbox_xywh[0]
                state.last_center_y = ent.bbox_xywh[1]
                continue

            # Pose transition with hysteresis.
            raw_pose = _classify_pose(
                ent, last_center_x=state.last_center_x,
                last_center_y=state.last_center_y,
                kp_min_conf=self._kp_min_conf,
                walk_min_dist_px=self._walk_min_dist,
            )

            # Walk hold: if currently walking and the classifier says
            # "standing" (motion dropped), keep returning "walking" until
            # the hold timer expires. This prevents flicker during natural
            # stride pauses. Other poses (sitting, unknown) bypass the hold
            # — sitting down mid-walk is a real transition.
            if raw_pose == "walking":
                state.walk_hold_remaining = self._walk_hold
            elif raw_pose == "standing" and state.walk_hold_remaining > 0:
                state.walk_hold_remaining -= 1
                raw_pose = "walking"

            # Sit hold: if currently sitting and the classifier says
            # "standing" (keypoint jitter, leaning forward), keep returning
            # "sitting" until the hold expires. Walking bypasses the hold
            # — real motion overrides.
            if raw_pose == "sitting":
                state.sit_hold_remaining = self._sit_hold
            elif raw_pose == "standing" and state.sit_hold_remaining > 0:
                state.sit_hold_remaining -= 1
                raw_pose = "sitting"

            # Update center for next frame's motion check.
            state.last_center_x = ent.bbox_xywh[0]
            state.last_center_y = ent.bbox_xywh[1]

            if raw_pose == "unknown":
                # Hold the confirmed pose; reset any in-flight candidate
                # so a single garbage frame can't "count toward" a flip.
                state.pending_pose = state.last_pose
                state.pose_streak = 0
            elif raw_pose == state.last_pose:
                state.pending_pose = state.last_pose
                state.pose_streak = 0
            elif raw_pose == state.pending_pose:
                state.pose_streak += 1
                if state.pose_streak >= self._pose_hyst:
                    # Cooldown gate: suppress if too soon after last emit
                    cd = self._pose_cd.get(tid)
                    now_mono = time.monotonic()
                    if cd is not None and (now_mono - cd.last_emitted_ts) < self._pose_cooldown:
                        # Within cooldown — remember for catch-up later
                        cd.pending_value = raw_pose
                        cd.pending_conf = ent.conf
                    else:
                        # Cooldown expired or first event — emit
                        events.append(Event(
                            type="pose_change",
                            ts=ts,
                            track_id=tid,
                            zones=zones,
                            confidence=ent.conf,
                            payload={
                                "from_pose": state.last_pose,
                                "to_pose": raw_pose,
                            },
                        ))
                        self._pose_cd[tid] = _CooldownState(
                            last_emitted_ts=now_mono,
                            last_emitted_value=raw_pose,
                            pending_value=None,
                            pending_conf=0.0,
                        )
                    state.last_pose = raw_pose
                    state.pose_streak = 0
            else:
                # New candidate — start a fresh streak at 1.
                state.pending_pose = raw_pose
                state.pose_streak = 1

            # Zone transition with dwell. Note: zone lists from
            # zone_for_entity are deterministic in order (iteration over
            # config.ZONES), so list equality is safe.
            if zones == state.last_zones:
                state.pending_zones = state.last_zones
                state.zone_streak = 0
            elif zones == state.pending_zones:
                state.zone_streak += 1
                if state.zone_streak >= self._zone_dwell:
                    # Cooldown gate: suppress if too soon after last emit
                    cd = self._zone_cd.get(tid)
                    now_mono = time.monotonic()
                    if cd is not None and (now_mono - cd.last_emitted_ts) < self._zone_cooldown:
                        # Within cooldown — remember for catch-up later
                        cd.pending_value = zones
                        cd.pending_conf = ent.conf
                    else:
                        # Cooldown expired or first event — emit
                        events.append(Event(
                            type="zone_transition",
                            ts=ts,
                            track_id=tid,
                            zones=zones,
                            confidence=ent.conf,
                            payload={
                                "from_zones": state.last_zones,
                                "to_zones": zones,
                            },
                        ))
                        self._zone_cd[tid] = _CooldownState(
                            last_emitted_ts=now_mono,
                            last_emitted_value=list(zones),
                            pending_value=None,
                            pending_conf=0.0,
                        )
                    state.last_zones = zones
                    state.zone_streak = 0
            else:
                state.pending_zones = zones
                state.zone_streak = 1

        # Pass 1b: catch-up emission for expired cooldowns.
        # If a cooldown has expired and the pending value differs from the
        # last emitted value, emit the net-change event now.
        now_mono = time.monotonic()

        for tid, cd in list(self._pose_cd.items()):
            if (
                cd.pending_value is not None
                and (now_mono - cd.last_emitted_ts) >= self._pose_cooldown
                and cd.pending_value != cd.last_emitted_value
                and tid in self._tracks
            ):
                state = self._tracks[tid]
                events.append(Event(
                    type="pose_change",
                    ts=ts,
                    track_id=tid,
                    zones=state.last_zones,
                    confidence=cd.pending_conf,
                    payload={
                        "from_pose": cd.last_emitted_value,
                        "to_pose": cd.pending_value,
                    },
                ))
                cd.last_emitted_ts = now_mono
                cd.last_emitted_value = cd.pending_value
                cd.pending_value = None
            elif cd.pending_value is not None and (now_mono - cd.last_emitted_ts) >= self._pose_cooldown:
                # Cooldown expired but pending == last emitted → just clear
                cd.pending_value = None

        for tid, cd in list(self._zone_cd.items()):
            if (
                cd.pending_value is not None
                and (now_mono - cd.last_emitted_ts) >= self._zone_cooldown
                and cd.pending_value != cd.last_emitted_value
                and tid in self._tracks
            ):
                state = self._tracks[tid]
                events.append(Event(
                    type="zone_transition",
                    ts=ts,
                    track_id=tid,
                    zones=cd.pending_value,
                    confidence=cd.pending_conf,
                    payload={
                        "from_zones": cd.last_emitted_value,
                        "to_zones": cd.pending_value,
                    },
                ))
                cd.last_emitted_ts = now_mono
                cd.last_emitted_value = list(cd.pending_value)
                cd.pending_value = None
            elif cd.pending_value is not None and (now_mono - cd.last_emitted_ts) >= self._zone_cooldown:
                cd.pending_value = None

        # Pass 2: age out tracks that weren't in this frame. We don't
        # immediately fire lost_person — give the tracker a grace window
        # to re-acquire under brief occlusion (motion blur, overlap).
        for tid in list(self._tracks.keys()):
            if tid in seen_ids:
                continue
            state = self._tracks[tid]
            state.frames_missing += 1
            if state.frames_missing >= self._lost_grace:
                if state.confirmed:
                    # Real track that persisted long enough — report departure.
                    events.append(Event(
                        type="lost_person",
                        ts=ts,
                        track_id=tid,
                        zones=[],
                        confidence=state.last_conf,
                        payload={
                            "last_zones": state.last_zones,
                            "last_bbox_xywh": state.last_bbox,
                            "frames_missing": state.frames_missing,
                        },
                    ))
                # Unconfirmed tracks that vanish are silently dropped —
                # they were phantom detections that never persisted long
                # enough to be reported as new_person.
                del self._tracks[tid]
                self._pose_cd.pop(tid, None)
                self._zone_cd.pop(tid, None)

        return events


# ---------------------------------------------------------------------------
# Tiny convenience for callers that prefer a function over a class —
# primarily for quick REPL poking. Not used by main.py (which should own
# the EventDetector instance for the whole process lifetime so state is
# preserved across ticks).
# ---------------------------------------------------------------------------

_singleton: Optional[EventDetector] = None


def tick(result: Optional[YoloResult]) -> list[Event]:
    """Module-level convenience wrapping a lazily-created singleton detector."""
    global _singleton
    if _singleton is None:
        _singleton = EventDetector()
    return _singleton.tick(result)


def reset() -> None:
    """Drop the module-level singleton's state. Tests use this."""
    global _singleton
    _singleton = None


def _now() -> float:
    """Expose time.monotonic() so callers don't need to import time just for ts logging."""
    return time.monotonic()

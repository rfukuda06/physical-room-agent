# DESIGN.md — How the pieces fit

Living document. **Update this every time a new module is implemented or a data flow changes.** The goal is that anyone (including future-you) can read this file and understand exactly how data moves through the system without re-reading every source file.

Legend:
- ✅ Implemented and wired up
- 🟡 Stub exists, not yet functional
- ⬜ Not started

---

## Current state (Day 1 — end of Block 5)

```
┌────────────────────────────────────────────────────────────────────────┐
│                        HARDWARE (physical world)                        │
│                                                                         │
│   MacBook webcam ──▶ USB pipe ──▶ OpenCV                               │
│   MacBook mic   ⬜  (not yet consumed)                                 │
│   Kasa plugs    ⬜  (not yet discovered)                               │
└────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│  ✅ perception/camera.py — CameraCapture                                │
│                                                                         │
│   Background thread reads cv2.VideoCapture(0) at 30 FPS / 720p.         │
│   Publishes two separate streams from the same hardware read:           │
│                                                                         │
│     (a) latest_frame()         → full-res BGR ndarray, live             │
│         └── consumed by YoloEngine                                      │
│                                                                         │
│     (b) buffer_snapshot(sec)   → rolling 10s @ 10 FPS / 720p            │
│         └── future: Observer/Reasoner will read this to see             │
│             "what led up to" an event                                   │
│                                                                         │
│   Locks: separate `_latest_lock` and `_buffer_lock` so a slow           │
│   buffer reader can't starve the live feed.                             │
└────────────────────────────────────────────────────────────────────────┘
                                    │
              camera.latest_frame() │ (shallow copy, BGR ndarray)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│  ✅ perception/yolo_engine.py — YoloEngine                              │
│                                                                         │
│   Own daemon thread. On each new camera frame:                          │
│     1. model.track(frame, persist=True, tracker="botsort.yaml")         │
│        → Ultralytics `Results` object from yolo26n-pose.pt              │
│     2. _extract_entities() converts GPU tensors → plain dataclasses     │
│     3. results[0].plot() renders boxes + skeleton onto a BGR copy       │
│     4. Publishes the latest YoloResult under _result_lock               │
│                                                                         │
│   Device: "mps" (Apple Silicon GPU) with auto-fallback to "cpu".        │
│   Latency: ~33ms/frame avg on MPS. Right at the 30 FPS budget; the      │
│   loop drops (never queues) frames that arrive mid-inference.           │
│                                                                         │
│   Output shape (what latest_result() returns):                          │
│                                                                         │
│     YoloResult                                                          │
│       ├─ timestamp: float                                               │
│       ├─ frame_shape: (H, W)                                            │
│       ├─ infer_ms: float                                                │
│       ├─ annotated_frame: BGR ndarray (for preview/dashboard)           │
│       └─ entities: list[YoloEntity]                                     │
│            ├─ track_id: int | None      (stable across frames via      │
│            │                             BoT-SORT)                      │
│            ├─ cls_name: "person"        (pose model → only 1 class)    │
│            ├─ bbox_xywh: (cx, cy, w, h) (pixels)                       │
│            ├─ conf: float                                               │
│            ├─ keypoints_xy:   17 × (x, y)    (COCO skeleton)           │
│            └─ keypoints_conf: 17 × float                                │
│                                                                         │
│   NOT in YoloResult (derived later by event_detector):                  │
│     pose class, zone membership, entering/leaving, velocity, duration.  │
└────────────────────────────────────────────────────────────────────────┘
                                    │
         engine.latest_result()     │ (copy — annotated frame cloned,
                                    │  entities are value types)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│  ✅ perception/event_detector.py — Block 5                              │
│                                                                         │
│   Stateful transducer: YoloResult per frame → list[Event] per tick.    │
│   Owns per-track state between calls; all derivations (pose class,     │
│   zone membership, lifecycle) live here, NOT in YOLO.                  │
│                                                                         │
│   Public API:                                                          │
│     EventDetector().tick(result) -> list[Event]                        │
│     detector.active_track_ids() -> list[int]                           │
│     detector.pose_for(track_id) -> str | None                          │
│                                                                         │
│   Noise controls (see config.EVENT_*):                                 │
│     • POSE_HYSTERESIS_FRAMES — pose must persist N frames to flip      │
│     • ZONE_DWELL_FRAMES      — zone set must persist N frames to flip  │
│     • LOST_PERSON_GRACE      — ~1s of absence before lost_person fires │
│     • WALK_MIN_DX_PX         — center-x delta that promotes            │
│                                 standing → walking                      │
│                                                                         │
│   NOT implemented here (deferred):                                     │
│     object_moved — non-person tracked class displacement. Demo is      │
│     person-centric; add when there's a use case that needs it.         │
└────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│  ✅ perception/zone_map.py — Block 4                                    │
│                                                                         │
│   Reads config.ZONES (polygon dict, pixel vertices) lazily on first     │
│   query and caches each zone as an int32 (N, 1, 2) array for            │
│   cv2.pointPolygonTest. Exposes:                                        │
│                                                                         │
│     zones_for_point(x, y)   -> list[str]                                │
│     zone_for_entity(entity) -> list[str]                                │
│     reload_zones()          -> None   (invalidate cache after edits)    │
│                                                                         │
│   zone_for_entity picks the query point per-entity:                     │
│     person + both ankle kps >= ANKLE_CONF_MIN (0.5) → ankle midpoint    │
│     person + occluded ankles                       → bbox bottom-center │
│     non-person                                     → bbox center        │
│                                                                         │
│   Ships its own click-to-define CLI (python -m perception.zone_map)     │
│   for populating config.ZONES against the live camera. Walkthrough      │
│   in SETUP.md §4c.                                                      │
├────────────────────────────────────────────────────────────────────────┤
│  🟡 perception/audio.py — (Block 6, Day 1)                              │
│     sounddevice mic capture + dB meter + YAMNet 521-class tagging.     │
├────────────────────────────────────────────────────────────────────────┤
│  🟡 perception/plugs.py — (Block 7, Day 1)                              │
│     python-kasa discovery + power reads + on/off control.               │
├────────────────────────────────────────────────────────────────────────┤
│  🟡 agents/*.py — (Day 2) observer, reasoner, world_state, routing,     │
│                    baselines, decisions. All stubs right now.           │
├────────────────────────────────────────────────────────────────────────┤
│  🟡 actuators/*.py — (Day 2) TTS speaker + smart_plug wrapper. Stubs.   │
├────────────────────────────────────────────────────────────────────────┤
│  🟡 server/*.py — (Day 3) FastAPI + WebSocket hub. Stubs.               │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Data contracts (what each module promises to emit)

### CameraCapture
- `latest_frame() -> np.ndarray | None` — shape `(H, W, 3)`, dtype `uint8`, BGR.
- `buffer_snapshot(seconds) -> list[(monotonic_ts, np.ndarray)]`.
- Thread-safe; returns fresh copies.

### YoloEngine
- `latest_result() -> YoloResult | None` — see dataclass above.
- Thread-safe; consumer never blocks on inference.
- Emits **every** processed frame (no filtering — event_detector decides what counts as an event).

### zone_map
- `zones_for_point(x, y) -> list[str]` — every zone whose polygon contains `(x, y)`, or `[]`.
- `zone_for_entity(entity: YoloEntity) -> list[str]` — entity-aware query point (ankle midpoint → bbox-bottom fallback → bbox center for non-persons).
- `reload_zones() -> None` — force re-read of `config.ZONES` on next query.
- Reads `config.ZONES` (polygon dict). Cache is thread-safe; read-only after load.

### event_detector

Emits one canonical `Event` dataclass. Every event carries:

```
type: str               # "new_person" | "lost_person" | "pose_change" | "zone_transition"
ts: float               # monotonic timestamp from the YoloResult
track_id: int | None    # BoT-SORT track id
zones: list[str]        # zones at the moment of the event
confidence: float       # underlying YOLO detection confidence
payload: dict           # type-specific extras (see below)
```

Per-type payload:

| type              | payload keys                                         |
| ----------------- | ---------------------------------------------------- |
| `new_person`      | `bbox_xywh`, `initial_pose`                          |
| `lost_person`     | `last_zones`, `last_bbox_xywh`, `frames_missing`     |
| `pose_change`     | `from_pose`, `to_pose`                               |
| `zone_transition` | `from_zones`, `to_zones`                             |

Pose states: one of `standing | sitting | walking | down | unknown`.
`unknown` is never emitted as an event value; it's an internal "hold
last pose" signal when keypoints are too unreliable to classify.

**Identity assumption (load-bearing):** track IDs are per-lifetime only.
When a person walks fully out and returns, BoT-SORT gives them a *new*
id, so Layer 0 emits `lost_person(id=N)` followed later by
`new_person(id=M)` for what's physically the same human. Layer 0 does
not attempt identity re-association; that's a semantic call the
Observer or Reasoner makes from context. Any downstream code that
wants "same person re-entered" semantics must join on something other
than track_id.

Event `type` strings are mirrored in `config.REASONER_ALWAYS` so the
Reasoner routing policy (`agents/routing.py`) matches them verbatim.

---

## Zone-map assumptions (load-bearing for accuracy)

The zone system relies on a **ground-plane projection**: a single camera
can only give us 3D position if we assume the queried pixel is on the
floor. That's true for a person's feet, so zones are drawn on the
*floor*, not on objects, and queries use a foot-point (ankle midpoint,
falling back to bbox-bottom when ankles are occluded). Multi-zone
membership is intentional — a point between desk and door legitimately
belongs to both, and downstream code decides what to do with it.

These assumptions hold only under specific camera placement (≥1.8 m
high, tilted ~20° downward, facing across the room). See `SETUP.md` §4b
for the full checklist. **If the camera moves, zones drift and must be
re-captured** via `python -m perception.zone_map` (walkthrough in
`SETUP.md` §4c).

---

## Runtime threads (currently)

```
┌── MainThread ────────── interactive preview loop (cv2.imshow, keys)
├── camera-capture ────── cv2.VideoCapture.read() → _latest_frame, buffer
└── yolo-engine ────────── pulls latest_frame, runs model.track, publishes
```

No IPC, no queues, no async — just shared state behind locks. Keep it this way until we *need* something more.

---

## Update discipline

Add to this file when any of the following change:
- A new module becomes non-stub (flip ⬜/🟡 → ✅, describe its data contract).
- A module's output shape changes (update the dataclass block).
- A thread is added or removed.
- A new data flow arrow between modules is introduced.
- Camera-placement assumptions change (height, tilt, mount). Update
  `SETUP.md` §4b and re-run `python -m perception.zone_map`.

Don't let this doc drift. If you edit a module and don't edit this file, the next session will have to reverse-engineer what you did.

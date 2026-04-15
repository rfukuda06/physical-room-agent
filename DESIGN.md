# DESIGN.md — How the pieces fit

Living document. **Update this every time a new module is implemented or a data flow changes.** The goal is that anyone (including future-you) can read this file and understand exactly how data moves through the system without re-reading every source file.

Legend:
- ✅ Implemented and wired up
- 🟡 Stub exists, not yet functional
- ⬜ Not started

---

## Current state (Day 1 — end of Block 3)

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
│   Background thread reads cv2.VideoCapture(0) at 30 FPS / 1080p.        │
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
│  🟡 perception/event_detector.py — (Block 5, Day 1)                     │
│                                                                         │
│   Will consume YoloResult streams and emit named events:                │
│     new_person, lost_person, zone_transition, pose_change,              │
│     entering, leaving, hand_raised, etc.                                │
│                                                                         │
│   Logic lives here (NOT in YOLO):                                       │
│     • keypoint geometry → pose class (knee/hip/shoulder angles)         │
│     • bbox center ∈ zone polygon → zone_transition                      │
│     • track_id first seen / last seen → new_person, lost_person         │
└────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│  🟡 perception/zone_map.py — (Block 4, Day 1)                           │
│     Will load zones.json (pixel polygons) and answer "is (x,y) in       │
│     zone Z?" for event_detector.                                        │
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

Don't let this doc drift. If you edit a module and don't edit this file, the next session will have to reverse-engineer what you did.

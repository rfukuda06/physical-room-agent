# DESIGN.md — How the pieces fit

Living document. **Update this every time a new module is implemented or a data flow changes.** The goal is that anyone (including future-you) can read this file and understand exactly how data moves through the system without re-reading every source file.

Legend:
- ✅ Implemented and wired up
- 🟡 Stub exists, not yet functional
- ⬜ Not started

---

## Current state (Day 3 — lamp/fan control rollout complete)

```
┌────────────────────────────────────────────────────────────────────────┐
│                        HARDWARE (physical world)                        │
│                                                                         │
│   MacBook webcam ──▶ USB pipe ──▶ OpenCV                               │
│   MacBook mic ──▶ PortAudio ──▶ sounddevice (device 1)                 │
│   Kasa KP125M plugs ──▶ Wi-Fi/KLAP ──▶ python-kasa ──▶ PlugManager    │
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
│         └── consumed by Observer (resize → 384px, JPEG, 258 tokens)    │
│             to see "what led up to" an event                            │
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
│     detector.zones_for(track_id) -> list[str]   (added Day 2 Block 1) │
│                                                                         │
│   Noise controls (see config.EVENT_*):                                 │
│     • POSE_HYSTERESIS_FRAMES — pose must persist N frames to flip      │
│     • ZONE_DWELL_FRAMES      — zone set must persist N frames to flip  │
│     • LOST_PERSON_GRACE      — ~1s of absence before lost_person fires │
│     • WALK_MIN_DIST_PX       — 2D center-point distance that promotes  │
│                                 standing → walking (x + y, not x-only) │
│     • WALK_HOLD_FRAMES       — once walking, hold for N frames through │
│                                 stride pauses before reverting to       │
│                                 standing (~0.5s). Sitting bypasses hold.│
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
│     zones_for_point(x, y)            -> list[str]                       │
│     zone_for_entity(entity, pose_hint) -> list[str]                     │
│     reload_zones()                   -> None  (invalidate cache)        │
│                                                                         │
│   zone_for_entity picks the query point per-entity + pose:              │
│     any person + confident ankles                  → ankle midpoint     │
│     sitting + occluded ankles                      → bbox center        │
│     standing/walking + occluded ankles             → bbox bottom-center │
│     non-person                                     → bbox center        │
│                                                                         │
│   Ships its own click-to-define CLI (python -m perception.zone_map)     │
│   for populating config.ZONES against the live camera. Walkthrough      │
│   in SETUP.md §4c.                                                      │
┌────────────────────────────────────────────────────────────────────────┐
│  ✅ perception/audio.py — Block 6                                       │
│                                                                         │
│   Two concurrent workers: a PortAudio callback (sounddevice-managed)    │
│   appends 1024-sample chunks to a ring buffer and computes per-chunk    │
│   RMS dB.  A daemon thread ("audio-classify") wakes every 500 ms,      │
│   reads the last 1 s of audio, runs YAMNet (521 → 32 whitelisted       │
│   classes), applies temporal smoothing, and publishes events.           │
│                                                                         │
│   Public API:                                                          │
│     AudioMonitor(device_index, sample_rate, …)                         │
│       .start() / .stop()                                                │
│       .tick()          -> list[Event]    # drain pending audio events   │
│       .latest_state()  -> AudioState     # snapshot for WorldState     │
│       .stats()         -> dict           # diagnostics                 │
│                                                                         │
│   Temporal smoothing: a class must persist ≥ YAMNET_PERSISTENCE_WINDOWS │
│   (2) consecutive 500 ms windows before reporting.  Speech uses the     │
│   same threshold for both on and off transitions (symmetric hysteresis).│
│                                                                         │
│   dB spike detection: fires when current dB exceeds a 30 s rolling      │
│   mean by ≥ AUDIO_SPIKE_DB_THRESHOLD (25 dB). 3 s cooldown.            │
│                                                                         │
│   AudioClassifier is an ABC — YamNetClassifier is the current impl;    │
│   swap in CLAP / BEATs / EfficientAT without changing AudioMonitor.    │
└────────────────────────────────────────────────────────────────────────┘
                                    │
              audio_monitor.tick()  │ (list[Event] — same Event dataclass
                                    │  as event_detector; track_id=None)
                                    ▼
                           [ orchestrator merges with YOLO events ]
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│  ✅ main.py — Orchestrator (Block 8, updated Day 2 Block 3)             │
│                                                                         │
│   Wires all Layer 0 components + Observer into a monitoring loop:       │
│     1. Starts CameraCapture (background thread)                         │
│     2. Starts YoloEngine (background thread)                            │
│     3. Starts AudioMonitor (PortAudio callback + classify thread)       │
│     4. Creates EventDetector (synchronous, pulled each tick)            │
│     5. Starts PlugManager (async loop + background discover thread)     │
│     6. Creates WorldState (shared in-memory model)                      │
│     6b. Calibration phase (Day 2 Block 2):                              │
│         - CalibrationCollector.run() blocks ~30s                        │
│         - Polls audio/YOLO/plugs, accumulates samples                   │
│         - Keeps WorldState warm (update_* calls each tick)              │
│         - Video window shows progress bar + countdown overlay           │
│         - Console prints status every 5s                                │
│         - Stores Baselines via world.set_baselines()                    │
│     6c. Creates Observer + ObserverWorker (Day 2 Block 3):              │
│         - ObserverWorker starts daemon thread                           │
│         - Periodic refresh timer runs independently                     │
│     7. Main loop:                                                       │
│        - detector.tick(engine.latest_result()) → YOLO events            │
│        - audio.tick()                          → audio events           │
│        - merge → print to console + render on video overlay             │
│        - world.update_from_yolo/audio/devices/push_event                │
│        - push events to ObserverWorker (non-blocking)                   │
│        - poll ObserverWorker for Beat 1 results:                        │
│          → print [BEAT 1] narration to console + overlay                │
│          → call should_call_reasoner() for routing decision             │
│                                                                         │
│   Video window shows: YOLO boxes/skeleton + zone polygons + rolling     │
│   event log (top-left) + track status (bottom-left) + audio state       │
│   (bottom-right) + Beat 1 narrations in event log.                      │
│   Console prints one line per event + status every 2s.                  │
│   Press 'd' in video window to dump WorldState snapshot to console.     │
│                                                                         │
│   Graceful shutdown on Ctrl+C or 'q' keypress. Prints event summary.   │
│                                                                         │
│   Also publishes everything to the DashboardBroadcaster for the        │
│   live dashboard (Day 3 Phase 1):                                       │
│     - publish_event on every merged event                               │
│     - publish_snapshot (throttled 10 Hz) on world state                 │
│     - publish_narration on each Observer poll result                    │
│     - publish_routing on each Reasoner routing decision                 │
│     - publish_frame (throttled 15 FPS) of the annotated BGR frame       │
│                                                                         │
│   NOT YET WIRED (future):                                              │
│     - Next.js frontend (Day 3 Phase 3+ beyond dashboard Phase 1-3)     │
└────────────────────────────────────────────────────────────────────────┘
                                    │
├────────────────────────────────────────────────────────────────────────┤
│  ✅ perception/plugs.py — PlugManager (Block 7, Day 1)                  │
│     python-kasa discovery + background power polling + on/off control.  │
├────────────────────────────────────────────────────────────────────────┤
│  ✅ agents/world_state.py — WorldState (Day 2 Block 1)                   │
│     Thread-safe in-memory model. Aggregates YOLO, audio, plug state.   │
│     Updated every tick. Snapshotable for LLM prompts.                  │
├────────────────────────────────────────────────────────────────────────┤
│  ✅ agents/routing.py — Hybrid Reasoner routing policy                   │
├────────────────────────────────────────────────────────────────────────┤
│  ✅ agents/baselines.py — CalibrationCollector (Day 2 Block 2)            │
│     30s startup phase. Learns audio floor, power profile, occupancy,   │
│     and persistent YAMNet classes. Stores Baselines in WorldState.     │
├────────────────────────────────────────────────────────────────────────┤
│  ✅ agents/observer.py — Observer + ObserverWorker (Day 2 Block 3)       │
│     Gemini 2.5 Flash integration. Event-driven + 45s periodic refresh.  │
│     Threaded worker with 0.5s debounce. Fallback on API failure.        │
│  ✅ agents/reasoner.py — Reasoner + ReasonerWorker (Day 2/3 boundary)    │
│     Claude Sonnet integration. Receives Observer output + full world     │
│     context. Returns ReasonerOutput with lamp/fan/lamp_reason/fan_reason │
│     fields plus narration, speak, alert, reasoning.                      │
│  ✅ agents/decisions.py — DecisionEngine (Day 2/3 boundary)              │
│     Translates ReasonerOutput into physical actions with 5 code-enforced │
│     guardrails (in order): (1) idempotency — skip if device already in  │
│     requested state; (2) cooldown — at most one agent toggle per device  │
│     per DEVICE_COOLDOWN_S; (3) override lockout — back off for           │
│     MANUAL_OVERRIDE_LOCKOUT_S after a manual user toggle; (4) no-person │
│     ON guard — never command ON when WorldState shows 0 people; (5)     │
│     plug-unreachable — refuse if PlugManager.is_available() is False.   │
│     Every accepted or refused decision is published to the broadcaster.  │
│     Constructor: DecisionEngine(plugs, speaker, world).                  │
│  ✅ agents/empty_room_watcher.py — EmptyRoomWatcher (Day 2/3 boundary)  │
│     Pure-logic debounce that fires a `room_empty_confirmed` callback     │
│     exactly once after the room has been empty for EMPTY_ROOM_DEBOUNCE_S│
│     following a confirmed ≥1 → 0 person-count transition. Cold-start-   │
│     empty rooms are silently ignored until the first person appears.     │
├────────────────────────────────────────────────────────────────────────┤
│  ✅ actuators/speaker.py — Speaker, TTS beat queue (Day 2/3 boundary)   │
│     Wraps edge-tts/pyttsx3. Exposes enqueue_beat1/enqueue_beat2.        │
│  ✅ actuators/smart_plug.py — thin wrapper; plug control goes via        │
│     PlugManager. Wired through DecisionEngine.                           │
├────────────────────────────────────────────────────────────────────────┤
│  ✅ server/broadcaster.py — DashboardBroadcaster (Day 3 Phase 1)        │
│     Thread-safe fan-out from sync producers (main loop) to async       │
│     FastAPI consumers (WS + MJPEG). Uses run_coroutine_threadsafe to   │
│     cross the thread boundary. Per-client asyncio.Queue + drop-oldest  │
│     backpressure. Latest-JPEG slot with threading.Condition for MJPEG. │
│  ✅ server/app.py — FastAPI app (Day 3 Phase 1)                         │
│     Routes: GET /, GET /config, GET /video/stream (MJPEG),             │
│     WS /ws/state. run_server_in_thread() launches uvicorn on a daemon. │
│  🟡 server/events.py — separate stub; broadcaster covers this role now. │
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

Pose states: one of `standing | sitting | walking | unknown`.
`unknown` is never emitted as an event value; it's an internal "hold
last pose" signal when keypoints are too unreliable or geometry is
ambiguous (e.g. flat torso, bbox wider than tall).

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

### AudioMonitor

- `tick() -> list[Event]` — drain accumulated audio events since last call.
- `latest_state() -> AudioState | None` — current audio snapshot.
- `stats() -> dict` — buffer chunks, dB, classify latency, whitelist count.
- Thread-safe; events never lost (atomic swap on tick).

Audio events use the same `Event` dataclass as YOLO events, with
`track_id=None` and `zones=[]` (audio is not spatial or person-specific).

| type                    | payload keys                                  |
| ----------------------- | --------------------------------------------- |
| `unusual_sound_class`   | `class_name`, `confidence`, `db_level`        |
| `audio_spike`           | `current_db`, `baseline_db`, `delta_db`       |
| `speech_start`          | `confidence`, `db_level`                      |
| `speech_end`            | `duration_seconds`                            |

Speech events fire on *transitions only* — silence→speech emits
`speech_start`, speech→silence emits `speech_end`.  Continuing speech
is tracked silently via `AudioState.speech_active`.  Both transitions
use a dedicated 4-window hysteresis (`_SPEECH_ON_WINDOWS` /
`_SPEECH_OFF_WINDOWS` = 4 × 500 ms = 2 s) so brief utterances (a
cough, a single word) don't trigger `speech_start`, and natural
pauses between sentences don't trigger `speech_end`.

`AudioState` snapshot:

```
AudioState
  ├─ audio_level_db: float         # current RMS dB (0 = full scale)
  ├─ top_classes: [(name, conf)]   # non-speech, filtered + smoothed
  ├─ dominant_class: str           # top label or "speech" / "silence"
  ├─ speech_active: bool           # True while speech is ongoing
  ├─ recent_spike: bool            # True if spike fired recently
  ├─ spike_magnitude_db: float     # delta of last spike (0 if none)
  └─ timestamp: float              # monotonic
```

`unusual_sound_class` is in `config.REASONER_ALWAYS` — the Reasoner
always fires on novel non-speech sounds (door, glass, alarm, etc.).

**YAMNet limitations (load-bearing for Observer/Reasoner prompt design):**

YAMNet was trained on YouTube audio clips, not real-time room recordings
from a laptop microphone.  In practice its reliable signals are:

1. **dB spike** — the most trustworthy audio signal.  Pure loudness
   detection, no classification needed.
2. **Speech detection** — high confidence, works well.
3. **A handful of loud/distinctive sounds** — knocking, music, alarms
   when close to the mic.

For the majority of the 30 whitelisted classes (coughing, footsteps,
glass breaking, door opening, etc.) YAMNet is **unreliable from a
laptop mic at room distance**.  Quiet or brief sounds often don't
register at all.

**Rules for the Observer and Reasoner:**

- **Do not over-trust** `unusual_sound_class` — treat it as a hint,
  not ground truth.  Use the camera frame to confirm.
- **Do not assume nothing happened** because YAMNet didn't classify a
  sound.  Absence of a classification does NOT mean absence of the
  event — the model simply may not have picked it up.
- **Audio spikes fire on ambient noise too** (chair scrape, AC
  cycling, random bumps).  If the Observer can't determine what caused
  a spike from the camera frame, it should say so honestly rather than
  hallucinate an explanation.  "I heard a loud sound but couldn't
  identify what caused it" is a valid observation.

### PlugManager

- `start()` — starts background asyncio loop in a daemon thread.
- `discover(timeout=15) -> bool` — blocks until both plugs found or timeout; fires background polling task on success.
- `state(alias) -> PlugState | None` — most recent polled snapshot, or None if plug not yet found.
- `all_states() -> dict[str, PlugState]` — all known plug states.
- `turn_on(alias) -> bool` / `turn_off(alias) -> bool` — synchronous control; refreshes `PlugState` immediately after toggle.
- `stop()` — cancels polling task, shuts down event loop.

`PlugState` snapshot:

```
PlugState
  ├─ alias: str          # "light" or "fan"
  ├─ is_on: bool
  ├─ power_w: float      # current draw in watts (from KP125M energy module)
  ├─ voltage_v: float    # volts
  ├─ current_a: float    # amps
  └─ ts: float           # monotonic timestamp of last poll
```

Discovery strategy: UDP broadcast first, direct-IP fallback for any alias not surfaced by broadcast (IPs can rotate on hotspot restarts, so alias matching is the authoritative source).

### WorldState (Day 2 Block 1)

Thread-safe in-memory world model. Aggregates all Layer 0 signals into
one queryable, JSON-serializable object. Updated every main-loop tick.
Snapshotable for LLM prompt building.

**Public API:**
- `update_from_yolo(result, detector)` — rebuild entity list from YOLO + EventDetector
- `update_audio(audio_state)` — copy latest AudioState
- `update_devices(plugs)` — copy latest plug states
- `push_event(event)` — append to 50-event ring buffer
- `set_baselines(baselines)` — store calibration results (Block 2)
- `apply_observer_update(dict)` — merge semantic fields from Observer (Block 3)
- `apply_reasoner_update(dict)` — merge semantic fields from Reasoner (Block 4)
- `snapshot() -> dict` — full deep copy, JSON-serializable
- `snapshot_for_observer() -> dict` — lighter (no semantic fields, no events)
- `snapshot_for_reasoner() -> dict` — full (semantic fields + event history)

**Snapshot shape:**

```
snapshot()
  ├─ timestamp: ISO datetime string
  ├─ entities: list[dict]
  │    ├─ id: int                    # track_id
  │    ├─ bbox_xywh: [cx, cy, w, h]
  │    ├─ zones: ["desk", ...]
  │    ├─ pose: "sitting" | "standing" | "walking" | "unknown"
  │    ├─ seconds_in_frame: float    # how long this track has been visible
  │    └─ velocity_px_per_s: float
  ├─ people_count: int
  ├─ audio:
  │    ├─ level_db, dominant_class, speech_active
  │    ├─ top_classes: [{label, confidence}, ...]
  │    ├─ recent_spike: bool
  │    └─ spike_magnitude_db: float
  ├─ devices: {"light": {on, power_w}, "fan": {on, power_w}}
  ├─ baselines: {audio_mean_db, audio_std_db, typical_occupancy,
  │              power_idle_lamp_w, power_idle_fan_w,
  │              ambient_audio_classes, calibrated}
  ├─ scene_description: str          # written by Observer
  ├─ activity_summary: str           # written by Observer
  ├─ mood: "quiet"|"active"|"transitional"  # written by Observer
  └─ recent_events: list[dict]       # last 50 events (serialized)
```

**Internal state tracking:** WorldState maintains `_entity_memo` (keyed
by track_id) to track `first_seen` timestamps and previous center
positions across ticks. This lets it compute `velocity_px_per_s` and
`seconds_in_frame` without asking the EventDetector. Memos are pruned
when tracks leave.

**Thread safety:** Single `threading.Lock`. All public methods acquire
it. `snapshot*()` returns deep copies so callers never hold references
to live data. The lock is held briefly (no I/O inside it).

### CalibrationCollector (Day 2 Block 2)

- `CalibrationCollector(world, engine, detector, audio, plugs, duration)` — constructor.
- `run(overlay_callback) -> Baselines` — blocks for `duration` seconds, returns computed baselines.

Runs as a distinct startup phase between WorldState init and the main loop.
All Layer 0 systems are already running; the collector polls their published
state at regular intervals:

| Data source | Poll interval | Method used | Notes |
|---|---|---|---|
| Audio dB + classes | ~500 ms | `audio.latest_state()` | Speech periods excluded from dB if `CALIBRATION_EXCLUDE_SPEECH_FROM_FLOOR` |
| Occupancy count | ~1 s | `detector.active_track_ids()` | Median used, robust to tracker drops |
| Plug power | ~5 s | `plugs.state(alias)` | Handles None gracefully if plugs not yet discovered |

During calibration, WorldState is kept warm (update_from_yolo/audio/devices
called each tick) so entities, audio, and device state are already populated
when monitoring begins.

`ambient_audio_classes` are YAMNet classes that appear in >=
`CALIBRATION_AMBIENT_CLASS_MIN_RATIO` (30%) of classification windows,
excluding speech classes. These are stored in Baselines so Observer/Reasoner
prompts can deprioritize them as part of the room's normal sound profile.

### Observer (Day 2 Block 3)

- `Observer(world, camera)` — constructor. Creates a `genai.Client` for Gemini API calls.
- `Observer.call(trigger_events, frame) -> dict | None` — synchronous Gemini call.

**Input assembly:**
1. `world.snapshot_for_observer()` — entities, audio, devices, baselines (no semantic fields)
2. Current frame + 1 prior frame from `camera.buffer_snapshot(3)` (~2s ago)
3. Frames resized to ≤384px (both dims) → 258 tokens/image in Gemini's tokenizer
4. JPEG encoded at quality 70

**Gemini call config:**
- Model: `config.GEMINI_MODEL` (default: `gemini-2.5-flash`)
- `thinking_budget=0` — disables reasoning overhead for low-latency factual output
- `response_mime_type="application/json"` — structured JSON output
- `temperature=0.3`, `max_output_tokens=300`
- System prompt instructs factual-only output, YAMNet-as-hint rule

**Output contract:**

```
{
  "narration": str,           # short factual description, spoken as Beat 1
  "world_state_update": {
    "scene_description": str, # one-line scene summary
    "activity_summary": str,  # what person(s) are doing
    "mood": str               # "quiet" | "active" | "transitional"
  },
  "escalate": bool,           # should the Reasoner fire?
  "escalate_reason": str      # brief explanation
}
```

**Fallback on API failure:** deterministic narration from sensor data
(people count + event types). Escalates conservatively (only for
`REASONER_ALWAYS` event types). Exponential backoff on consecutive failures.

### ObserverWorker (Day 2 Block 3)

- `ObserverWorker(observer)` — constructor.
- `.start()` / `.stop()` — lifecycle.
- `.push_events(events, frame)` — non-blocking, called from main loop.
- `.poll_result() -> (dict, list[str]) | None` — non-blocking, returns (output, event_types).

**Debouncing:** 0.5s after wake, drain all pending events into one call.
**Periodic refresh:** 45s timer. When no events push for 45s, fires a
refresh call with empty event list (still sends camera frames).
**Threading:** single daemon thread. No queue needed — latest result only.

### Reasoner (Day 2/3 boundary)

- `Reasoner(world, camera)` — constructor. Creates an `anthropic.Anthropic` client.
- `Reasoner.call(observer_output, trigger_events) -> ReasonerOutput | None` — synchronous Claude call.

Triggered only when the hybrid routing policy escalates (see `agents/routing.py`): event type in `REASONER_ALWAYS` OR Observer returned `escalate=true`. Reads the previous `session_narrative` from WorldState and rewrites it with new understanding, compounding context over the session lifetime.

**Output contract (`ReasonerOutput` — Pydantic model):**

```
ReasonerOutput
  ├─ narration: str              # Beat 2 spoken text (≤40 words) or ""
  ├─ lamp: "on" | "off" | null  # device command (null = no change)
  ├─ fan:  "on" | "off" | null
  ├─ lamp_reason: str            # plain-English justification for lamp decision
  ├─ fan_reason: str             # plain-English justification for fan decision
  ├─ alert: bool                 # true → security/anomaly alert
  ├─ speak: bool                 # whether narration should play via TTS
  ├─ reasoning: str              # internal chain-of-thought (NOT spoken)
  ├─ activity_label: str         # "focused_work"|"idle"|"on_call"|"eating"|
  │                              #   "break"|"active"|"transitioning"|"unknown"
  ├─ session_narrative: str      # rewritten running session interpretation
  └─ world_state_update:
       ├─ scene_description: str
       └─ activity_summary: str
```

The `lamp`/`fan` fields are **null** when the Reasoner sees no reason to change the device state. When non-null, they carry the Reasoner's *judgment*; the DecisionEngine applies guardrails before any physical action.

### ReasonerWorker (Day 2/3 boundary)

- `ReasonerWorker(reasoner, decision_engine)` — constructor.
- `.start()` / `.stop()` — lifecycle.
- `.enqueue(observer_output, trigger_events, frame)` — non-blocking, called from main loop.
- `.poll_result() -> ReasonerOutput | None` — non-blocking, returns latest result.

Single daemon thread. Same pattern as ObserverWorker: 0.5s debounce, latest-result slot. Passes output to `DecisionEngine.execute()` automatically after each successful call.

### DecisionEngine (Day 2/3 boundary)

- `DecisionEngine(plugs, speaker, world)` — constructor.
- `.execute(output: ReasonerOutput) -> None` — run all actions in fixed order: (1) lamp, (2) fan, (3) alert, (4) Beat 2 TTS.

**The 5 guardrails (applied to every device command, in order):**

| # | Guard | Refusal reason string |
|---|---|---|
| 1 | **Idempotency** — skip if device already in requested state | `"idempotent"` |
| 2 | **Cooldown** — at most one agent toggle per device per `DEVICE_COOLDOWN_S` | `"cooldown (Xs left)"` |
| 3 | **Override lockout** — back off for `MANUAL_OVERRIDE_LOCKOUT_S` after a manual user toggle | `"override_lockout (Xs left)"` |
| 4 | **No-person ON guard** — never command ON when `WorldState.people_count() == 0` | `"no_person_for_on"` |
| 5 | **Plug-unreachable** — refuse if `PlugManager.is_available(alias)` is False | `"plug_unreachable"` |

Every accepted or refused decision is published to the broadcaster (`publish_narration("decision_engine", {...})`) so the dashboard can show what happened and why. The `agent_reason` field carries the Reasoner's original `lamp_reason`/`fan_reason` for debugging.

### EmptyRoomWatcher (Day 2/3 boundary)

- `EmptyRoomWatcher(on_empty, debounce_s, now_fn)` — constructor.
- `.update(person_count: int) -> None` — called every main-loop tick.

Pure-logic debounce: fires the `on_empty` callback exactly once after the room has been continuously empty for `debounce_s` seconds following a confirmed ≥1 → 0 person-count transition. Cold-start-empty rooms (room is empty when the agent starts, before any person has ever appeared) are silently ignored — the watcher waits for the first person before it can fire on their departure. If a person reappears before the debounce threshold, the watcher resets and waits for the next transition.

The `on_empty` callback in main.py synthesizes a `room_empty_confirmed` event and pushes it to the ObserverWorker, which escalates to the Reasoner via the normal routing path. This triggers the lamp-off logic without baking occupancy logic into the DecisionEngine itself.

### DashboardBroadcaster (Day 3 Phase 1)

Thread-safe bridge between the synchronous perception loop and the
asyncio event loop FastAPI lives on. Singleton — import `broadcaster`
from `server.broadcaster`.

**Producer API (callable from any thread):**
- `publish_event(dict)` — one Layer 0 event
- `publish_snapshot(dict)` — `WorldState.snapshot()` output
- `publish_narration(agent, dict)` — `"observer"` or `"reasoner"`
- `publish_routing(dict)` — `{trigger, fired, escalate, reason}`
- `publish_frame(jpeg_bytes)` — pre-encoded MJPEG frame

**Consumer API (inside async handlers):**
- `register() -> asyncio.Queue` — per-client queue, size 512
- `unregister(queue)` — remove on disconnect
- `wait_for_frame(last_seq, timeout) -> (bytes, seq)` — blocks on a
  `threading.Condition` until a newer frame arrives

**Bridge mechanism:** `bind_loop()` captures the running asyncio loop
at FastAPI startup. `_broadcast()` uses `asyncio.run_coroutine_threadsafe`
to put messages onto each client's queue from outside the loop.
Backpressure: queue full → drop oldest, never block the producer.

### FastAPI app (Day 3 Phase 1)

Endpoints:

| Method | Path            | Purpose                                          |
|--------|-----------------|--------------------------------------------------|
| GET    | `/`             | Health check (returns `"ok"`)                    |
| GET    | `/config`       | One-shot dashboard config (zones, camera, agents)|
| GET    | `/video/stream` | MJPEG (`multipart/x-mixed-replace`) of annotated frames |
| WS     | `/ws/state`     | Live fan-out of the four message kinds           |

**WebSocket message shapes** (JSON over `/ws/state`):

```
{"kind": "snapshot",  "data": <WorldState.snapshot()>}
{"kind": "event",     "data": {type, ts, elapsed, track_id, zones, payload}}
{"kind": "narration", "agent": "observer"|"reasoner",
                      "data": {narration, escalate, escalate_reason,
                               world_state_update, trigger_events, ts}}
{"kind": "routing",   "data": {trigger, fired, escalate, reason, ts}}
```

CORS is open (`*`) during development for the Next.js dev server on
port 3000. Locked down before anything ships.

**Same-origin proxy via Next.js rewrites:** the dashboard talks to the
backend through `/backend/:path*` (configured in
`dashboard/next.config.ts`). Required because browsers refuse to render
cross-origin `multipart/x-mixed-replace` inside an `<img>` tag even with
permissive CORS — the rewrite makes the MJPEG stream appear same-origin
to the browser while Next.js forwards the bytes to port 8000 server-side.
The WebSocket still connects directly (rewrites are HTTP-only, and WS
cross-origin works fine).

**MJPEG generator** uses `loop.run_in_executor(None, wait_for_frame, ...)`
to bridge the blocking `threading.Condition.wait` into the async loop
without stalling other requests. Each new frame yields one multipart
part; no polling, no duplicates.

### Dashboard frontend (Day 3 Phase 3)

Next.js 16 (App Router) + React 19 + Tailwind 4. Lives in `dashboard/`,
run with `npm run dev` on port 3000.

```
dashboard/
  app/
    layout.tsx          Root layout + metadata
    page.tsx            Grid: [VideoPanel | StatsPanel]  + 3 log teasers
    globals.css         Tailwind import + forced dark theme
  lib/
    api.ts              Backend URLs + TypeScript types mirroring the
                        WS message shapes in server/broadcaster.py
    useDashboardStream.ts
                        Client hook: opens /ws/state, splits messages
                        into state slices (world, events, observer-
                        Narrations, reasonerNarrations, routings),
                        auto-reconnects with 1s delay on disconnect,
                        also fetches /config on mount
  components/
    VideoPanel.tsx      <img src=/video/stream> + SVG zone overlay
                        (viewBox in camera-pixel coords so it scales
                        to fit whatever size the video element is) +
                        corner badges (LIVE, people count, audio dB)
    StatsPanel.tsx      PeopleCard · AudioCard · DevicesCard ·
                        BaselinesCard — live-updates via the world
                        snapshot stream
    LogShell.tsx        Shared chrome for the three scrolling log
                        panels: title bar, count badge, fixed-height
                        scroll viewport (min-h-0 + flex-1 trick).
    EventLog.tsx        Layer 0 events, newest first, color-coded by
                        type, formatted via lib/fmt.ts::fmtEvent.
    ObserverLog.tsx     Past Gemini narrations, escalate/triaged
                        badge, trigger-event chips.
    ReasonerLog.tsx     Interleaved stream: Beat-2 narrations (lamp/
                        fan/alert/speak chips + reasoning line) AND
                        routing skips (muted "skipped: <trigger> — …"
                        rows). Fired routings are suppressed because
                        they'd double-up with the narration entries.
```

Buffers: events keep the last 200; narrations/routings the last 100.
Entries are tagged on arrival with `_localId` (monotonic counter,
stable React key) and `_receivedAt` (browser wall-clock ms, renders
as HH:MM:SS). Python's monotonic `event.ts` isn't meaningful on the
client so we use arrival time uniformly across all three log streams.
The hook bails early on parse errors and is safe to unmount cleanly
(refs hold the WS handle, cleanup closes it and cancels reconnects).

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

## Runtime threads (currently — Block 3)

```
┌── MainThread ──────────── Phase 1: CalibrationCollector.run() (~30s)
│                            Phase 2: main.py loop: tick() detectors, merge
│                            events, push to ObserverWorker, poll results,
│                            render cv2 overlay, print to console
├── camera-capture ──────── cv2.VideoCapture.read() → _latest_frame, buffer
├── yolo-engine ─────────── pulls latest_frame, runs model.track, publishes
├── [PortAudio callback] ── sounddevice-managed: appends audio chunks, dB
├── audio-classify ──────── YAMNet every 500ms → smoothing → events
├── observer-worker ─────── debounce 0.5s → Observer.call() → Gemini API
│                            (~1s response). Also fires periodic refresh
│                            every 45s during quiet periods. Results posted
│                            back to main thread via poll_result().
├── kasa-loop ───────────── asyncio event loop (daemon); receives coroutines
│                            from main thread via run_coroutine_threadsafe
├── plug-discover ───────── one-shot thread: runs PlugManager.discover(),
│                            exits after both plugs found (or timeout)
├── dashboard-server ────── uvicorn asyncio loop (daemon) running FastAPI.
│                            Receives messages from the main loop via
│                            DashboardBroadcaster (run_coroutine_threadsafe).
│                            Serves /video/stream (MJPEG), /ws/state (WS),
│                            and /config to the dashboard frontend.
└── [kasa poll task] ─────── asyncio Task inside kasa-loop: calls device.update()
                              every 5 s, writes PlugState into _states dict
```

No IPC, no queues between the synchronous layers — just shared state behind locks. The kasa layer is the exception: it needs asyncio because python-kasa is async-only, so it lives in its own event loop with a thread-safe bridge (`run_coroutine_threadsafe`). The observer-worker thread communicates with MainThread via simple `threading.Event` signals and a shared result slot (no queue needed — only the latest result matters).

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

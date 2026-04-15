# LEARNING.md — Decisions, mistakes, and "aha" moments

Running log so you can reflect on what was hard, what you changed your mind about, and why. Most recent entries at the **top**. Each entry is dated and tagged.

Tags: `[decision]` `[mistake]` `[aha]` `[tradeoff]` `[gotcha]`

---

## 2026-04-15 — Day 2, Block 0 (video resolution audit)

### `[decision]` Dropped capture resolution from 1080p to 720p

**Initial config:** `CAMERA_CAPTURE_WIDTH/HEIGHT = 1920×1080`. The justification in the comment was "so YOLO's tracker gets crisp input."

**Why that's wrong:** YOLO immediately resizes every frame to 640×640 before inference — it doesn't matter if you hand it 1080p or 720p, it sees the same 640×640 either way. Gemini and Claude receive frames from the ring buffer, which was already downsampled to 1280×720. The dashboard stream target was also 720p. So 1080p capture was paying real USB bandwidth and memory cost with zero benefit at any consumer.

**The fix:** Set `CAMERA_CAPTURE_WIDTH/HEIGHT = 1280×720`. Capture and buffer are now the same resolution; the resize step in `_read_loop()` becomes a no-op. Updated `config.py` and `DESIGN.md`.

**Lesson:** Trace the data all the way through every consumer before choosing a capture resolution. "Higher is better" only holds if something downstream actually uses the extra pixels.

---

## 2026-04-14 — Day 1, Block 4 (zone map)

### `[aha]` Zones are a *floor* problem, not an *object* problem

**Initial instinct:** draw polygons around the desk, the door, the couch. Ask "is the person's bbox in the desk polygon?"

**Why that's wrong:** a laptop webcam sits well below ceiling height. A 2D pixel doesn't map to a 3D room location — it maps to a *ray* from the camera. A person standing between the camera and the couch has a bbox center that can easily overlap the "couch polygon" drawn on the couch itself, even though they're nowhere near the couch.

**The fix:** use the ground-plane assumption. A person's **feet** touch the floor, so the ray through their foot pixel intersects the floor at exactly one point. That makes foot-pixel → floor-position a 1:1 mapping with no calibration required. So zones are drawn on the *floor region around* the object (the walkable area in front of the desk), and queries use a foot-point.

**Lesson:** for a single camera, "where is X in the room" is unanswerable in general but becomes answerable when you constrain to a known plane. Pick your query geometry to exploit that.

---

### `[decision]` Ankle-keypoint midpoint, with bbox-bottom fallback

**First call:** just use the bbox bottom-center (`cy + h/2`). Simple, always available.

**Problem:** when a person sits behind a desk, the desk occludes their legs. The bbox shrinks to only their upper body, and "bbox bottom" lands on the desk edge, not the floor. The zone flickers between `['desk']` and `[]` depending on whether that pixel happens to fall inside the polygon.

**Better:** YOLO-pose already returns 17 keypoints per person, and indices 15/16 are left/right ankles with per-keypoint confidence. When both ankles are confident (>= 0.5), the midpoint is the truest "where the feet are." When they aren't (occluded, off-screen, blurry), fall back to bbox-bottom.

**Lesson:** use the richer signal when you have it, and degrade gracefully when you don't — don't pick the lowest common denominator just because it's simpler to code.

---

### `[tradeoff]` Rejected automatic zone detection (for now)

Considered using Gemini Flash ("here's a frame, return polygons for desk/door/couch") or Grounded-SAM (text-prompted segmentation) to skip the manual click step entirely.

**Gemini:** free and in-stack, but multimodal LLMs are genuinely bad at precise pixel coordinates — they hallucinate plausible-looking polygons that are off by 50-150px per edge. You'd verify visually anyway.

**Grounded-SAM:** pixel-precise, genuinely automatic, real state-of-the-art. Also a 2-4 hour integration, ~1-2 GB of model weights, and new deps for a 0.5h block.

**And the deeper issue:** "zone" isn't a visual concept — it's a *semantic room subdivision*. A model can find the desk object; it can't know where you want the "desk zone" boundary (just the desk surface? a 1m radius? the whole north wall?). Automatic detection still requires correction, which is the same step the click tool makes you do directly.

**Lesson:** automate when the cost of automation < cost of the manual loop, not reflexively. 2 minutes once per camera move ≥ 2 hours once per project.

---

### `[decision]` Consolidated smoke tests under `tests/`

The `_preview_main` functions in `perception/camera.py` and `perception/yolo_engine.py` are really smoke tests — you run them, watch, judge by eye. They lived inside the modules they exercised.

**Moved to:** `tests/smoke_camera.py`, `tests/smoke_yolo.py`, plus a new `tests/smoke_zone_map.py` that overlays zones + foot-points on the YOLO feed. The new files just `import` and call the existing `_preview_main` functions — no duplicate logic, and `python -m perception.camera` still works.

**Why the indirection:** gives a single obvious directory to look in when someone asks "how do I test this locally?", and leaves room for real pytest unit tests (`test_*.py`) alongside the interactive ones once there's non-hardware logic worth pinning (e.g. `world_state` diffing).

---

### `[gotcha]` SETUP.md is now load-bearing documentation, not just a session checklist

The zone approach encodes assumptions about the camera (≥1.8m high, tilted down, corner-mounted). If those aren't true, zones silently produce wrong answers — there's no error message. Writing camera-placement guidance into SETUP.md §4b, and the zone-tool walkthrough into §4c, makes those assumptions explicit and checkable. Treating SETUP.md as "just a reminder to activate the venv" would have buried real geometric preconditions in tribal knowledge.

---

## 2026-04-14 — Day 1, Block 3 (YOLO engine)

### `[decision]` Chose YOLO26 over YOLO11 — after initially picking YOLO11

**First call:** I recommended `yolo11n-pose` because YOLO26 was "too new" and the T4 GPU benchmark showed v26 slightly *slower* (1.5ms vs 1.7ms).

**Why that was wrong:** I was reading the GPU row on a comparison chart. This project runs on a MacBook — no NVIDIA GPU, so T4 TensorRT numbers are irrelevant. The row that matters is **CPU ONNX**, where YOLO26n is 38.9ms vs YOLO11n's 56.1ms — a ~30% speedup. Plus YOLO26-pose has the RLE keypoint head, which matters for detecting sitting/slouching under desk occlusion.

**Lesson:** always check which hardware row applies to *your* deployment before picking a model. "Faster on a datacenter GPU" ≠ "faster on my laptop."

---

### `[gotcha]` `ultralytics==8.4.37` is the latest on PyPI, but YOLO26 still works

Expected `pip install -U ultralytics` to pull a newer version that knew about YOLO26. Nope — 8.4.37 was already the latest. Worried the package was too old.

**Turns out:** Ultralytics ships model weights separately, via GitHub release assets. The package recognizes newer architectures as long as the class definitions are in the installed code. `YOLO("yolo26n-pose.pt")` auto-downloaded from `github.com/ultralytics/assets/releases/download/v8.4.0/` and loaded cleanly.

**Lesson:** for Ultralytics specifically, "model version" and "package version" are decoupled. Don't assume a version bump is needed.

---

### `[aha]` YOLO-pose only knows one class: `person`

Spent time explaining what YOLO can detect to Renzo. He correctly pushed back: "why does `other=0` in the preview, why isn't it seeing my cup?"

Ran a probe on a saved debug frame. YOLO26-pose's class map is literally `{0: 'person'}`. It's trained on the COCO pose dataset, which has one class. The 80-class detector is a *separate* model (`yolo26n.pt`).

**Implication for the architecture:** YOLO provides *continuity and tracking* (who is who, what pose, where in the frame), and Gemini provides *semantic richness* (that's a cup, that's a phone). The split isn't accidental — it's what makes event-driven, affordable perception possible. Gemini on every frame would cost ~$13/hr; YOLO is free continuous triage.

---

### `[tradeoff]` Pose classification deferred from Block 3 to Block 5

Considered including sitting/standing/walking classification inside `yolo_engine.py`. Chose to keep Block 3 scoped to raw YOLO output (bboxes, track IDs, raw keypoints) and push pose-class derivation into `event_detector.py`.

**Why:** mixing raw perception and interpretation in one module obscures the layering. Keypoints are machine data; "sitting" is a named event. event_detector is where all named events should be born, for consistency with `zone_transition`, `new_person`, etc.

---

### `[gotcha]` `avg_infer_ms: 33.5` is borderline, not broken

Initially wrote "if >33ms, bump `INFER_EVERY_N_FRAMES`." User ran preview, got 33.5ms, asked if we should change something.

**Correction:** the engine's `last_frame_id` skip means no queue builds up — frames are dropped gracefully. 33.5ms just means occasional dropped frames, which are invisible for room-scale events (pose changes last hundreds of ms). Real thresholds for action:

- Drift to 45–50ms average
- Tracker starts losing IDs
- Fan audibly ramps

Leave at 1 unless symptoms show.

**Lesson:** turn soft thresholds into symptom-triggered rules, not hard numbers.

---

### `[decision]` YoloEngine API = background thread + `latest_result()`, not sync

Considered a simpler `run_once(frame) → result` API. Rejected because:

1. Main loop would stall 30–40ms per frame on inference.
2. Day 2's Observer + Day 3's WebSocket will also want `latest_result()` without blocking.
3. Mirrors the pattern in `camera.py`, so there's one thread-safety idiom to remember.

Tradeoff: one more thread to debug, but the pattern is already proven by `CameraCapture`.

---

### `[aha]` Zones aren't a YOLO concept — they're pixel rectangles in config

Renzo asked "can YOLO even create zones?" Good prompt — I had been blurring "YOLO can do X" with "we can derive X from YOLO output." Zones are defined in `config.ZONES` as pixel tuples; `event_detector` checks bbox centers against them. YOLO just hands us bbox centers.

**Follow-on:** means the camera has to stay still for the whole session. If it moves, pixel zones point at the wrong parts of the room. Added to decision backlog: Block 4 will build an interactive click-to-define zones tool.

---

## Template for new entries

```
## YYYY-MM-DD — Day N, Block M (short title)

### `[tag]` Short headline of what happened

What I thought / what I did.

What actually happened / why the first attempt was wrong.

The lesson or the rule I now follow.
```


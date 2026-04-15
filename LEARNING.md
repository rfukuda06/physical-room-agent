# LEARNING.md — Decisions, mistakes, and "aha" moments

Running log so you can reflect on what was hard, what you changed your mind about, and why. Most recent entries at the **top**. Each entry is dated and tagged.

Tags: `[decision]` `[mistake]` `[aha]` `[tradeoff]` `[gotcha]`

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

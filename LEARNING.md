# LEARNING.md — Decisions, mistakes, and "aha" moments

Running log so you can reflect on what was hard, what you changed your mind about, and why. Most recent entries at the **top**. Each entry is dated and tagged.

Tags: `[decision]` `[mistake]` `[aha]` `[tradeoff]` `[gotcha]`

---

## 2026-04-15 — Hardware (iPhone as camera/audio source)

### `[decision]` Tried iPhone via Continuity Camera — reverted to MacBook

**What we tried:** Use the iPhone as both the camera and microphone source via macOS Continuity Camera. Motivation: easier to mount the iPhone high in the room for a better overhead angle, and frees up the MacBook for normal use.

**What actually happened:** The audio connection worked fine (iPhone mic shows up as a sounddevice input). The camera connected but was locked into a zoomed-in field of view — Continuity Camera's Center Stage auto-framing crops and zooms to keep people centered, making the frame too tight for whole-room monitoring. Turning Center Stage off didn't fully resolve the lens/zoom behavior.

**Decision:** Reverted `CAMERA_INDEX` back to `0` (MacBook built-in) and `AUDIO_DEVICE_INDEX` back to `1` (MacBook mic). iPhone camera unusable for this use case until the zoom/framing issue is resolved.

**Lesson:** Continuity Camera is designed for video calls (face framing), not room monitoring (wide fixed shot). If iPhone camera is needed in the future, investigate disabling all video effects (Center Stage, Studio Light, Portrait) or using a third-party app like Camo that exposes raw camera output without Apple's auto-framing.

---

## 2026-04-15 — Day 1, Block 6 (audio perception)

### `[aha]` YAMNet is a hint engine, not a ground-truth classifier

**What I expected:** YAMNet would reliably classify most of the 30
whitelisted room sounds — coughing, footsteps, door opening, glass
breaking, clapping, etc.  The Observer would use these classifications
as solid evidence.

**What actually happened:** YAMNet was trained on YouTube audio, not
real-time laptop-mic recordings at room distance.  In testing, it
reliably detected speech and a few loud/distinctive sounds (knocking
when close to mic), but **missed the majority of the 30 classes**.
Quiet sounds (cough, footsteps, breathing) never registered.  Brief
transients (claps) got averaged away in the 1-second window.  The dB
spike detector, which is pure loudness with no classification, was far
more reliable for detecting that *something* happened.

Also: audio spikes fire on mundane ambient noise (chair scraping, AC
cycling) — not just notable events.  If the Observer is told "audio
spike occurred" and forced to explain it, it will hallucinate.

**Lesson — three rules for Observer/Reasoner prompt design:**
1. Treat `unusual_sound_class` as a weak hint, not fact.  Confirm
   with the camera frame.
2. Never assume something didn't happen because YAMNet didn't report
   it.  Absence of classification ≠ absence of event.
3. If a spike fires and the camera shows nothing interesting, say "I
   heard something but can't tell what" — don't invent a narrative.

**Practical signal priority:** dB spike > speech transitions >
classification hints.  Audio's role is to nudge the Observer to look
at the frame, not to be an authoritative sensor.

---

### `[decision]` Speech as transitions, not raw state

**First plan:** track `speech_active` as a flag in AudioState and emit
`unusual_sound_class` for speech like any other class.

**Why that's wrong:** speech is the most common sound in a room.
Emitting an event every time someone talks would flood the Observer
with routine noise. But ignoring speech entirely loses useful context
(someone started talking to the room agent, or the room went quiet).

**What we did instead:** treat speech the same way EventDetector treats
pose — emit events on *transitions* only. `speech_start` fires when
silence becomes speech; `speech_end` fires when speech becomes silence.
Continuing speech is tracked silently via `AudioState.speech_active`.
Both transitions use the same `YAMNET_PERSISTENCE_WINDOWS` hysteresis
(2 windows = 1 s) to avoid flickering on brief pauses between
sentences.

**Lesson:** when a signal is *always on* during normal use, the useful
information is in its edges (on/off transitions), not its level.
Observer/Reasoner can use `speech_start` as context ("user is talking
to me") without drowning in "still talking" events every 500 ms.

---

## 2026-04-15 — Day 1, Block 5 (event detector)

### `[gotcha]` BoT-SORT track IDs are per-lifetime, not per-identity

**First instinct:** treat `track_id` as a stable handle for "this
person", so `lost_person` → `new_person` with the same id means "they
came back".

**Why that's wrong:** BoT-SORT does a short-window re-association
(appearance + motion) across occlusion, but it cannot re-identify
someone who walked fully out of frame and returned 10 seconds later.
The tracker just assigns a fresh id. So a human who steps out and
comes back gets two separate lifecycles: `lost_person(id=7)`, then
later `new_person(id=12)`, with nothing connecting them in the stream.

**What we did:** made this explicit in the event_detector docstring
and DESIGN.md — Layer 0 does NOT do identity re-association. If
anything wants "same person returned" semantics, it must reason from
context (time gap, zone, appearance) at a higher layer. The Observer
prompt will account for this later; for now we just don't pretend
track_id means more than it does.

**Lesson:** tracker IDs answer "is this the same detection as last
frame?", not "is this the same physical object as last week?". Don't
let naming convenience blur that.

---

### `[aha]` Hysteresis is the difference between a useful event and a firehose

**First draft of the detector:** "emit `pose_change` whenever the
classifier's output differs from last frame's." Ran the synthetic
test; got ~6 pose_change events in 200 synthetic "sitting" frames
because the classifier occasionally dipped to "unknown" (low keypoint
confidence) and bounced back.

**Fix:** three pieces of hysteresis, all keyed off *consecutive* frame
counts in per-track state:
- Pose transition needs `EVENT_POSE_HYSTERESIS_FRAMES` (5) of
  consistent new classification before flipping.
- Zone transition needs `EVENT_ZONE_DWELL_FRAMES` (5) of consistent
  zone membership — stops "driving by" a zone boundary from counting.
- `unknown` pose doesn't count toward a streak; it holds the previous
  confirmed state. Bad frames shouldn't vote.
- Lost-person has its own grace window (`LOST_PERSON_GRACE_FRAMES`,
  30 ≈ 1s) so brief tracker hiccups under occlusion don't turn into
  spurious lost/new pairs within a single physical presence.

**Lesson:** any state machine whose inputs are noisy should count
consecutive supporting observations, not just current-vs-last. Cost:
four small counters per track. Benefit: the event stream is actually
usable by the Observer without a flood filter on top.

---

## 2026-04-15 — Day 2, Block 0 (video resolution audit)

### `[mistake]` Dropped capture resolution from 1080p to 720p

**Initial config:** `CAMERA_CAPTURE_WIDTH/HEIGHT = 1920×1080`. The justification in the comment was "so YOLO's tracker gets crisp input."

**Why that's wrong:** YOLO immediately resizes every frame to 640×640 before inference — it doesn't matter if you hand it 1080p or 720p, it sees the same 640×640 either way. Gemini and Claude receive frames from the ring buffer, which was already downsampled to 1280×720. The dashboard stream target was also 720p. So 1080p capture was paying real USB bandwidth and memory cost with zero benefit at any consumer.

**The fix:** Set `CAMERA_CAPTURE_WIDTH/HEIGHT = 1280×720`. Capture and buffer are now the same resolution; the resize step in `_read_loop()` becomes a no-op.

**Lesson:** Trace the data all the way through every consumer before choosing a capture resolution. "Higher is better" only holds if something downstream actually uses the extra pixels.

---

## 2026-04-14 — Day 1, Block 4 (zone map)

### `[aha]` Zones are a *floor* problem, not an *object* problem

**Initial instinct:** draw polygons around the desk, the door, the couch. Ask "is the person's bbox in the desk polygon?"

**Why that's wrong:** a laptop webcam sits well below ceiling height. A 2D pixel doesn't map to a 3D room location — it maps to a *ray* from the camera. A person standing between the camera and the couch has a bbox center that can easily overlap the "couch polygon" drawn on the couch itself, even though they're nowhere near the couch.

**The fix:** use the ground-plane assumption. A person's **feet** touch the floor, so the ray through their foot pixel intersects the floor at exactly one point. That makes foot-pixel → floor-position a 1:1 mapping with no calibration required. So zones are drawn on the *floor region around* the object (the walkable area in front of the desk), and queries use a foot-point.

**Lesson:** for a single camera, "where is X in the room" is unanswerable in general but becomes answerable when you constrain to a known plane. Pick your query geometry to exploit that.

---

### `[gotcha]` Ankle-keypoint midpoint, with bbox-bottom fallback

**First call:** just use the bbox bottom-center (`cy + h/2`). Simple, always available.

**Problem:** when a person sits behind a desk, the desk occludes their legs. The bbox shrinks to only their upper body, and "bbox bottom" lands on the desk edge, not the floor. The zone flickers between `['desk']` and `[]` depending on whether that pixel happens to fall inside the polygon.

**Better:** YOLO-pose already returns 17 keypoints per person, and indices 15/16 are left/right ankles with per-keypoint confidence. When both ankles are confident (>= 0.5), the midpoint is the truest "where the feet are." When they aren't (occluded, off-screen, blurry), fall back to bbox-bottom.

**Lesson:** use the richer signal when you have it, and degrade gracefully when you don't — don't pick the lowest common denominator just because it's simpler to code.

---

## 2026-04-14 — Day 1, Block 3 (YOLO engine)

### `[mistake]` Chose YOLO26 over YOLO11 — after initially picking YOLO11

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

Spent time explaining what YOLO can detect. Renzo correctly pushed back: "why does `other=0` in the preview, why isn't it seeing my cup?"

Ran a probe on a saved debug frame. YOLO26-pose's class map is literally `{0: 'person'}`. It's trained on the COCO pose dataset, which has one class. The 80-class detector is a *separate* model (`yolo26n.pt`).

**Implication for the architecture:** YOLO provides *continuity and tracking* (who is who, what pose, where in the frame), and Gemini provides *semantic richness* (that's a cup, that's a phone). The split isn't accidental — it's what makes event-driven, affordable perception possible. Gemini on every frame would cost ~$13/hr; YOLO is free continuous triage.

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

## Template for new entries

```
## YYYY-MM-DD — Day N, Block M (short title)

### `[tag]` Short headline of what happened

What I thought / what I did.

What actually happened / why the first attempt was wrong.

The lesson or the rule I now follow.
```

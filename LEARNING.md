# LEARNING.md — Decisions, mistakes, and "aha" moments

Running log so you can reflect on what was hard, what you changed your mind about, and why. Most recent entries at the **top**. Each entry is dated and tagged.

Tags: `[decision]` `[mistake]` `[aha]` `[tradeoff]` `[gotcha]`

---

## 2026-04-17 — Day 3, Dashboard Phase 3 (Next.js scaffold)

### `[gotcha]` Cross-origin MJPEG in `<img>` renders as a black box

**What I assumed:** a plain `<img src="http://127.0.0.1:8000/video/stream">` on the Next.js page (`localhost:3000`) would just work, the same way it works when you paste the backend URL directly into the address bar. FastAPI already had `CORSMiddleware(allow_origins=["*"])` for the JSON/WS endpoints, so I expected the MJPEG to be covered too.

**What actually happened:** the page loaded, the WebSocket connected, `/config` fetched fine, and the SVG zone overlay rendered on top of the video panel — but the video area itself stayed solid black. No JS error, no `onError`, no fallback text. Direct access to `http://127.0.0.1:8000/video/stream` in a new tab worked perfectly, so the stream itself was fine.

Browsers treat `multipart/x-mixed-replace` inside a cross-origin `<img>` differently from direct navigation. Even with `Access-Control-Allow-Origin: *` on the response, the image loader refuses to render the stream (or silently drops parts). This isn't a CORS error in the traditional "blocked by CORS policy" sense — the request succeeds, the response is just ignored by the img element.

**Fix:** made the browser see same-origin requests via Next.js rewrites.

```ts
// dashboard/next.config.ts
async rewrites() {
  return [{ source: "/backend/:path*", destination: "http://127.0.0.1:8000/:path*" }];
}
```

Then the frontend uses a relative URL `/backend/video/stream`. Next.js dev server proxies the bytes to port 8000 server-side; the browser thinks it's all same-origin. WebSocket stays on the absolute URL because Next rewrites are HTTP-only, and cross-origin WS doesn't trigger the same image-loader quirk.

**Lesson:** CORS and CORB (Cross-Origin Read Blocking) are different layers. CORS headers let the browser *read* a cross-origin JSON or image. CORB-style policies decide whether certain response types are delivered at all to non-matching contexts. For anything exotic (MJPEG, SSE, streamed responses), the safe default for a browser app is a same-origin proxy — fewer ways for it to silently fail, and it's the same config you'll want in production anyway.

---

## 2026-04-16 — Day 2, Block 1 (model & SDK choices)

### `[decision]` Gemini 2.0 Flash → 2.5 Flash, and migrating to google-genai SDK

**What the plan said:** Use Gemini 2.0 Flash via the `google-generativeai` SDK (v0.8.6).

**What we discovered:** Gemini 2.0 Flash is being **shut down June 1, 2026** — less than 2 months away. The `google-generativeai` SDK is deprecated in favor of `google-genai`. Meanwhile, Gemini 3 Flash exists but is in preview, 67% more expensive, and overkill for the Observer's factual-description role.

**Decision:** Use **Gemini 2.5 Flash** (`gemini-2.5-flash`) — stable, cheap, fast enough for ~1s Beat 1 responses. Migrate to `google-genai` SDK when implementing Block 3. Model name goes in `config.py` for easy swapping.

**Lesson:** Always check model deprecation timelines before building against a specific version. "Latest" in an architecture doc can go stale in weeks.

---

### `[decision]` Sonnet 4.6 over Opus for the Reasoner — speed wins

**What we considered:** Whether Opus 4.6's better reasoning justified its slower response time for Beat 2.

**Why Sonnet wins:** The two-beat rhythm demands Beat 2 arrives ~2-4 seconds after Beat 1. Opus takes 5-15 seconds with images — by the time it responds, the event feels stale. Sonnet at ~2-4s hits the sweet spot. The real intelligence lever is prompt quality and context richness, not model tier.

**Lesson:** In real-time systems, latency constraints pick the model for you. If the UX requires a response within N seconds, that's your filter — then optimize prompts within that budget.

---

### `[decision]` Calibration reduced from 5 min to 30 seconds

**What the plan said:** 5–10 minute calibration phase at startup.

**What we realized:** 30 seconds gives ~30 audio dB samples (1/sec), 6 power readings (5 sec polling), and ~60 YAMNet classification windows (500ms each). That's more than enough for mean/std/median on all baselines. The 5-min number was a conservative placeholder that would make dev iteration painful.

**Lesson:** Question placeholder numbers from planning docs. Calculate the actual sample counts you need, then derive the duration — don't just pick a round number that "feels safe."

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

## 2026-04-16 — Day 2, Block 3 (Observer / Gemini integration)

### `[gotcha]` Gemini 2.5 Flash has thinking ON by default — kills latency

**What I assumed:** Gemini 2.5 Flash would respond in ~0.5-1s for a simple factual description task, matching its "fast" branding.

**What we discovered:** Gemini 2.5 Flash has *thinking* (internal chain-of-thought reasoning) enabled by default. This adds significant latency and cost — the model reasons internally before responding, even for simple tasks like "describe what you see in this image." The thinking tokens also count toward output cost at $2.50/M tokens.

**Fix:** Set `thinking_config=types.ThinkingConfig(thinking_budget=0)` in the `GenerateContentConfig`. This disables the internal reasoning entirely, which is correct for the Observer's role — it does factual description, not reasoning (that's the Reasoner's job).

**Also discovered:** Gemini 2.5 Flash-Lite ($0.10/M input, $0.40/M output) has thinking OFF by default and is 3-6x cheaper. It's a viable fallback if Flash quality isn't needed. Made it swappable via `config.GEMINI_MODEL`.

**Lesson:** Always check whether a model has "thinking" or "reasoning" mode enabled by default. For low-latency production use, explicitly disable it unless you need it. The model name ("Flash") doesn't guarantee fast — the config does.

### `[gotcha]` Image token cost depends on resolution — resize before sending

**What I assumed:** each image sent to Gemini costs roughly the same number of tokens.

**What's actually true:** Gemini tokenizes images based on dimensions. Both dims ≤ 384px = 258 tokens flat. Larger images get tiled at 768×768, each tile = 258 tokens. Our raw 1280×720 frames would cost 2 tiles = 516 tokens each. By resizing to ≤384px (preserving aspect ratio), we halve the image token cost — 258 tokens instead of 516 per frame.

**Lesson:** for multimodal LLM calls, always check the image tokenization rules. A cheap resize before the API call can cut your image costs 50%+ with negligible quality loss for scene understanding tasks.

---

## 2026-05-18 — Day 2/3 boundary, Lamp & fan control

### `[architecture]` Two-tier decision split: Claude does judgment, code does guardrails

**What I thought:** the Reasoner prompt alone was enough. If you tell Claude "only turn on the lamp when someone is present" and "don't toggle twice in five seconds", the model will follow the rule. The agent's behavior comes from the quality of the instructions, so adding code checks would just be redundancy.

**What actually happened:** prompts drift. A context-heavy call shifts the model's attention; a borderline event produces "lamp: on" when the room is empty because the reasoning section convinced itself a person might still be there. And there is no way to write a prompt clause that enforces "the user touched it 8 seconds ago — back off for 30 more seconds" reliably. The rule requires exact wall-clock arithmetic, not natural language judgment. Beyond correctness, the two-beat rhythm means Beat 2 narration arrives *while the lamp is already toggling* — any retry logic in the prompt is too slow to matter.

The fix was to make the split explicit: Claude owns judgment ("should the lamp change state and why?"), the DecisionEngine owns enforcement (five named guardrails in code: idempotency, cooldown, override lockout, no-person ON guard, plug-unreachable). The DecisionEngine publishes every refusal — reason string and all — to the dashboard broadcaster, so when nothing happens you can see exactly which guard fired and why. The Reasoner prompt doesn't mention the guardrails at all; Claude just decides, and the engine filters.

The result is an agent that *feels* respectful rather than bossy. The lamp doesn't flicker. A manual override isn't ignored thirty seconds later. Turning on a light for an empty room never happens.

**Lesson:** when an LLM is your decider, code-level guardrails are not a fallback for a weak prompt — they are the *contract* between the model and the physical world. Keep the guards small (five in this project), keep them named, and surface every refusal to the dashboard so you can debug *why* nothing happened without adding logging after the fact.

---

## Template for new entries

```
## YYYY-MM-DD — Day N, Block M (short title)

### `[tag]` Short headline of what happened

What I thought / what I did.

What actually happened / why the first attempt was wrong.

The lesson or the rule I now follow.
```

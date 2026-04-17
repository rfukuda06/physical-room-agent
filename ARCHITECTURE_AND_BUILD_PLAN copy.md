# Physical AI Room Agent — Architecture & Build Plan

**Project codename:** Newton-for-a-Room
**Target:** Archetype AI interview MVP
**Build window:** 2–3 days
**Goal:** A live Physical Agent that observes a room through camera, microphone, and smart plug power sensors; reasons about what's happening via multi-modal fusion; and autonomously acts through voice and connected devices (lamp, fan).

---

## 0. Status — this is a living proposal, not a spec

Everything below is a **working plan**, not a commitment. We are in the planning phase, and most of the concrete choices in this document (specific event types, thresholds, the routing policy, cost estimates, the day-by-day timeline, the dashboard layout, even which LLMs handle which layer) are **placeholders to think against, not decisions to defend**.

The detail exists to make the shape of the problem visible — so that when we sit down to build, we're not reasoning from scratch. But expect large parts of this doc to be rewritten as we:

- discover what YOLO actually produces on your webcam,
- learn how Gemini and Claude actually behave on these prompts,
- find out which events matter in practice vs. which ones sound important on paper,
- run into limits (latency, noise, hardware quirks) that reshape the architecture.

**Treat every specific value in this doc — event names, thresholds, routing rules, cost numbers, timelines — as a first draft.** If something feels wrong during implementation, change it. The architecture (three tiers, two-beat rhythm, event-driven, multi-modal fusion) is the load-bearing idea. Everything else is scaffolding.

When we change something material, we'll update this doc so it stays accurate. If it stops being accurate, it stops being useful.

---

## 1. Vision

A continuously running Physical Agent that perceives your physical space in real time and demonstrates the complete agentic loop: **Sense → Fuse → Reason → Act**. The agent narrates observations aloud, detects anomalies against learned baselines, investigates unusual events across multiple sensor modalities, and takes physical actions (controlling lighting and airflow) based on its reasoning.

This directly mirrors Archetype AI's stated architecture: Newton (continuous multi-modal perception) + Physical Agents (on-demand reasoning and action). The MVP demonstrates understanding of the platform thesis, not just one vertical.

### 1.1 Product purpose (the one-sentence thesis)

This project is not a smart-home gadget and not a bag of features. It is the **smallest possible working instance of physical AI** — a single agent that senses a physical space with multiple modalities, reasons about what's happening in human terms, and acts back on the space. The room is a scaled-down factory; the lamp is a scaled-down actuator. The same "sense → fuse → reason → act" loop that runs Archetype's industrial systems, pointed at a room.

**One-sentence pitch:**
> Newton-for-a-Room is a physical AI agent that continuously watches a space and helps it serve the people in it — it senses presence, activity, sound, and power; reasons about what's happening in plain English; and adjusts the environment accordingly. It's the room-scale version of the industrial systems Archetype builds.

**Shorter:** *A room that pays attention and responds.*

### 1.2 What the product narrates (the feature set, organized around the thesis)

Every capability is an instance of the same loop exercised on a different signal. These are not separate features — they are the same primitive proving it generalizes.

1. **Presence awareness** — who's in the room, when they arrived, when they left, how long the room has been empty. *(Driven by YOLO person tracking + zone transitions.)*
2. **Activity awareness** — what the person is doing in human terms: working, taking a break, on a call, heading out. *(Driven by Gemini's VLM interpretation of frames, with pose + zone as structured context. Object identification — laptop, phone, cup — happens here inside Gemini, not at Layer 0.)*
3. **Environmental awareness** — sound events and power-draw anomalies against the learned baseline. Phone ringing, door opening, unexplained noise while the room is empty, fan drawing more power than usual. *(Driven by YAMNet + Kasa plug telemetry.)*
4. **Action narration** — the agent explains *why* it's doing something, not just what. "Turning the lamp off because you've been out of the room for a minute." Narration is the **trust interface**, not decoration — without it the system looks like a motion-sensor light.
5. **Memory / summary** — the agent as a stateful system-of-record. "While you were gone, one person came by around 2:15 and left 3 minutes later. Nothing unusual." This is what distinguishes a reasoning agent from reactive rules.

### 1.3 The filter (what stays in, what gets cut)

Every future feature or scope decision is tested against one question:

> **Does this make the room better at perceiving, reasoning about, and responding to what's happening in it?**

If yes → build it. If it's clever but doesn't feed the loop → drop it.

---

## 2. System Architecture

### Three-tier sensing and reasoning (Architecture B — event-driven, two-beat rhythm)

The core design principle: **YOLO handles fast, local, structured perception. Gemini Flash handles quick factual description of events. Claude Sonnet handles deep reasoning, judgment, and action decisions.** This creates a natural two-beat rhythm when events warrant it: something happens → fast factual acknowledgment (Gemini, ~1 second) → thoughtful interpretation and response (Claude, ~2-3 seconds later). Like a security guard who first says "someone's at the door" then a beat later says "looks like a delivery person, nothing to worry about."

Importantly, **the Reasoner does NOT fire on every Observer call.** Routine, low-stakes events get Beat 1 only. The Reasoner is invoked via a **hybrid routing policy** (Section 2.5) that guarantees critical event types always reach Claude, while letting the Observer triage everything else. This keeps cost predictable and the demo crisp — interviewers don't sit through deep reasoning on trivial pose twitches.

```
┌──────────────────────────────────────────────────────────────────┐
│                 LAYER 0: LOCAL PERCEPTION (YOLO + Sensors)        │
│            (continuous, local, runs every frame/sample)           │
│                                                                   │
│  YOLO (Ultralytics) — owns fast structured perception:           │
│    • Object detection: identify people, furniture, objects        │
│    • Tracking: persistent IDs on people/objects across frames     │
│    • Pose estimation: body keypoints (standing, sitting, bent,    │
│      arms raised, fell)                                          │
│    • Entry/exit events: new person appeared, person left frame    │
│    • Zone transitions: person moved from desk → door              │
│    • Object state changes: object appeared, disappeared, moved    │
│    • Counts: N people present, chair occupied, laptop visible     │
│                                                                   │
│  Frame Buffer — rolling 30s of frames stored in memory           │
│                                                                   │
│  Audio (sounddevice + YAMNet) — continuous dB monitoring,        │
│    spike detection, AND sound classification:                     │
│    • Rolling 1s mic windows classified every ~500ms via YAMNet    │
│    • 521 AudioSet classes → filtered to a whitelist of ~30–50     │
│      room-relevant labels (speech, clap, door, glass_break,       │
│      footsteps, cough, laugh, music, alarm, knock, typing, etc.)  │
│    • Temporal smoothing: class must persist ≥2 windows to report │
│    • Produces top-k labels with confidences, not just dB          │
│  Smart Plugs (python-kasa) — power draw + on/off state           │
│                                                                   │
│  YOLO does NOT do:                                                │
│    • Rich semantic interpretation ("getting ready to leave")      │
│    • Natural language narration                                   │
│    • Deciding if something matters to a human                     │
│    • Resolving ambiguity                                          │
│                                                                   │
│  Output: structured machine state + event triggers to Layer 1    │
└──────────────────────────────────────────────────────────────────┘
                              │
                  ┌───────────┴───────────┐
                  │   EVENT TRIGGERS:      │
                  │  • new/lost person     │
                  │  • pose change         │
                  │  • zone transition     │
                  │  • object moved        │
                  │  • audio spike         │
                  │  • sound class detected│
                  │  • speech start/end    │
                  │  • power state change  │
                  │  • low YOLO confidence │
                  │  • 30–60s idle refresh │
                  └───────────┬───────────┘
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│          LAYER 1: OBSERVER AGENT (Gemini 2.0 Flash)              │
│     (event-driven + periodic refresh, fast factual description)  │
│                                                                   │
│  Receives from Layer 0:                                          │
│    • Current frame + 2–3 buffered prior frames                   │
│    • Structured YOLO signals (detections, poses, tracks, zones)  │
│    • Audio context (level, recent spikes, YAMNet top-k classes   │
│      with confidences — e.g. "clapping: 0.82, speech: 0.06")     │
│    • Device state (lamp power, fan power)                        │
│                                                                   │
│  Gemini Flash's responsibilities (FAST, ~0.5–1s response):      │
│    • Quick factual description of what just happened             │
│      "person_7 moved desk→door" → "Person stood up and walked    │
│       toward the door"                                           │
│    • Narrate audio-only events that have no visual signal        │
│      (phone ringing off-camera, cough, music, door sound when    │
│       the door is out of frame)                                  │
│    • Resolve visual ambiguity YOLO cannot handle, and use        │
│      YAMNet class labels to ground interpretation of sound       │
│    • Generate structured JSON update for the world state         │
│    • Produce a quick spoken narration (Beat 1 of two-beat)       │
│    • Pass event context + its description to Layer 2             │
│                                                                   │
│  Gemini Flash does NOT do:                                       │
│    • Judge whether something is dangerous or unusual             │
│    • Reason about intent, mood, or security implications         │
│    • Decide on physical actions (lamp, fan, alerts)              │
│    • Generate deep explanations or hypotheses                    │
│                                                                   │
│  When Gemini Flash gets called:                                  │
│    • Immediately on YOLO event triggers                          │
│    • Background refresh every 30–60s during quiet periods        │
│                                                                   │
│  Output: quick narration → TTS (Beat 1), world state update,    │
│          event context passed to Layer 2                         │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼ (conditional — hybrid routing, Section 2.5)
┌──────────────────────────────────────────────────────────────────┐
│            LAYER 2: REASONER AGENT (Claude Sonnet)               │
│    (event-driven, deep reasoning + judgment + action decisions)   │
│                                                                   │
│  Triggered when the routing policy says deep reasoning is needed: │
│    • Event type is in REASONER_ALWAYS (must-escalate list), OR   │
│    • Observer's output sets escalate=true                         │
│  Runs in parallel while Beat 1 narration is already playing.     │
│                                                                   │
│  Receives:                                                       │
│    • Observer's factual description of the event                 │
│    • Current frame + buffered prior frames                       │
│    • Full world state (machine + semantic + baselines)           │
│    • Recent event history (last 50 events)                       │
│    • All sensor channels (YOLO tracks, audio level + YAMNet      │
│      class history, power)                                       │
│    • Current time of day, duration of occupancy, session context │
│                                                                   │
│  Claude Sonnet's responsibilities (DEEP, ~2–3s response):       │
│    • Reason about what the event MEANS                           │
│      "Person is heading to the door after 2 hours at the desk    │
│       — likely taking a break"                                   │
│    • Judge whether something is routine, notable, or dangerous   │
│      "Movement at 3:12 AM, outside normal hours — suspicious"    │
│    • Assess physical/emotional state from accumulated evidence   │
│      "Posture has been deteriorating for 30 minutes — may be     │
│       fatigued. Coughing detected 4 times in last 10 min."       │
│    • Fuse YAMNet class history with YOLO + time + power:         │
│      "glass_breaking + no visible person + 3 AM → alert"         │
│      "door + new person + audio spike → confirmed entry"         │
│      "music for 2h + steady posture → focused, don't interrupt"  │
│    • Decide on physical actions:                                 │
│      - Speak a thoughtful follow-up (Beat 2 narration)           │
│      - Control lamp (on/off based on occupancy, time, context)   │
│      - Control fan (based on extended presence, activity level)   │
│      - Escalate alert for security situations                    │
│    • Write structured event to timeline with full reasoning      │
│    • Update semantic world state with deeper interpretation      │
│                                                                   │
│  Output: spoken reasoning → TTS (Beat 2), actuator commands,     │
│          event log entries, updated semantic state                │
└──────────────────────────────────────────────────────────────────┘
```

### 2.5 Reasoner routing policy (hybrid — provisional)

**This section is a first cut.** The specific event types, the `escalate` flag shape, and the routing split will almost certainly change once we see how the Observer and Reasoner actually behave on real inputs. The *principle* — that the Reasoner shouldn't fire on every Observer call, and that gating should be a mix of static rules + Observer-driven escalation — is what we're committing to. The specifics below are a starting point to build against.

The Observer fires frequently (every structural event + periodic refresh), but the Reasoner is expensive and only useful when real judgment is required. The current plan is a hybrid:

**1. Must-escalate event types (REASONER_ALWAYS) — starting list, expect to adjust:**
- `new_person`, `lost_person` — occupancy changes drive most actions
- `unusual_sound_class` — glass_break, alarm, scream, and similar high-salience YAMNet classes
- `power_anomaly` — unexpected appliance state changes
- `security_event` — any composite event flagged by the event detector as security-relevant
- `periodic_refresh_hourly` — one guaranteed full-reasoning pass per hour for baseline sanity

**2. Observer-triaged events** — Gemini decides. Each Observer call returns:
```json
{
  "narration": "...",
  "world_state_update": {...},
  "escalate": true | false,
  "escalate_reason": "short string explaining why"
}
```
If `escalate == true`, the Reasoner runs. Otherwise, Beat 1 plays and the pipeline stops.

**3. Routine events** — stay in Observer-only. Pose twitches, minor zone nudges, low-magnitude audio fluctuations produce a Beat 1 and no more. No Claude call, no Beat 2.

**Effect on the two-beat rhythm.** Not every event produces a Beat 2. A quiet office with one person typing might produce 20 Beat 1s in an hour and only 3 Beat 2s (new person arrival, leaving for lunch, return from lunch). That's correct — the agent is being discerning, not chatty.

**Effect on cost.** With routing, the Reasoner fires roughly 30-50% as often as the Observer, cutting Claude cost by ~50-70%. See Section 9.

```python
# Pseudocode — agents/routing.py
REASONER_ALWAYS = {
    "new_person", "lost_person",
    "unusual_sound_class",
    "power_anomaly",
    "security_event",
    "periodic_refresh_hourly",
}

def should_call_reasoner(event, observer_output) -> bool:
    if event.type in REASONER_ALWAYS:
        return True
    if observer_output.get("escalate") is True:
        return True
    return False
```

### The two-beat rhythm in practice

```
YOLO: person_3 pose sitting→standing, zone desk→door, audio -12dB

  Beat 1 (Gemini Flash, ~1 second):
    TTS: "Person stood up and is walking toward the door."
    [plays immediately — system feels responsive]

  Beat 2 (Claude Sonnet, ~3 seconds later):
    TTS: "You've been working for about 90 minutes — good time
          for a break. Turning off the lamp and keeping the fan
          running to cool the room."
    [lamp clicks off]
    [plays after Beat 1 — system feels thoughtful]
```

Another example:
```
YOLO: new person_8 detected, audio spike +20dB, time = 3:12 AM

  Beat 1 (Gemini Flash, ~1 second):
    TTS: "New person detected entering from the right side."

  Beat 2 (Claude Sonnet, ~3 seconds later):
    TTS: "This is unusual — it's 3:12 AM, well outside normal
          occupancy hours. The person is not following a typical
          entry pattern. Turning on all lights and logging a
          security alert."
    [lamp clicks on]
    [event logged as security alert]
```

### Example flow through all three layers

```
Layer 0 (YOLO):  person_3 pose changed: sitting → standing
                 person_3 zone transition: desk → door
                 audio: level dropped 12dB
                      ↓ triggers
Layer 1 (Gemini): [receives current frame + 2 prior frames + YOLO signals]
                  "The person stood up from the desk and is walking toward
                   the door. Audio dropped, suggesting they stopped talking.
                   Likely leaving the room."
                  → Narrates via TTS
                  → Updates world state: occupancy = departing
                      ↓ 2 seconds later, YOLO: person_3 lost from frame
Layer 1 (Gemini): [receives frame showing empty room + YOLO lost-person event]
                  "Room is now empty. Recommending: turn off lamp."
                  → Action: lamp OFF
                  → Updates world state: occupancy = 0, lamp = off
```

### Shared state: the World Model

A lightweight in-memory structured object. Updated by YOLO continuously (structured machine state) and by the Observer on each call (semantic state). Queried by the Investigator when anomalies trigger.

```
WorldState:
  timestamp: ISO datetime

  # Structured machine state (updated by YOLO every frame)
  entities:
    - id: "person_3"
      type: person
      bbox: [x, y, w, h]
      zone: "desk" | "door" | "couch" | "unknown"
      pose: "sitting" | "standing" | "walking" | "bent_over" | "unknown"
      pose_keypoints: {...}  # raw keypoint data
      velocity_px_per_s: float
      frames_tracked: int
      last_seen: timestamp
    - id: "object_12"
      type: "laptop"
      bbox: [x, y, w, h]
      stationary_since: timestamp

  counts:
    people: int
    objects_of_interest: int

  # Sensor state (updated continuously by Layer 0)
  audio:
    level_db: float
    recent_spike: bool
    spike_magnitude_db: float
    # YAMNet classification (updated every ~500ms)
    top_classes: list[(label: str, confidence: float)]   # top-k above threshold
    dominant_class: str                                   # top label, or "unknown"
    class_history: ring buffer of last N classifications  # for smoothing + Reasoner context
    speech_active: bool                                   # derived from class history
  devices:
    lamp: {on: bool, power_w: float}
    fan:  {on: bool, power_w: float}

  # Semantic state (updated by Gemini on each call)
  scene_description: str
  activity_summary: str
  mood: "quiet" | "active" | "transitional"

  # Baselines (learned during calibration)
  baselines:
    audio_mean_db: float
    audio_std_db: float
    typical_occupancy: int
    light_baseline: float
    power_idle_lamp_w: float
    power_idle_fan_w: float

  # Event history
  recent_events: ring buffer of last 50 events
```

---

## 3. Tech Stack

### Layer 0 — Local Perception
- **OpenCV** — webcam frame capture only (the pipe)
- **Ultralytics YOLO26n-pose** — detection, tracking (BoT-SORT), pose estimation (17 COCO keypoints)
- **sounddevice** — microphone streaming, real-time dB level monitoring, spike detection
- **TensorFlow + tensorflow-hub** — YAMNet (MobileNetV1-based audio tagger, 521 AudioSet classes, ~3.7M params, ~20ms CPU inference per 0.96s window, free and local)
- **python-kasa** — TP-Link Kasa KP125M smart plug control + energy monitoring
- **NumPy** — baseline statistics, anomaly scoring, frame buffer management

### Layer 1 — Observer Agent (Beat 1: fast factual)
- **google-genai** (migrating from deprecated `google-generativeai`) — **Gemini 2.5 Flash** (`gemini-2.5-flash`). Changed from Gemini 2.0 Flash which is being shut down June 1, 2026. Stable, cheap, fast enough for ~1s response.

### Layer 2 — Reasoner Agent (Beat 2: deep reasoning + actions)
- **anthropic** — **Claude Sonnet 4.6** (`claude-sonnet-4-6-20250514`). ~2-4s response time, strong reasoning. Opus too slow (~10-15s) for the two-beat rhythm.

### Actuators
- **edge-tts** or **pyttsx3** — text-to-speech for spoken narration
- **python-kasa** — lamp and fan control via smart plugs

### Server + Dashboard
- **FastAPI** — web server + WebSocket hub for real-time data push
- **Next.js + React + Tailwind + Recharts** — live dashboard
  - Fallback: **Streamlit** if Day 3 is tight

### Hardware
- Laptop with webcam, microphone, speakers
- 2× TP-Link Kasa KP125M Smart Plug (with energy monitoring)
- Lamp + small fan (any standard plug-in appliance)

---

## 4. Module Breakdown

```
project_root/
├── perception/
│   ├── camera.py            # OpenCV webcam capture + frame buffer (rolling 30s)
│   ├── yolo_engine.py       # YOLO detection, tracking, pose estimation
│   ├── zone_map.py          # Define room zones (desk, door, couch, etc.)
│   ├── event_detector.py    # Consumes YOLO output, emits structured events
│   │                        #   (new_person, lost_person, pose_change,
│   │                        #    zone_transition, object_moved,
│   │                        #    sound_class_detected, speech_started,
│   │                        #    speech_ended, unusual_sound)
│   ├── audio.py             # sounddevice mic streaming + dB + spike detection
│   │                        #   + YAMNet classification on rolling 1s window
│   │                        #   + whitelist filter + temporal smoothing
│   │                        #   + AudioClassifier interface (swap YAMNet→CLAP/BEATs later)
│   └── plugs.py             # python-kasa plug discovery, power reading, control
│
├── agents/
│   ├── observer.py          # Layer 1: Gemini Flash integration (Beat 1)
│   │                        #   - receives events + frames + YOLO state
│   │                        #   - quick factual description
│   │                        #   - world state JSON update
│   │                        #   - passes context to Reasoner
│   ├── reasoner.py          # Layer 2: Claude Sonnet integration (Beat 2)
│   │                        #   - deep reasoning about meaning + intent
│   │                        #   - security / safety judgment
│   │                        #   - action decisions (lamp, fan, alerts)
│   │                        #   - thoughtful narration
│   ├── world_state.py       # Shared world model (machine + semantic state)
│   ├── baselines.py         # Calibration phase: learn normal room profile
│   └── decisions.py         # Action decision logic (speak, actuate, log)
│
├── actuators/
│   ├── speaker.py           # TTS output (edge-tts or pyttsx3)
│   └── smart_plug.py        # Lamp and fan control wrappers
│
├── server/
│   ├── main.py              # FastAPI app + WebSocket hub
│   └── events.py            # Event bus connecting all layers
│
├── dashboard/
│   └── (Next.js app or Streamlit)
│       ├── webcam feed with YOLO overlay (bounding boxes, pose, IDs)
│       ├── sensor strip charts (audio dB, lamp watts, fan watts, motion)
│       ├── world state panel (current structured + semantic state)
│       ├── agent reasoning feed (Observer = gray, Investigator = cyan, Action = green)
│       ├── event timeline (clickable markers for last hour)
│       └── system status (mode, API calls, running cost)
│
├── config.py                # API keys, thresholds, device IPs, zone definitions
└── main.py                  # Orchestrator: starts all layers, manages lifecycle
```

---

## 5. Data Flow

### Startup & calibration (~30 seconds)

*(Reduced from original 5–10 min estimate. 30 s provides ~30 audio dB samples, 6 power readings at 5 s polling, and ~60 YAMNet classification windows — sufficient for all baselines.)*

1. Discover Kasa smart plugs on the network, authenticate
2. Start YOLO, camera, and audio streams
3. YOLO processes frames, builds initial object/person inventory
4. Audio records baseline noise floor (mean dB, std) + learns the set of YAMNet classes that are "normal" for this room (e.g. persistent fan hum, computer noise) so they can be suppressed from Reasoner context
5. YAMNet model loads (~15MB) and the class whitelist is installed
6. Power draw recorded for idle lamp and fan
7. After calibration: status shifts to "Monitoring", narration begins

### Normal operation (quiet room)

1. **YOLO** runs every frame (~30fps), updates machine state in WorldState
2. **Audio** streams continuously, updates dB level + runs YAMNet on rolling 1s windows every ~500ms, updates `top_classes` / `dominant_class` / `class_history` in WorldState
3. **Smart plugs** push power state changes
4. No events firing → no Gemini calls
5. **Background refresh** every 30–60s: Observer gets a frame + current YOLO state, generates brief status narration ("Room quiet. One person at desk. Lamp on at 9 watts."), keeps semantic state fresh
6. Layer 0 can generate simple narration from structured data alone between refreshes

### Event detected (two-beat rhythm, when routing escalates)

1. YOLO detects: `person_3 pose: sitting → standing` + `zone: desk → door`
2. `event_detector.py` emits `zone_transition` event
3. Event bus triggers **Observer** (Gemini Flash) immediately
4. Observer receives: current frame + 2 buffered frames + YOLO signals + audio level + plug state
5. **Beat 1 (~1s):** Gemini responds with narration + escalate flag:
   ```
   narration: "Person stood up from desk and is walking toward the door."
   escalate: true
   escalate_reason: "possible departure; may need to actuate devices"
   ```
6. Beat 1 narration sent to TTS → spoken aloud immediately
7. World state updated with factual description
8. Routing policy checks: `zone_transition` not in REASONER_ALWAYS, but `escalate=true` → Reasoner fires
9. **Beat 2 (~3s):** Claude responds: "You've been at the desk for about 90 minutes. Good time for a stretch. I'll keep an eye on the room."
10. Beat 2 narration sent to TTS → spoken after Beat 1 finishes

### Event detected (routine — Beat 1 only)

1. YOLO detects: `person_3 pose: sitting → leaning_forward` (minor pose drift while typing)
2. `event_detector.py` emits `pose_change` event
3. Event bus triggers **Observer**
4. **Beat 1 (~1s):** Gemini responds:
   ```
   narration: "Person leaned forward, likely focused on screen."
   escalate: false
   escalate_reason: "minor posture adjustment, no action needed"
   ```
5. Beat 1 TTS plays. Pipeline stops. No Claude call. ~$0.00012 total.

### Event escalation (two-beat)

1. YOLO detects: `person_3 lost` (left frame entirely)
2. **Beat 1 (Observer):** "Room appears empty. Person left the frame."
3. **Beat 2 (Reasoner):** Checks world state — person departed after 90 min, audio dropped to baseline, time is 2:30 PM (normal hours). "Looks like you stepped out. Turning off the lamp to save energy. Fan stays on to cool the room." → lamp OFF

### Security situation (two-beat)

1. YOLO detects: new `person_8` appeared (not previously tracked), time = 3:12 AM
2. Audio: spike of +20dB above baseline + YAMNet reports `door (0.71), footsteps (0.22)`
3. **Beat 1 (Observer, ~1s):** "New person detected entering from the right side. Door sound and footsteps classified."
4. **Beat 2 (Reasoner, ~3s):** Checks context — 3:12 AM, outside baseline occupancy hours (8 AM–11 PM), unfamiliar tracking ID, entry pattern is unusual. "This is unusual — it's 3:12 AM, well outside normal occupancy hours. The person entered quickly and is not following a typical pattern. Turning on all lights and logging a security alert." → lamp ON, event logged as security alert

### Fatigue / wellness detection (two-beat, background refresh)

1. Background refresh fires (60s cadence, quiet room)
2. **Beat 1 (Observer):** "Person at desk, seated. Posture slightly slouched. No changes in environment."
3. **Beat 2 (Reasoner):** Reviews world state history — person has been seated for 2h 14m, pose data shows progressive slouch over last 30 min, head position dropping. "You've been sitting for over two hours and your posture has been dropping. Consider taking a break — maybe stretch or grab some water."

---

## 6. Dashboard

Single-page live view with five panels:

1. **Webcam feed** — live video with YOLO overlays: bounding boxes, tracking IDs, pose skeleton, zone labels
2. **Sensor strip charts** (last 60s) — audio dB level, motion intensity, lamp power draw, fan power draw
3. **World State panel** — formatted live-updating cards showing entities, counts, zones, semantic summary
4. **Agent reasoning feed** — scrolling log, color-coded:
   - Layer 0 events (gray): "person_3 zone: desk→door"
   - Observer narration (blue): "Person walked toward the door"
   - Investigator diagnosis (cyan): "New person detected, classifying as visitor"
   - Actions (green): "Lamp OFF"
5. **Event timeline** — horizontal strip with clickable markers showing notable events over the last hour
6. **System status bar** — current mode (Calibrating/Monitoring/Investigating), YOLO fps, API call count, running cost

---

## 7. Day-by-Day Build Plan (rough sequencing, not a schedule)

The blocks below are in the order we *expect* to tackle things — perception first so we have signals to reason over, then agents, then the dashboard. Hour estimates are guesses and will absolutely be wrong. Use them as "is this a big task or a small one" signals, not as deadlines.

### Day 1 (≈8 hours) — Perception & Core Loop

| Block | Task | Hours |
|---|---|---|
| 1 | Project scaffolding, dependencies, API keys, Kasa plug discovery + test toggle | 1.0 |
| 2 | Camera capture with OpenCV + frame buffer (rolling 30s in memory) | 0.5 |
| 3 | YOLO engine: detection + tracking + pose estimation running on webcam feed | 2.0 |
| 4 | Zone map: define room zones (desk, door, couch, etc.) by pixel regions | 0.5 |
| 5 | Event detector: consume YOLO output, emit structured events (new_person, lost_person, pose_change, zone_transition, object_moved) | 1.5 |
| 6 | ✅ Audio streaming with sounddevice + dB monitoring + spike detection + YAMNet classification on rolling 1s windows + whitelist filter + temporal smoothing. Also added speech transition events (speech_start / speech_end) that fire on silence↔speech transitions only, not on steady state — mirrors EventDetector's pose_change pattern. | 1.5 |
| 7 | ✅ Smart plug integration: `PlugManager` wraps python-kasa in a private asyncio loop (daemon thread). Discovers KP125M plugs by alias (broadcast + IP-hint fallback), polls power every 5 s, exposes synchronous `turn_on/off/state` API. Wired into main.py: discovery runs non-blocking in a side thread; plug status shows in console status line and cv2 overlay. | 0.5 |
| 8 | ✅ Smoke test: `main.py` orchestrator wires camera + YOLO + audio into a single loop. Events from both EventDetector and AudioMonitor merge into one stream, print to console, and render on a unified cv2 overlay (zones, event log, track status, audio dB/class). Graceful Ctrl+C/q shutdown with session summary. | 0.5 |

**End-of-Day-1 checkpoint:** YOLO is running on your webcam with tracking IDs and pose estimation visible. Audio levels are streaming. Smart plugs respond to commands. Structured events are firing when you move, stand up, or leave frame. Everything prints to console. No Gemini/narration yet.

### Day 2 (≈8 hours) — Intelligence & Actions

| Block | Task | Hours |
|---|---|---|
| 1 | ✅ World state schema + update logic (YOLO feeds machine state, Gemini feeds semantic state). Implemented `agents/world_state.py`: thread-safe `WorldState` class with `EntityState`, `AudioSnapshot`, `DeviceState`, `Baselines` dataclasses. Updated every main-loop tick via `update_from_yolo/audio/devices/push_event`. Snapshotable for LLM prompts. Added `zones_for(tid)` to EventDetector. Wired into `main.py` with 'd' debug keybind to dump snapshot. | 1.0 |
| 2 | Baseline calibration module: ~30 s learning phase for audio floor, power profile, occupancy (reduced from original 5–10 min — 30 s gives enough samples for all baselines) | 1.0 |
| 3 | Observer agent (Beat 1): Gemini 2.5 Flash integration (changed from 2.0 Flash — being shut down June 2026). Migrate to `google-genai` SDK. Event-driven calls with frame buffer + YOLO signals, quick factual narration, world state JSON updates | 2.0 |
| 4 | Reasoner agent (Beat 2): Claude Sonnet 4.6 integration — receives Observer output + full context, deep reasoning, judgment, action decisions, thoughtful narration | 2.0 |
| 5 | Two-beat TTS pipeline: Beat 1 plays immediately, Beat 2 queues after Beat 1 finishes | 0.5 |
| 6 | Decision module: lamp follows occupancy, fan follows extended presence, anomaly alerts spoken | 1.0 |
| 7 | Integration test: walk-out/walk-in, clap, new person, sit-too-long scenarios | 0.5 |

**End-of-Day-2 checkpoint:** Full agentic loop works end-to-end. Walk out → lamp off with spoken explanation. Walk in → lamp on. Clap → investigation triggered. New person → announced. Agent narrates through speakers. Everything runs autonomously.

### Day 3 (≈6 hours) — Dashboard & Demo Polish

| Block | Task | Hours |
|---|---|---|
| 1 | FastAPI server + WebSocket hub pushing all layer data to frontend | 1.0 |
| 2 | Dashboard: webcam feed with YOLO overlay | 0.5 |
| 3 | Dashboard: sensor charts, world state panel, reasoning feed | 1.5 |
| 4 | Dashboard: event timeline + system status bar | 1.0 |
| 5 | Prompt tuning: narration style (concise, physics-oriented, professional) | 0.5 |
| 6 | Demo script rehearsal: run through all essential moments 3–5 times | 1.0 |
| 7 | Buffer for bugs, edge cases, polish | 0.5 |

**End-of-Day-3 checkpoint:** Polished, demo-ready. Dashboard looks clean with live YOLO overlays. Narration feels natural. Every essential demo moment is reliable. Cost counter shows <$0.05 for the rehearsal.

### Fallback compression (if only 2 days)
Drop the full Next.js dashboard. Use Streamlit for a basic UI (2 hours). Skip event timeline and system status bar. Focus on the agentic behavior being solid — the demo can work with just the webcam window + YOLO overlay + spoken narration + smart plug actions, no dashboard needed.

---

## 8. Demo Script

Target duration: **3 minutes**.

| Time | Action | What the audience sees/hears |
|---|---|---|
| 0:00 | Launch the app | Dashboard appears, YOLO overlay on webcam, "Calibrating — learning this room..." |
| 0:30 | Calibration completes | "Baseline established. 1 person at desk. Ambient noise 34 dB. Lamp on at 9 watts. Monitoring." |
| 0:45 | Sit normally | Background refresh: Beat 1: "Person at desk, seated, no changes." Beat 2: "You've been working steadily. Room conditions are comfortable." |
| 1:00 | Stand up, walk to door | Beat 1 (~1s): "Person stood up and is walking toward the door." Beat 2 (~3s later): "Looks like you're heading out. You've been working about 90 minutes — good time for a break." |
| 1:15 | Leave the frame | Beat 1: "Room appears empty." Beat 2: "Turning off the lamp. Keeping fan running to cool the room while you're away." **Lamp clicks off.** |
| 1:30 | Walk back in | Beat 1: "Person detected in frame." Beat 2: "Welcome back. Restoring the lamp." **Lamp clicks on.** |
| 1:45 | Clap twice | Beat 1: "Clap detected — sharp percussive sound." Beat 2: "That was a clap (YAMNet confidence 0.82) — non-threatening acoustic event. Audio hit 56 dB, 22 above baseline. Logged." |
| 2:00 | Phone rings off-camera | Beat 1: "Phone ringing detected — no visible source in frame." Beat 2: "Audio-only event: ringtone classified with 0.78 confidence. Not acting on it, but demonstrates the agent hears things the camera can't see." |
| 2:15 | Second person walks in | Beat 1: "New person detected entering from the right. Door sound and footsteps classified." Beat 2: "Someone just entered. YAMNet confirms door + footsteps, YOLO confirms new tracking ID — three independent signals agree. Announcing their arrival." |
| 2:45 | Show the dashboard | Walk through: event timeline with all events, reasoning feed showing Beat 1 (blue) vs Beat 2 (cyan) entries, cost counter showing ~$0.04 |
| 3:00 | Wrap | "This same architecture — sense, fuse, reason, act — is what Archetype builds for manufacturing, construction, and smart cities. I just pointed it at a room." |

---

## 9. Cost Model (rough — update once we have real call counts)

These numbers are ballpark estimates from the planning phase. Once we're actually running, replace them with measured counts from a representative session. The order-of-magnitude matters (a few dollars per hour, not a few hundred); the exact figures do not.

With the hybrid routing policy (Section 2.5), the Reasoner fires on ~30–50% of Observer calls. The Observer still fires on every structural event and refresh; only Claude is gated.

| Component | Rate | Typical qty/hr | $/hr |
|---|---|---|---|
| YOLO (local) | Free | Continuous | $0.00 |
| Audio monitoring (local) | Free | Continuous | $0.00 |
| Smart plug reads (local) | Free | Continuous | $0.00 |
| Observer event-driven (Gemini Flash) | $0.00012/call | ~40–80 events | $0.005–0.010 |
| Observer background refresh (Gemini Flash) | $0.00012/call | ~60–120 (30–60s cadence) | $0.007–0.014 |
| Reasoner — routed calls only (Claude Sonnet) | ~$0.003/call | ~30–80 (routed from ~100–200 Observer calls) | $0.09–0.24 |
| TTS (edge-tts) | Free | Unlimited | $0.00 |
| **Total** | | | **$0.10–0.26/hr** |

The Reasoner remains the primary cost driver, but routing cuts calls roughly in half vs. the naive "always fire" design. To reduce cost further during development, set `REASONER_ENABLED = False` in config.py to run Layer 0 + Observer alone (drops to ~$0.01/hr).

Interview demo (30 min): **<$0.15**
Full development/testing over 3 days (Reasoner disabled most of the time): **<$8**

---

## 10. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| YOLO pose estimation unreliable on laptop webcam angles | Medium | Medium | Test early Day 1. Fall back to detection-only tracking (no pose) if needed. Pose is a nice-to-have, not essential for core demo. |
| Kasa KP125M plug requires cloud auth, discovery fails | Medium | High | Set up Kasa app + account before Day 1. Pre-discover and hardcode plug IPs. Keep a "mock plug" fallback in code. |
| YOLO tracking loses IDs (same person gets new ID) | Medium | Medium | Use Ultralytics' built-in BoT-SORT tracker. Accept occasional re-ID as tolerable for MVP. Observer resolves via visual context. |
| Gemini latency on event-driven calls feels sluggish | Low | Low | Two-beat design handles this — Beat 1 is fast by design. Beat 2 (Claude) takes 2-3s but that's expected and feels natural. |
| Claude Sonnet latency varies (2-5s) | Medium | Medium | Beat 2 is async. Beat 1 already played, so the user isn't waiting in silence. If Claude is slow, the system still feels responsive. |
| Audio in demo room triggers false spikes | Medium | Medium | Calibration phase sets noise floor. Minimum spike duration filter (sustained >0.5s). |
| Webcam lighting/angle is poor | Low | High | Test in actual demo environment ahead of time. YOLO is robust to moderate lighting variation. |
| API outage mid-demo | Low | High | Cache last Observer response. Degrade to Layer 0 narration from YOLO structured data (less fluent but functional). |

---

## 11. Explicitly Out of Scope

- Facial recognition or identifying specific people by name
- Training any custom ML models
- Persistent storage across sessions (everything in memory)
- Mobile app or cloud deployment
- Multi-room or multi-camera support
- Historical replay / time-travel debugging
- User authentication on dashboard
- Voice input / conversational interaction with the agent

---

## 12. Interview Talking Points

1. **Three-tier architecture** — local perception (YOLO) → fast factual description (Gemini Flash) → deep reasoning and action (Claude Sonnet). Why each tier exists and the cost/capability/latency tradeoff.
2. **Two-beat rhythm** — fast acknowledgment followed by thoughtful interpretation. The system feels both responsive AND intelligent. Mirrors how humans process events (notice first, reason second).
3. **Event-driven design** — API calls only fire when YOLO detects meaningful changes, keeping every call purposeful. Background refreshes prevent drift during quiet periods.
3a. **Hybrid Reasoner routing (Section 2.5)** — the Reasoner does not fire on every Observer call. A static list of high-stakes event types always escalates, and the Observer tags borderline cases with an `escalate` flag. Trivial events (minor pose twitches, low-magnitude audio) get Beat 1 only. This is a deliberate design choice: the Observer IS the fast triage layer, and making the Reasoner run every time undermines that separation. The savings (50–70% of Claude calls) are a bonus; the main reason is conceptual integrity.
4. **YOLO as structured perception** — detection + tracking + pose + zones. The local model produces machine-readable state; the LLMs produce human-readable meaning and judgment. YAMNet plays the analogous role for audio: turning raw waveforms into structured class labels that the LLMs can reason over, rather than asking the LLMs to interpret raw dB levels.
5. **Learned baselines** — the agent adapts to the environment rather than shipping with hardcoded thresholds. Calibration phase establishes what "normal" looks like for this specific room.
6. **Multi-modal fusion** — video + audio (level + YAMNet semantic classes) + power together resolve ambiguity that any single modality misses. Example: "person left" is confirmed by YOLO (lost tracking) + audio (level drop, YAMNet speech_ended) + power (no appliance change). "Security concern" requires YOLO (unfamiliar person) + time context (3 AM) + YAMNet (door + footsteps classes). YAMNet specifically lets the agent reason about *audio-only* events (phone ringing, cough, music) that have no visual signal at all.
7. **Separation of perception from judgment** — Gemini Flash describes what happened (factual). Claude Sonnet judges what it means and what to do (reasoning). This is a deliberate architectural choice that maps to Archetype's separation of Newton (perception) from Physical Agents (reasoning + action).
8. **Direct alignment with Archetype's architecture** — Layer 0 is the sensor layer. Layer 1 is the "Newton"-style perception model. Layer 2 is the "Physical Agent" reasoning layer. Same pattern, different scale.
9. **Extension path** — swap the sensors (industrial accelerometers, factory cameras, IoT feeds) and the same agent architecture applies. That's the platform thesis.

---

## 13. Stretch Goals (if Day 3 has slack)

- **Voice input** — let you ask the agent questions ("What happened while I was out?")
- **Appliance classification from power signatures** — detect what's plugged in based on power draw patterns
- **Time-of-day context** — factor sunrise/sunset and daily routines into reasoning
- **Occupancy prediction** — "you usually take a break around now"
- **YOLO segmentation** — use instance segmentation for richer scene understanding beyond bounding boxes
- **Multi-person interaction reasoning** — detect if two people are conversing, collaborating, or ignoring each other
- **Upgrade audio classifier** — swap YAMNet for EfficientAT MN10 (same lightweight profile, AST-class accuracy) for a free quality bump, or add CLAP/BEATs behind an endpoint as a second-stage deep-look triggered only on high-magnitude / ambiguous spikes. The `AudioClassifier` interface in `perception/audio.py` is designed for this swap.
- **Speech transcription** — add Whisper (local) for on-device transcripts, feeding raw utterances into WorldState so the Reasoner can pick up intent, not just acoustic class. Currently out of scope; listed here because it's the natural next modality after audio classification.

# Physical Room Agent

An always-on Physical AI agent that turns a single room into something it can perceive, narrate, and act on. It fuses live video, microphone audio, and smart-plug telemetry into one running understanding of the room. It reasons about who's there and what they're doing, speaking what it sees out loud, and controlling connected devices.

## Demo

![Physical Room Agent demo](docs/physical-ai-demo.mp4)

[Watch the demo](docs/physical-ai-demo.mp4)

## What it can do

- **Watches and tracks people** — sees who enters, leaves, sits, stands, walks, and which zone of the room they're in.
- **Listens and classifies sound** — speech, alerts, unusual noises, tuned to your room's baseline.
- **Narrates the moment** — speaks two beats: a fast factual one, then a slower interpretive one.
- **Reasons about activity** — infers what's happening (focused work, a call, a break).
- **Compounds its understanding over time** — its model of the session keeps getting richer the longer it runs, instead of resetting on every event.
- **Controls connected devices** — toggles smart plugs based on judgment, not fixed rules. My setup uses a lamp and a fan.
- **Refuses its own bad ideas** — code-level guardrails block unsafe or redundant actions before they leave the agent.

## Architecture

Three layers with a "two-beat rhythm":

| Layer | Role | Models / Libs | Latency |
|-------|------|---------------|---------|
| **0 — Perception** | Continuous local sensing. Produces structured events. No LLM calls. | YOLO26n-pose (detection + BoT-SORT tracking + pose), YAMNet (audio), python-kasa (plugs) | real-time |
| **1 — Observer** | Fast factual description of what just happened. Emits Beat 1 narration + updates world state + an `escalate` flag. | Gemini 2.5 Flash (`thinking_budget=0`) | ~1s |
| **2 — Reasoner** | Deep judgment and device decisions. Compounds a session narrative across calls. | Claude Sonnet 4.6 (with ephemeral prompt caching) | ~2–3s |

Perception runs constantly. When YOLO or YAMNet detects something, it wakes the Observer; if the Observer flags `escalate=true` (or the event is a must-escalate type), the Reasoner takes over with judgment and device decisions. Shared state lives in an in-memory `WorldState`.

## Data flow

```
                                   camera + mic + plug telemetry
                                                 │
                                                 ▼
   ┌────────────────────┐         ┌─────────────────────────────────┐
   │                    │ ◄─────► │           Perception            │
   │                    │         └────────────────┬────────────────┘
   │                    │                          │ Events:
   │                    │                          │   new_person · pose_change ·
   │                    │                          │   zone_transition · audio_spike ·
   │                    │                          │   speech_start
   │                    │                          │ + current frame + WorldState snapshot
   │                    │                          ▼
   │                    │         ┌─────────────────────────────────┐
   │                    │ ◄─────► │            Observer             │ ──► Observer Narration
   │ WorldState         │         └────────────────┬────────────────┘
   │                    │                          │ world_state_update
   │ shared in-memory   │                          │ escalate flag
   │ state, read and    │                          ▼
   │ written by every   │         ┌─────────────────────────────────┐
   │ layer. Compounds   │ ◄─────► │            Reasoner             │ ──► Reasoner Narration
   │ understanding      │         └────────────────┬────────────────┘
   │ across the session │                          │ lamp / fan commands
   │                    │                          │ session_narrative · activity_label
   │                    │                          ▼
   │                    │         ┌─────────────────────────────────┐
   │                    │         │         DecisionEngine          │
   │                    │         │   5 guardrails:                 │
   │                    │         │     override lockout · cooldown │
   │                    │         │     idempotency · no-person-ON  │
   │                    │         │     plug availability           │
   │                    │         └────────────────┬────────────────┘
   │                    │                          │ accepted toggles
   │                    │                          ▼
   └────────────────────┘                     ┌─────────┐
                                              │  Plugs  │
                                              └─────────┘

                      Each layer broadcasts data to the Dashboard via WebSocket.
```

## Design highlights

| Decision | Why it matters |
|---|---|
| **Hybrid Reasoner routing** | Most events stop at the cheap Observer; the expensive Reasoner only fires when something actually warrants judgment. Main cost and latency lever. |
| **Compounding session model** | The Reasoner rewrites its own `session_narrative` on every call — understanding compounds across the session instead of resetting each tick. |
| **Code-enforced guardrails** | Five hard checks run before any device command leaves the agent. The Reasoner's judgment is policy; the guardrails are law. |
| **30-second calibration** | At startup the agent learns the room's audio baseline, idle power, and ambient sounds — anomaly detection is tuned to *this* room. |

## Repo layout

```
perception/   # Layer 0: camera, YOLO, zones, event detection, audio/YAMNet, smart plugs
agents/       # Layers 1 & 2 + shared state: observer, reasoner, world_state, routing,
              #   decisions, baselines, empty_room_watcher
actuators/    # TTS speaker (Beat 1 / Beat 2 queue)
server/       # FastAPI + WebSocket hub + thread→async broadcaster bridge
dashboard/    # Next.js 16 + React 19 + Tailwind v4 frontend (App Router)
tests/        # pytest — perception smoke + DecisionEngine + EmptyRoomWatcher + WorldState
config.py     # API keys, thresholds, device IPs, zone definitions, routing config
main.py       # Orchestrator — starts all daemons, runs calibration, drives the main loop
```

## Tech stack

- **Python 3.11** — core runtime, threaded + async
- **Gemini 2.5 Flash** — Observer model (`thinking_budget=0`, via `google-genai`)
- **Claude Sonnet 4.6** — Reasoner model, with ephemeral prompt caching (via `anthropic`)
- **Ultralytics YOLO26n-pose** — detection, BoT-SORT tracking, 17-keypoint pose (MPS on Apple Silicon)
- **TensorFlow/YAMNet** — 521-class audio monitoring with persistence smoothing
- **FastAPI + WebSockets** — backend state stream and MJPEG video stream
- **Next.js 16 + React 19 + Tailwind v4** — App Router dashboard with live agent log panels
- **python-kasa** — TP-Link KP125M smart plug control + energy telemetry (KLAP auth)
- **edge-tts** — Microsoft cloud TTS → MP3 → afplay/ffplay

## Threading model

Perception, LLM calls, TTS, and the dashboard each run on their own daemon thread so they never block each other — the main loop queues Observer/Reasoner work and polls for results each tick, while a broadcaster bridges the sync perception world to FastAPI's async event loop.

```
main thread (~30 Hz event loop)
  ├── camera capture          (daemon)
  ├── YOLO inference          (daemon — MPS on Apple Silicon)
  ├── audio capture + YAMNet  (daemon)
  ├── smart plug poller       (daemon — asyncio loop wrapped in a thread)
  ├── ObserverWorker          (daemon — Gemini calls, queued)
  ├── ReasonerWorker          (daemon — Claude calls, queued)
  ├── TTS speaker             (daemon — Beat 2 waits on Beat 1)
  ├── FastAPI + WebSocket     (daemon — dashboard server + broadcaster bridge)
  ├── EmptyRoomWatcher tick   (in-loop, pure logic)
  └── DecisionEngine          (in-loop — guardrails before any toggle)
```

## Setup

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` at the repo root:

```
GEMINI_API_KEY=...
CLAUDE_API_KEY=...
KASA_USERNAME=...    # TP-Link Kasa account email (KLAP auth)
KASA_PASSWORD=...
```

Smart plug IPs, camera/audio indexes, zones, and thresholds live in `config.py`. Plug IPs are hints only — the app re-discovers plugs by alias (`light`, `fan`) at startup, since hotspot IPs rotate.

## Running

The agent runs as two processes side by side — the Python orchestrator and the dashboard. The dashboard is where you actually see what the agent is doing, so run both.

In one terminal, start the agent:

```bash
source venv/bin/activate
python main.py
```

On startup it runs a ~30s calibration phase before the three layers go hot. The Reasoner can be disabled in `config.py` during development to keep API costs near zero.

In a second terminal, start the dashboard:

```bash
cd dashboard
npm install
npm run dev
```

Opens at `http://localhost:3000` and streams MJPEG video, perception events, Observer narration, Reasoner narration, and DecisionEngine outcomes over WebSocket from `http://127.0.0.1:8000`.

## Cost profile

Rough per-hour API spend with typical room activity:

- Layer 0 + Observer only: **~$0.01/hr**
- Full pipeline (Observer + Reasoner via hybrid routing): **~$0.10–0.26/hr**

Cost stays low because the pipeline is event-driven (not polling) and the Reasoner is gated — most events stop at Beat 1.

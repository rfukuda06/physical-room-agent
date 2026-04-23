# Newton-for-a-Room

A Physical AI room agent that watches, listens, and reasons about what's happening in a single room — then narrates it and acts on connected devices. Built as an Archetype AI interview MVP.

## Demo

![Demo screenshot](docs/demo.png)

<!-- Drop the screenshot at `docs/demo.png` (or update the path above). -->

## What it does

The agent treats one room as its whole world. A webcam, a microphone, and a pair of TP-Link Kasa smart plugs feed a three-tier perception → observation → reasoning pipeline. The output is a running two-beat narration ("someone just sat down at the desk" → "they look like they're starting a focus session — dimming the lamp") plus direct control of a lamp and fan.

## Architecture

Three layers with a "two-beat rhythm":

| Layer | Role | Models / Libs | Latency |
|-------|------|---------------|---------|
| **0 — Perception** | Continuous local sensing. Produces structured events. No LLM calls. | YOLO26n-pose (detection + BoT-SORT tracking + pose), YAMNet (audio), python-kasa (plugs) | real-time |
| **1 — Observer** | Fast factual description of what just happened. Emits Beat 1 narration + updates world state. | Gemini 2.5 Flash (`thinking_budget=0`) | ~1s |
| **2 — Reasoner** | Deep judgment and device decisions. Gated by hybrid routing — only fires for must-escalate event types or when the Observer flags `escalate=true`. | Claude Sonnet 4.6 | ~2–3s |

Shared state lives in an in-memory `WorldState` object. No database, no cross-session persistence.

See `DESIGN.md` for module-level diagrams and data contracts, and `ARCHITECTURE_AND_BUILD_PLAN copy.md` for the living build plan.

## Repo layout

```
perception/   # Layer 0: camera, YOLO, zones, event detection, audio/YAMNet, smart plugs
agents/       # Layers 1 & 2: observer, reasoner, world state, baselines, routing, decisions
actuators/    # TTS + smart plug control wrappers
server/       # FastAPI + WebSocket hub
dashboard/    # Next.js + React + Tailwind frontend
config.py     # API keys, thresholds, device IPs, zone definitions
main.py       # Orchestrator — starts all layers, manages lifecycle
```

## Tech stack

- **Python 3.11** — core runtime
- **Ultralytics YOLO26n-pose** — detection, BoT-SORT tracking, 17-keypoint pose
- **sounddevice + TensorFlow/YAMNet** — 521-class audio monitoring
- **python-kasa** — TP-Link KP125M smart plug control + energy telemetry
- **google-genai** — Gemini 2.5 Flash (Observer)
- **anthropic** — Claude Sonnet 4.6 (Reasoner)
- **edge-tts / pyttsx3** — text-to-speech
- **FastAPI + WebSockets** — backend
- **Next.js + React + Tailwind + Recharts** — dashboard

## Setup

Full checklist is in `SETUP.md`. Short version:

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

Smart plug IPs, camera/audio indexes, zones, and thresholds live in `config.py`. Plug IPs are hints only — the app re-discovers plugs by alias (`light`, `fan`) at startup.

## Running

```bash
source venv/bin/activate
python main.py
```

On startup the agent runs a ~30s calibration phase to learn the room's audio floor, power profile, occupancy pattern, and persistent YAMNet classes to suppress. After calibration the three layers run concurrently. The Reasoner can be disabled in `config.py` during development to keep API costs near zero.

Dashboard (optional):

```bash
cd dashboard
npm install
npm run dev
```

## Cost profile

Rough per-hour API spend with typical room activity:

- Layer 0 + Observer only: **~$0.01/hr**
- Full pipeline (Observer + Reasoner via hybrid routing): **~$0.10–0.26/hr**

Cost stays low because the pipeline is event-driven (not polling) and the Reasoner is gated — most events stop at Beat 1.

## Status

Day 2 in progress. Layer 0 is fully implemented and tested. WorldState, calibration, and the Observer (Gemini 2.5 Flash) are wired into the main loop with 0.5s event debouncing and a 30s periodic refresh. Reasoner, TTS, and device-decision logic are the next blocks.

## Docs

- `ARCHITECTURE_AND_BUILD_PLAN copy.md` — living build plan (read Section 0 first)
- `DESIGN.md` — module diagrams and data contracts
- `SETUP.md` — per-session startup checklist
- `LEARNING.md` — running log of surprising findings and corrected assumptions
- `prompt_engineering.md` — notes on Observer/Reasoner prompt iteration

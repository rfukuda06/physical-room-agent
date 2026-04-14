# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Habbits I want you to have:
- Anytime you make a change to the code, briefly explain what change you made and how it works, especially when its a new software/api/concept being integrated. Assume you are explaining to a college student that has only surface level knowledge.
- Specifically explain how FastAPI, Next.js, and React work when implementig them as I want to learn these things for the first time.
- When context builds up to a point of affecting claude code's performance, suggest to the user effective context management strategies.

**Newton-for-a-Room** — a Physical AI Room Agent built as an Archetype AI interview MVP. The agent observes a room via camera, microphone, and smart plug sensors; reasons about events using a two-LLM pipeline; and acts through voice narration and device control (lamp, fan).

The full architecture and build plan is in `ARCHITECTURE_AND_BUILD_PLAN copy.md`. Read it before making significant design decisions — and read Section 0 first, which frames the doc as a living proposal.

## Status

Greenfield project, planning phase. The architecture document is a **working plan, not a spec**: the three-tier shape (local perception → fast description → deep reasoning) and the two-beat rhythm are the load-bearing ideas; most specifics (event type names, thresholds, routing rules, cost numbers, timelines, dashboard layout) are first-draft placeholders that will be rewritten as we discover how things actually behave. When something changes materially during implementation, update the doc so it stays accurate.

## Architecture

Three-tier event-driven system with a "two-beat rhythm":

- **Layer 0 (Local Perception):** YOLO (detection, tracking, pose) + sounddevice/YAMNet (audio classification) + python-kasa (smart plugs). Runs continuously, produces structured events. No LLM calls.
- **Layer 1 (Observer — Gemini 2.0 Flash):** Triggered by Layer 0 events. Fast factual description (~1s). Produces Beat 1 narration + world state JSON update.
- **Layer 2 (Reasoner — Claude Sonnet):** Gated by a hybrid routing policy (see `agents/routing.py` and architecture doc §2.5). Fires only when the event type is in `config.REASONER_ALWAYS` OR the Observer's output sets `escalate=true`. Routine events get Beat 1 only, no Claude call.

Shared state lives in an in-memory `WorldState` object — no database, no persistence across sessions.

## Planned Module Structure

```
perception/        # Layer 0: camera, YOLO, zones, event detection, audio/YAMNet, smart plugs
agents/            # Layers 1 & 2: observer (Gemini), reasoner (Claude), world state, baselines, decisions
actuators/         # TTS output, smart plug control wrappers
server/            # FastAPI + WebSocket hub
dashboard/         # Next.js + React + Tailwind (or Streamlit fallback)
config.py          # API keys, thresholds, device IPs, zone definitions
main.py            # Orchestrator: starts all layers, manages lifecycle
```

## Tech Stack

- **Python** — core runtime (async, event-driven)
- **Ultralytics YOLO v8/v11** — detection, tracking, pose estimation
- **sounddevice + TensorFlow/YAMNet** — audio monitoring and classification (521 AudioSet classes)
- **python-kasa** — TP-Link Kasa KP125M smart plug control + energy monitoring
- **google-generativeai** — Gemini 2.0 Flash (Layer 1 observer)
- **anthropic** — Claude Sonnet (Layer 2 reasoner)
- **edge-tts or pyttsx3** — text-to-speech
- **FastAPI** — backend server with WebSocket support
- **Next.js + React + Tailwind + Recharts** — dashboard (Streamlit as fallback)

## Key Design Decisions

- **Event-driven, not polling:** API calls fire only on meaningful YOLO/audio events + periodic 30-60s background refreshes. Combined with hybrid Reasoner routing, this keeps cost low (~$0.10-0.26/hr with Reasoner enabled, ~$0.01/hr without).
- **Hybrid Reasoner routing:** The Reasoner is NOT called on every Observer call — only on must-escalate event types or when the Observer flags `escalate=true`. This preserves the Observer's role as fast triage and prevents the architecture from collapsing into "two models called sequentially on everything."
- **Two-beat rhythm:** Beat 1 (Gemini, fast factual) plays immediately so the system feels responsive. Beat 2 (Claude, deep reasoning) follows ~2-3s later with judgment and actions.
- **Learned baselines:** 5-10 min calibration phase at startup learns the room's normal audio floor, power profile, occupancy pattern, and persistent YAMNet classes to suppress.
- **Reasoner is disableable:** During development, run Layer 0 + Observer only to save cost. Enable Reasoner for integration testing and demo.
- **YAMNet temporal smoothing:** Audio class must persist >= 2 consecutive windows before reporting, to avoid spurious classifications.

## Environment Variables (Expected)

Loaded from `.env` at the repo root via `python-dotenv` (see `config.py`). `.env` is gitignored.

```
GEMINI_API_KEY    # Gemini 2.0 Flash (Layer 1 Observer)
CLAUDE_API_KEY    # Claude Sonnet (Layer 2 Reasoner)
KASA_USERNAME     # TP-Link Kasa account email — required for KLAP auth on KP125M plugs
KASA_PASSWORD     # TP-Link Kasa account password
```

Smart plug IPs, camera/audio indexes, zones, and thresholds live in `config.py`. Plug IPs are hints only — we re-discover by alias (`light`, `fan`) at startup because hotspot IPs rotate.

## Getting Started Each Session

See `SETUP.md` for the step-by-step checklist (activate venv, verify deps, confirm `.env`, check hardware, run `main.py`). At the start of a working session, remind the user to `source venv/bin/activate` from the repo root if they haven't already — almost every `ModuleNotFoundError` in this project traces back to a non-activated venv.

**Python version:** 3.11 (the `venv/` directory uses `python3.11`).

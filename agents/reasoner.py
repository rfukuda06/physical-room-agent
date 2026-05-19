"""
Layer 2 — Reasoner agent (Claude Sonnet 4.6). Beat 2 of the two-beat rhythm.

How it works:
--------------
The Reasoner is the system's "slow thinker" — and more importantly, its memory.
It runs only when the routing policy escalates (see agents/routing.py): either
because the event type is in config.REASONER_ALWAYS, or because the Observer
flagged escalate=True, or on a minutely summary timer.

The Observer (Gemini Flash) already produced Beat 1 — a fast factual description
of the current moment. The Reasoner's job is fundamentally different: it maintains
a SESSION MODEL that compounds in intelligence over time, something the Observer
cannot do because it only sees the present.

Key concept — the session_narrative:
  On every call, the Reasoner reads its previous session_narrative from WorldState
  and rewrites it with new understanding. After 2 minutes it knows more than after
  30 seconds. After 10 minutes its model of the room is richer than anything any
  single Observer call could produce. This compounding is what makes the Reasoner
  genuinely smarter than the Observer over time, despite the Observer having better
  raw perception (it sees pixels; the Reasoner only gets text summaries).

Output contract (JSON, validated by Pydantic):
  {
    "narration": "Beat 2 spoken narration (≤40 words) or empty string",
    "lamp": "on" | "off" | null,
    "fan": "on" | "off" | null,
    "alert": false,
    "speak": true,
    "reasoning": "internal chain-of-thought, NOT spoken",
    "activity_label": "focused_work | idle | on_call | eating | break | active | transitioning | unknown",
    "session_narrative": "updated running interpretation of the session",
    "world_state_update": {
      "scene_description": "...",
      "activity_summary": "..."
    }
  }
"""

from __future__ import annotations

import base64
import json
import logging
import threading
import time
from typing import Literal, Optional

import anthropic
import cv2
import numpy as np
from pydantic import BaseModel, Field

import config
from agents.world_state import WorldState
from perception.camera import CameraCapture

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

class ReasonerWorldStateUpdate(BaseModel):
    scene_description: str = ""
    activity_summary: str = ""


class ReasonerOutput(BaseModel):
    """Structured output from one Reasoner call."""
    narration: str = Field(
        default="",
        description=(
            "Beat 2 spoken narration. 40-word hard cap. "
            "Empty string if Beat 1 covered everything — do not restate it."
        ),
    )
    lamp: Optional[Literal["on", "off"]] = Field(
        default=None,
        description=(
            "Lamp command: 'on', 'off', or null (no change). Null is the "
            "default — only command a toggle when policy in <action_rules> "
            "clearly applies. Guardrails in DecisionEngine may still refuse."
        ),
    )
    fan: Optional[Literal["on", "off"]] = Field(
        default=None,
        description=(
            "Fan command: 'on', 'off', or null (no change). Same rules as lamp."
        ),
    )
    lamp_reason: str = Field(default="", description="Short internal justification for the lamp action. Not spoken. Empty when lamp is null.")
    fan_reason: str = Field(default="", description="Short internal justification for the fan action. Not spoken. Empty when fan is null.")
    alert: bool = Field(default=False)
    speak: bool = Field(default=True)
    reasoning: str = Field(
        default="",
        description="Internal chain-of-thought. NOT spoken. Logged for debug.",
    )
    activity_label: str = Field(
        default="unknown",
        description=(
            "Current activity: focused_work | idle | on_call | eating | "
            "break | active | transitioning | unknown"
        ),
    )
    session_narrative: str = Field(
        default="",
        description=(
            "Updated running interpretation of the session. Builds on the "
            "previous narrative — never resets, only grows richer."
        ),
    )
    world_state_update: ReasonerWorldStateUpdate = Field(
        default_factory=ReasonerWorldStateUpdate,
    )


# ---------------------------------------------------------------------------
# System prompt — stable, cached across calls
#
# Minimum 2048 tokens required for Anthropic prompt caching to activate.
# Do NOT put timestamps or dynamic values here — those go in the user message.
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
<identity>
You are the Reasoner in Newton-for-a-Room, a physical AI room agent. Two LLMs \
work in sequence on every significant event:

  Observer (Gemini 2.5 Flash, ~1s): Sees raw camera frames. Describes what \
changed RIGHT NOW. Produces Beat 1 — fast, factual, present-tense.

  Reasoner (you, Claude Sonnet, ~2-3s): Sees the Observer's output plus the \
full history of the session. Maintains a running model. Produces Beat 2 — \
deeper, contextual, longitudinal.

Your fundamental structural advantage over the Observer: MEMORY THAT COMPOUNDS. \
The Observer sees only the present moment — it has no memory between calls. \
You own the past, the present, and the accumulated interpretation of what \
everything means together. Each call you update your session model. After \
ten minutes your model of this room is richer than anything any single Observer \
call could ever produce.

Your job is NOT to describe what just happened — the Observer already did that. \
Your job is to answer: what does this mean in the context of the full session? \
What does the pattern of events tell us? Is anything worth saying or doing that \
the Observer's snapshot view couldn't see?
</identity>

<session_model>
The most important thing you maintain is the SESSION NARRATIVE — a coherent \
paragraph (3-6 sentences) describing your evolving understanding of this session. \
It is stored in WorldState and passed to you as input on every call under \
the key "session_narrative". Read it. Then rewrite it incorporating everything \
you just learned.

What the session narrative should capture:
- How long the session has been running and its overall trajectory
- What the person is doing and how that has evolved over time
- Patterns you have noticed: sustained silence = focused work, recurring sounds, \
  movement rhythms, times the room emptied and refilled
- Your confidence in the occupancy count — BoT-SORT track ID churn is very \
  common (IDs climb into the dozens over minutes with one real person). Note \
  if the tracker is churning and what you believe true occupancy actually is
- Significant events and what you made of them at the time
- Anything you are currently monitoring or uncertain about

The narrative COMPOUNDS. It must be richer after ten minutes than after one \
minute. Do not write a fresh description of the current moment — synthesize \
the arc of the entire session so far.

You also maintain an ACTIVITY LABEL — a single string classifying the dominant \
current activity. Choose from:
  focused_work   — sitting, typing sounds, quiet, minimal movement, sustained
  idle           — sitting but no typing, no speech, no notable movement
  on_call        — speech detected while sitting
  eating         — hand-to-mouth gestures, brief pauses in other activity
  break          — standing, walking around, away from desk
  active         — lots of movement, multiple people interacting, dynamic
  transitioning  — occupancy change in progress or just occurred
  unknown        — not enough data yet (typically the first 30-60 seconds)

Update the activity_label on every call based on the full session context, \
not just the most recent event.
</session_model>

<observer_limitations>
The Observer has known limitations you must account for and note in your \
session narrative when relevant:

1. Track ID churn — BoT-SORT frequently re-assigns the same physical person \
   to a new track ID. new_person and lost_person events are extremely common \
   and usually do NOT represent real arrivals or departures. Track IDs climb \
   into the tens and hundreds over a session with a single real person present. \
   Use the Observer narration content as the authoritative signal for real \
   occupancy changes — not the raw event counts. If the Observer says "room \
   unchanged" on a new_person trigger, it is churn. Note this pattern in your \
   session narrative so you build up a calibrated sense of true occupancy.

2. Frame lag — The Observer's narration describes frames from zero to three \
   seconds before the API call returned. Fast events may be slightly stale.

3. Occupancy discrepancy — Gemini's people count (from frames) can differ \
   from the WorldState entity count (from the YOLO tracker). The narration \
   is more reliable for who is actually present.

4. Audio false positives — YAMNet misclassifies roughly 20-30 percent of \
   audio windows. Only treat audio as high-confidence when both an audio \
   classification AND a matching visual observation are present. Audio-only \
   signals are lower confidence.

5. Missed departures — If a person leaves quickly (under two seconds near the \
   door), the Observer may narrate "room unchanged" rather than catching it. \
   Cross-check lost_person events against the narration before concluding no \
   change occurred.

6. Camera angle artifacts — If the camera faces a reflective surface (mirror, \
   window), YOLO will detect reflections as separate persons. Note this in your \
   session narrative if it appears to be happening. A consistent ghost track \
   that never moves to the door zone is likely a reflection.
</observer_limitations>

<action_rules>
Lamp and fan: you may command "on", "off", or null (no change). Null is the
default — only command a toggle when one of these patterns is clearly present.
DecisionEngine enforces hard guardrails (cooldown, override lockout, no-person
guard, plug reachability) and will refuse toggles that violate them, but you
should not knowingly emit a refused command either — check DEVICE STATE first.

LAMP
  - Turn ON when a person is in the room AND the frame looks visibly dim AND
    the lamp is currently off.
  - Turn OFF when the room has been empty long enough that you receive the
    trigger room_empty_confirmed. Pair this with Fan OFF if the fan is on.
  - Do NOT command the lamp if DEVICE STATE shows lockout_active=true.
    The user touched it; respect that. You may narrate the acknowledgment
    exactly once on the first call where you see the override.

FAN
  - Turn OFF when speech has just started after a quiet stretch (the person
    appears to be on a call). The fan being on would be intrusive.
  - Turn ON again when the call ends (speech absent for a sustained period)
    AND the fan was on under agent control before the call started — check
    last_agent_command_intent in DEVICE STATE.
  - Otherwise leave the fan alone in v1. Do not invent thermal reasoning —
    you have no temperature signal.

GENERAL
  - Always populate lamp_reason / fan_reason with one short sentence per
    toggle you emit. Not spoken; this is your debug log.
  - When you command a toggle, narrate it in one short conversational line.
    The narration is the action's voice — don't actuate silently.
  - If DEVICE STATE shows cooldown_remaining > 0 or lockout_active=true for
    a device, do not command that device. Wait.
  - Never command ON when WorldState shows zero people present.

Alert decisions:
  - Set alert=true ONLY when at least two independent signals converge on a
    genuine security concern: unusual time of day AND unfamiliar activity AND
    unusual audio — all together.
  - Do not alert on routine occupancy changes or normal-hours events.

Speak decisions:
  - speak=true when you have something genuinely useful to say that the
    Observer did not cover, or when a minutely summary or device action
    warrants voicing.
  - speak=false when Beat 1 already covered everything and there is nothing
    meaningful to add. Set narration="" and speak=false together.
  - Always populate reasoning and update session_narrative regardless of speak.
  - For the minutely summary trigger (periodic_refresh_minutely): ALWAYS
    produce a narration, even if just confirming continuity.
</action_rules>

<narration_style>
When you do narrate (speak=true, narration is non-empty):
  - Hard cap: 40 words. No exceptions.
  - First word must be an action verb or natural opener.
  - Present tense. Conversational. Written to be spoken aloud.
  - Beat 2 should add something the Observer could NOT say — longitudinal \
    context, pattern recognition, session arc. If you would just restate \
    Beat 1, stay silent instead.
  - Never expose raw sensor values (dB, track IDs, confidence scores).
  - Never reference the Observer, Gemini, Beat 1, or internal system names.
</narration_style>

<output_format>
Respond with ONLY valid JSON. Start your response with { and end with }.
No text before or after the JSON object. No markdown fences.
{
  "narration": string,
  "lamp": "on" | "off" | null,
  "fan": "on" | "off" | null,
  "lamp_reason": string,
  "fan_reason": string,
  "alert": boolean,
  "speak": boolean,
  "reasoning": string,
  "activity_label": string,
  "session_narrative": string,
  "world_state_update": {
    "scene_description": string,
    "activity_summary": string
  }
}
</output_format>

<examples>
--- EXAMPLE 1: First minutely summary (session is 60s old) ---
Prior session_narrative: "Session just started. One person sitting at desk. Room quiet."
Trigger: periodic_refresh_minutely. Observer narration: "".
Situation: Person has been sitting continuously, no speech, two audio spikes both ignored.

Good output:
{"narration": "One minute in — looks like a quiet, focused work session so far.", "lamp": null, "fan": null, "lamp_reason": "", "fan_reason": "", "alert": false, "speak": true, "reasoning": "First minutely summary. Person has been sitting at desk the entire session. Two audio spikes that the Observer dismissed as ambient. No speech. Pattern looks like focused solo work. Worth noting — gives a sense of session context.", "activity_label": "focused_work", "session_narrative": "Solo work session, ~1 minute in. One person at the desk continuously since start. Very quiet — two brief audio spikes both appeared to be ambient noise with no visual reaction. Activity pattern consistent with focused computer work. Track ID churn is present (IDs changing frequently) but true occupancy appears to be one person.", "world_state_update": {"scene_description": "One person at desk", "activity_summary": "focused work at desk"}}

--- EXAMPLE 2: Minutely summary, nothing interesting ---
Prior session_narrative: "Solo work session, ~5 minutes in. Sustained focused work — continuous typing sounds, person has not moved from desk. One sneezing event at t=4min. Track ID churn very frequent. True occupancy: 1."
Trigger: periodic_refresh_minutely. Observer narration: "".
Situation: Nothing has changed. Same pattern continuing.

Good output:
{"narration": "Still deep in it — you haven't moved from the desk in six minutes.", "lamp": null, "fan": null, "lamp_reason": "", "fan_reason": "", "alert": false, "speak": true, "reasoning": "6-minute mark. Nothing new — same sustained focused work pattern. Minutely summary always narrates, so note the continuity.", "activity_label": "focused_work", "session_narrative": "Solo work session, ~6 minutes in. Sustained focused work throughout — continuous typing sounds, person has remained at desk the entire session. One sneezing event at t=4min. Track ID churn continues to be very frequent — BoT-SORT re-IDs happening every 30-60s. True occupancy confirmed as 1 person. No departures, no speech, consistent quiet work pattern.", "world_state_update": {"scene_description": "One person at desk", "activity_summary": "sustained focused work"}}

--- EXAMPLE 3: Escalation with session context adding real value ---
Prior session_narrative: "Solo work session, ~8 minutes in. Person typing continuously. Very quiet room."
Trigger: unusual_sound_class=Sneeze. Observer Beat 1: "A sneeze was heard."

Good output:
{"narration": "Bless you!", "lamp": null, "fan": null, "lamp_reason": "", "fan_reason": "", "alert": false, "speak": true, "reasoning": "Simple human response to a sneeze. Short and appropriate. Session context doesn't change the response here but confirms this is a normal work session moment.", "activity_label": "focused_work", "session_narrative": "Solo work session, ~8 minutes in. Person has been typing continuously at the desk. One sneezing event — confirmed by audio and visual observation. Very quiet, productive work pattern throughout. Track ID churn continues but true occupancy is consistently 1.", "world_state_update": {"scene_description": "One person at desk", "activity_summary": "focused work, brief sneeze"}}

--- EXAMPLE 4: Pattern recognition the Observer cannot do ---
Prior session_narrative: "Work session, 12 minutes in. Person has been at desk the whole time. Three separate audio spikes — all appeared to be ambient (typing, movement sounds). No speech. Focused work pattern."
Trigger: audio_spike. Observer Beat 1: "A sudden loud sound was detected."

Good output:
{"narration": "That's the fourth loud sound this session — probably just your environment. You seem unbothered.", "lamp": null, "fan": null, "lamp_reason": "", "fan_reason": "", "alert": false, "speak": true, "reasoning": "The Observer sees only this spike. I know this is the fourth in 12 minutes, all ambient, none correlated with any visual disturbance. Pattern suggests this is a noisy environment, not a concern. The cross-session context adds real value here.", "activity_label": "focused_work", "session_narrative": "Work session, ~12 minutes in. Person at desk continuously. Four audio spikes total — all ambient, no visual correlation, person unmoved each time. The environment appears to have periodic background noise. Sustained focus pattern with keyboard typing throughout. No speech, no departures.", "world_state_update": {"scene_description": "One person at desk", "activity_summary": "focused work, noisy environment noted"}}

--- EXAMPLE 5: A1 — sit down in a dim room ---
Prior session_narrative: "Session just started. No one in the room yet."
Trigger: new_person. Observer Beat 1: "Someone walked in and sat at the desk."
Situation: Frame is visibly dim. Lamp is off, fan is off, neither locked out.

Good output:
{"narration": "Got the light for you.", "lamp": "on", "fan": null, "lamp_reason": "Person just sat down in a visibly dim room; lamp off and not locked out.", "fan_reason": "", "alert": false, "speak": true, "reasoning": "Classic anticipation case — dim room + new arrival at desk. Lamp is off, no lockout. Turning on with a short narration.", "activity_label": "transitioning", "session_narrative": "Session just started. One person arrived and sat at the desk. Room was visibly dim — turned the lamp on automatically.", "world_state_update": {"scene_description": "One person at desk, lamp now on", "activity_summary": "settling in at desk"}}

--- EXAMPLE 6: R1 — call starts ---
Prior session_narrative: "Solo work session, ~8 minutes in. Quiet, sustained focused work."
Trigger: speech_detected. Observer Beat 1: "Person is talking — sounds like a phone call."
Situation: Person is sitting, speech persisted across windows, room had been quiet. Fan is on (agent turned it on earlier). Fan not locked out.

Good output:
{"narration": "Quieting the fan — sounds like you're on a call.", "lamp": null, "fan": "off", "lamp_reason": "", "fan_reason": "Speech just started after a long quiet stretch; fan on would be intrusive.", "alert": false, "speak": true, "reasoning": "Reading the room: sustained quiet then speech onset = call. Fan was on under agent control; turning it off improves call audio.", "activity_label": "on_call", "session_narrative": "Solo work session, ~8 minutes in. Quiet focused work until now; speech just started, looks like a phone call. Turned the fan off so it doesn't interfere.", "world_state_update": {"scene_description": "One person at desk, on a call", "activity_summary": "on a call"}}

--- EXAMPLE 7: R3 — user overrode the lamp ---
Prior session_narrative: "Solo work session, ~4 minutes in. Lamp came on when they sat down."
Trigger: periodic_refresh_minutely. Observer Beat 1: "".
Situation: DEVICE STATE shows lamp lockout_active=true (user just toggled it). last_manual_override happened recently.

Good output:
{"narration": "Noted — leaving the lamp how you set it.", "lamp": null, "fan": null, "lamp_reason": "Lockout active — user overrode the lamp. Respecting their choice.", "fan_reason": "", "alert": false, "speak": true, "reasoning": "User just manually changed the lamp; lockout is active. Acknowledge once and back off. Don't try to command the lamp again until lockout clears.", "activity_label": "focused_work", "session_narrative": "Solo work session, ~4 minutes in. Lamp came on when they sat down, but they manually changed it — leaving it alone for now. Otherwise quiet focused work.", "world_state_update": {"scene_description": "One person at desk, lamp under user control", "activity_summary": "focused work, user adjusted lamp"}}
</examples>
"""


# ---------------------------------------------------------------------------
# Reasoner — handles a single Claude API call
# ---------------------------------------------------------------------------

class Reasoner:
    """Layer 2 Reasoner agent — Claude Sonnet 4.6.

    Maintains a compounding session model via session_narrative in WorldState.
    Call `call(obs_result, trigger_event_types, frame)` to make one synchronous
    Claude API call. Use ReasonerWorker to run off the main thread.
    """

    _ALWAYS_FRAME_TYPES = frozenset({"new_person", "lost_person", "security_event"})
    _SKIP_FRAME_TYPES   = frozenset({"power_anomaly", "periodic_refresh_minutely",
                                      "periodic_refresh_hourly"})

    def __init__(self, world: WorldState, camera: CameraCapture) -> None:
        self._world = world
        self._camera = camera
        self._client = anthropic.Anthropic(api_key=config.CLAUDE_API_KEY)
        self._consecutive_failures: int = 0
        self._last_output: Optional[ReasonerOutput] = None

    # ----- public API -----

    def call(
        self,
        obs_result: dict,
        trigger_event_types: list[str],
        frame: Optional[np.ndarray] = None,
    ) -> Optional[ReasonerOutput]:
        """Make one synchronous Claude API call and return the parsed output."""
        if not config.REASONER_ENABLED:
            return None

        # 1. Full world state snapshot (includes session_narrative, activity_label,
        #    session_elapsed_s that the Reasoner reads to update its model)
        world_snapshot = self._world.snapshot_for_reasoner()

        # 2. Frame inclusion
        include_frame = self._should_include_frame(trigger_event_types, obs_result)
        frame_b64: Optional[str] = None
        if include_frame:
            f = frame if frame is not None else self._camera.latest_frame()
            if f is not None:
                try:
                    frame_b64 = self._encode_frame(f)
                except Exception as exc:
                    log.warning("Reasoner: frame encoding failed: %s", exc)

        # 3. Build user message
        content = self._build_user_message(
            obs_result, trigger_event_types, world_snapshot, frame_b64,
        )

        # 4. Call Claude
        try:
            response = self._client.messages.create(
                model=config.CLAUDE_MODEL,
                max_tokens=config.REASONER_MAX_TOKENS,
                system=[{
                    "type": "text",
                    "text": _SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=[{"role": "user", "content": content}],
            )
        except Exception as exc:
            log.warning("Reasoner: Claude API call failed: %s", exc)
            self._consecutive_failures += 1
            self._backoff_if_needed()
            return None

        # 5. Extract text block (thinking blocks prepend when enabled)
        raw: Optional[str] = None
        for block in response.content:
            if hasattr(block, "text"):
                raw = block.text
                break

        if not raw:
            log.warning("Reasoner: empty response (stop_reason=%s)", response.stop_reason)
            self._consecutive_failures += 1
            return None

        # Log cache metrics
        usage = response.usage
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        if cache_read:
            log.debug("Reasoner cache hit: %d read tokens", cache_read)

        # 6. Parse and validate
        output = self._parse_response(raw, response.stop_reason)
        if output is None:
            self._consecutive_failures += 1
            return None

        # 7. Success — write session model back to WorldState
        self._consecutive_failures = 0
        self._last_output = output

        self._world.apply_reasoner_update({
            **output.world_state_update.model_dump(),
            "session_narrative": output.session_narrative,
            "activity_label": output.activity_label,
        })

        return output

    # ----- internal helpers -----

    def _should_include_frame(
        self,
        trigger_event_types: list[str],
        obs_result: dict,
    ) -> bool:
        if config.REASONER_INCLUDE_FRAME_ALWAYS:
            return True
        if any(t in self._ALWAYS_FRAME_TYPES for t in trigger_event_types):
            return True
        if obs_result.get("escalate"):
            return True
        if trigger_event_types and all(t in self._SKIP_FRAME_TYPES for t in trigger_event_types):
            return False
        return True

    def _encode_frame(self, frame: np.ndarray) -> str:
        """Resize + JPEG compress, return base64 string (Anthropic vision format)."""
        h, w = frame.shape[:2]
        max_dim = config.OBSERVER_FRAME_MAX_DIM
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            frame = cv2.resize(
                frame,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )
        ok, buf = cv2.imencode(
            ".jpg", frame,
            [cv2.IMWRITE_JPEG_QUALITY, config.OBSERVER_FRAME_QUALITY],
        )
        if not ok:
            raise RuntimeError("cv2.imencode returned False")
        return base64.standard_b64encode(buf.tobytes()).decode("utf-8")

    def _build_user_message(
        self,
        obs_result: dict,
        trigger_event_types: list[str],
        world_snapshot: dict,
        frame_b64: Optional[str],
    ) -> list[dict]:
        """Build the messages[0].content list. Images first (Anthropic best practice)."""
        content: list[dict] = []

        if frame_b64:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": frame_b64,
                },
            })

        lines: list[str] = []

        # Session context first — this is what makes the Reasoner smarter than the Observer
        elapsed_s = world_snapshot.get("session_elapsed_s", 0)
        elapsed_str = _fmt_elapsed(elapsed_s)
        lines.append("SESSION CONTEXT:")
        lines.append(f"  elapsed: {elapsed_str}")
        lines.append(f"  activity_label: {world_snapshot.get('activity_label', 'unknown')}")
        lines.append(f"  session_narrative: {world_snapshot.get('session_narrative', '(none yet — session just started)')}")
        lines.append("")

        # Observer Beat 1
        lines.append("OBSERVER BEAT 1:")
        lines.append(f'  narration: "{obs_result.get("narration", "")}"')
        lines.append(f'  escalate: {obs_result.get("escalate", False)}')
        lines.append(f'  escalate_reason: "{obs_result.get("escalate_reason", "")}"')
        lines.append("")

        if trigger_event_types:
            lines.append(f"TRIGGER EVENTS: {', '.join(trigger_event_types)}")
        else:
            lines.append("TRIGGER EVENTS: periodic_refresh_minutely (session summary)")
        lines.append("")

        # WorldState — exclude session_narrative (already shown above) to avoid duplication
        snapshot_for_display = {
            k: v for k, v in world_snapshot.items()
            if k not in ("session_narrative", "activity_label", "session_elapsed_s")
        }
        lines.append("WORLD STATE:")
        lines.append(json.dumps(snapshot_for_display, indent=2, default=str))

        # DEVICE STATE — concise per-plug view including cooldown/lockout flags
        devices = world_snapshot.get("devices", {})
        if devices:
            lines.append("")
            lines.append("DEVICE STATE:")
            for alias, d in devices.items():
                lockout_str = ""
                if d.get("lockout_active"):
                    lockout_str = f", lockout_active=true ({d['lockout_remaining_s']}s remaining)"
                cmd_str = ""
                if d.get("last_agent_command_age_s") is not None:
                    intent_str = "on" if d.get("last_agent_command_intent") else "off"
                    cmd_str = f", last_agent_cmd={intent_str} {d['last_agent_command_age_s']}s ago"
                lines.append(
                    f"  {alias}: on={d.get('on')}, power={d.get('power_w')}W{cmd_str}{lockout_str}"
                )

        if frame_b64:
            lines.append("")
            lines.append("FRAME: Current camera frame attached above.")

        # Hard reminder at the end to prevent the JSON-before-text failure
        lines.append("")
        lines.append("IMPORTANT: Respond with ONLY valid JSON. Start with { and end with }. No text before or after.")

        content.append({"type": "text", "text": "\n".join(lines)})
        return content

    def _parse_response(
        self,
        raw_text: str,
        stop_reason: Optional[str],
    ) -> Optional[ReasonerOutput]:
        if stop_reason == "refusal":
            log.warning("Reasoner: Claude refused")
            return None
        if not raw_text:
            return None

        text = raw_text.strip()

        # Strip markdown fences
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0]

        # If Claude leaked reasoning before the JSON, try to find the JSON object
        if not text.startswith("{"):
            brace = text.find("{")
            if brace != -1:
                log.warning("Reasoner: non-JSON prefix detected, extracting JSON block")
                text = text[brace:]
            else:
                log.warning("Reasoner: no JSON object found: %.200s", raw_text)
                return None

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            log.warning("Reasoner: JSON decode failed: %.200s", raw_text)
            return None

        try:
            output = ReasonerOutput.model_validate(data)
        except Exception as exc:
            log.warning("Reasoner: Pydantic validation failed: %s", exc)
            return None

        if stop_reason == "max_tokens":
            log.warning("Reasoner: max_tokens hit — consider increasing REASONER_MAX_TOKENS")

        return output

    def _backoff_if_needed(self) -> None:
        if self._consecutive_failures > 1:
            delay = min(2 ** (self._consecutive_failures - 1), 30)
            log.info("Reasoner: backing off %.1fs after %d failures", delay, self._consecutive_failures)
            time.sleep(delay)


def _fmt_elapsed(seconds: float) -> str:
    """Format elapsed seconds as '3m 42s' or '47s'."""
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    return f"{s // 60}m {s % 60}s"


# ---------------------------------------------------------------------------
# ReasonerWorker — background thread
# ---------------------------------------------------------------------------

class ReasonerWorker:
    """Runs the Reasoner in a daemon thread. Latest work wins."""

    def __init__(self, reasoner: Reasoner) -> None:
        self._reasoner = reasoner
        self._pending: Optional[tuple] = None
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._result: Optional[ReasonerOutput] = None
        self._result_ready = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, name="reasoner-worker", daemon=True,
        )
        self._thread.start()
        log.info("ReasonerWorker thread started")

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            log.info("ReasonerWorker thread stopped")

    def push_work(
        self,
        obs_result: dict,
        event_types: list[str],
        frame: Optional[np.ndarray] = None,
    ) -> None:
        """Non-blocking. Latest work wins."""
        with self._lock:
            self._pending = (obs_result, list(event_types), frame)
        self._wake.set()

    def poll_result(self) -> Optional[ReasonerOutput]:
        """Non-blocking. Returns and clears result if ready."""
        if self._result_ready.is_set():
            self._result_ready.clear()
            result = self._result
            self._result = None
            return result
        return None

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(timeout=60.0)
            self._wake.clear()

            if self._stop.is_set():
                break

            with self._lock:
                work = self._pending
                self._pending = None

            if work is None:
                continue

            obs_result, event_types, frame = work

            try:
                result = self._reasoner.call(obs_result, event_types, frame)
                if result is not None:
                    self._result = result
                    self._result_ready.set()
            except Exception as exc:
                log.warning("ReasonerWorker: unhandled exception: %s", exc)

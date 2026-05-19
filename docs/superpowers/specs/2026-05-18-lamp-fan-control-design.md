# Lamp & Fan Control — Design Spec

**Status:** Draft (2026-05-18)
**Owner:** Reasoner (Claude Sonnet 4.6) with code-enforced guardrails in `DecisionEngine`.
**Context:** End-to-end Kasa plug control is wired (PlugManager → DecisionEngine → physical plug) but the Reasoner prompt is hard-coded to emit `lamp=null, fan=null` on every call. This spec turns the brain on.

---

## 1. Goals & non-goals

**Goals**
- The agent decides when to turn the lamp and fan on/off based on what's actually happening in the room.
- Six demo-quality scenarios fire reliably on a recorded session.
- The agent never fights the user, never flickers, never toggles a plug it can't reach.
- Every actuation is narrated in one short, human line.

**Non-goals**
- No dimming or fan-speed control (KP125M is binary on/off).
- No wall-clock or calendar awareness (no time-of-day rules in v1).
- No temperature inference (no thermal signal exists).
- No multi-room or multi-user behavior.

---

## 2. Architecture — two-tier decision split

```
Frame + WorldState + Plug state
            │
            ▼
   ┌──────────────────────┐
   │   Reasoner (Claude)  │   soft policy — judgment, narration
   │   emits lamp/fan     │
   │   ON / OFF / null    │
   └─────────┬────────────┘
             │  ReasonerOutput
             ▼
   ┌──────────────────────┐
   │   DecisionEngine     │   hard guardrails — code-enforced
   │   may refuse the     │
   │   toggle             │
   └─────────┬────────────┘
             │
             ▼
        PlugManager  →  physical plug
```

- **Reasoner owns the *interesting* reasoning.** It sees the frame (brightness), `world_snapshot` (occupancy, activity_label, session_narrative), and an injected `DEVICE STATE` block (is_on, power_w, lockout flags). It picks the action and writes the narration.
- **DecisionEngine owns the *safety net*.** It can refuse a toggle for cooldown, manual-override lockout, no-person-present, or plug-unreachable reasons. Refused toggles are logged with reason + surfaced on the dashboard so we can debug *why* nothing happened.

Rationale (already agreed in brainstorm): the Observer is amnesiac and can't reason about temporal context. The Reasoner already owns the session model. One decider = one mouth = no races.

---

## 3. Demo scenarios (v1)

Each scenario notes the **trigger** (what the agent perceives) and the **action** (what it does + a sample narration). Narrations are illustrative; Claude generates them at runtime per the existing `<narration_style>` rules (≤40 words, conversational, present tense).

### Act 1 — Anticipation
- **A1. Sit down in a dim room.** Person enters, frame is visibly dim, person sits at the desk. → **Lamp ON.** *"Got the light for you."*
- **A3. Return from a break.** Person left ≥90s ago (lamp + fan went off via C1). Person re-enters. → **Lamp ON.** *"Welcome back."* (Session narrative carries the "they stepped away, I powered down" context, so Claude knows this is a return, not a first arrival.)

### Act 2 — Reading the room
- **R1. Call starts.** Speech detected (YAMNet `Speech` class persists ≥2 windows) + person sitting + room had been quiet. → **Fan OFF.** *"Quieting the fan — sounds like you're on a call."*
- **R2. Call ends.** Speech absent for ≥30s after a sustained speech period. → **Fan ON** *only if* the fan was the agent's doing before R1 (i.e. agent-owned at the time R1 fired). *"You're off the call — fan's back on."*
- **R3. User overrides manually.** PlugManager poll observes `is_on` flipped without a matching agent command in the last ~10s. → DecisionEngine locks the device out for 5 minutes, emits a one-line narration on the next Reasoner call: *"Noted — leaving the lamp how you set it."*

### Act 3 — Cleanup
- **C1. Room is empty.** Person count drops from ≥1 → 0 and stays at 0 for **3 seconds** (debounce). A synthetic `room_empty_confirmed` event fires, routed through `REASONER_ALWAYS` so it bypasses Observer escalation. → **Lamp OFF, Fan OFF.** *"Powering down — room's all yours."* Total latency from physical departure ≈ 5–6s (3s debounce + ~2–3s Reasoner call).

---

## 4. Data contract changes

### 4.1 `ReasonerOutput` (agents/reasoner.py)

| Field | Before | After |
|---|---|---|
| `lamp` | `Optional[Literal["on","off"]]`, docs say *"always null"* | `Optional[Literal["on","off"]]`, `None` = no change |
| `fan` | same as lamp | same as lamp |
| `lamp_reason` | — | `str = ""` — short internal justification (not spoken) |
| `fan_reason` | — | `str = ""` — short internal justification (not spoken) |

Field descriptions in the Pydantic model and the JSON example in `<output_format>` must be rewritten to match.

### 4.2 `WorldState` (agents/world_state.py)

Add a `DeviceState` dataclass per plug:

```python
@dataclass
class DeviceState:
    alias: str
    is_on: bool
    power_w: float
    last_agent_command_at: float | None       # monotonic ts of last DE-issued toggle
    last_agent_command_intent: bool | None    # what the agent tried to set it to
    last_manual_override_at: float | None     # monotonic ts of last detected override
    lockout_until: float | None               # monotonic ts; if > now, no agent toggles
    polled_at: float                          # ts of underlying PlugManager poll
```

`WorldState` exposes:
- `lamp_state: DeviceState` / `fan_state: DeviceState`
- `update_from_plug_poll(alias, plug_state)` — called by the existing plug-poll thread on every PlugManager update.
- `record_agent_command(alias, intent)` — called by DecisionEngine immediately before issuing a `turn_on/turn_off`.
- `detect_manual_override(alias)` — invoked inside `update_from_plug_poll`. If the current `is_on` differs from `last_agent_command_intent` AND `last_agent_command_at` is older than 10s (or None), set `last_manual_override_at = now` and `lockout_until = now + 5min`.

### 4.3 `DecisionEngine` (agents/decisions.py)

New rules, applied in `_handle_lamp` / `_handle_fan` *before* any `plugs.turn_on/turn_off` call:

1. **No-op guard.** If `state is None`, return (already exists).
2. **Idempotency.** If `state == "on"` and `world.lamp_state.is_on` is already true → log debug, do nothing. Same for off.
3. **Cooldown.** If `world.lamp_state.last_agent_command_at` is < 60s ago → refuse, log `"cooldown"`, surface to broadcaster.
4. **Override lockout.** If `world.lamp_state.lockout_until > now` → refuse, log `"override_lockout"`, surface to broadcaster.
5. **No-person-ON guard.** If `state == "on"` and `world.entities` has zero persons in the current snapshot → refuse, log `"no_person_for_on"`, surface to broadcaster.
6. **Plug-unreachable.** If `plugs.is_available(alias)` is false → refuse, log `"plug_unreachable"`, surface to broadcaster.
7. **Issue the command.** Call `world.record_agent_command(alias, intent)` *first*, then `plugs.turn_on/turn_off`. If the PlugManager call returns False, log the failure (no retry — next Reasoner call will re-evaluate).

Refused/issued toggles publish a `decision_engine` event on `broadcaster` so the dashboard shows: device, action, accepted/refused, reason.

---

## 5. Reasoner prompt changes (agents/reasoner.py `_SYSTEM_PROMPT`)

### 5.1 Replace `<action_rules>` block

Today's block says *"Lamp and fan: return null for both. Device control is currently disabled."* Replace with policy that mirrors §3 scenarios, written as guidance (not a decision tree):

```
<action_rules>
Lamp and fan: you may command "on", "off", or null (no change). Null is the
default — only command a toggle when one of these patterns is clearly present.

LAMP
  - Turn ON when a person is in the room AND the frame looks visibly dim
    AND the lamp is currently off. Returning person after a powered-down
    absence is a strong case for ON.
  - Turn OFF when the room has been empty long enough that the Observer
    confirms it (not just track-ID churn). Match this with a Fan OFF too if
    the fan is on.
  - Do NOT command the lamp if the DEVICE STATE block shows an
    override_lockout — the user touched it; respect that. You may narrate
    the acknowledgment exactly once when you first see the override.

FAN
  - Turn OFF when speech has just started after a quiet stretch (the person
    appears to be on a call). The fan being on would be intrusive.
  - Turn ON again when the call ends (speech absent ≥30s) AND the fan was
    on before the call started.
  - Otherwise leave the fan alone in v1. Do not invent thermal reasoning —
    you have no temperature signal.

GENERAL
  - Always populate lamp_reason / fan_reason with one short sentence per
    toggle, even though it isn't spoken. This is your debug log.
  - When you command a toggle, narrate it in one short conversational line.
    The narration is the action's voice — don't actuate silently.
  - If DEVICE STATE shows a cooldown or lockout active for a device, do not
    command that device. Wait.
  - Never command ON when the WorldState shows no people present.
</action_rules>
```

### 5.2 Update `<output_format>`

Change the JSON skeleton so `lamp` and `fan` are shown as `string|null` (not always `null`) and add `lamp_reason`, `fan_reason`.

### 5.3 Examples

Add three new examples covering the v1 scenarios; keep two existing minutely-summary examples so the Reasoner doesn't forget how to do nothing.

- **EXAMPLE 5: A1** — dim frame, person sits down → lamp on, narration.
- **EXAMPLE 6: R1** — speech onset → fan off, narration.
- **EXAMPLE 7: R3** — DEVICE STATE shows override_lockout on lamp → lamp=null, narration acknowledging the override.

### 5.4 User-message additions

`_build_user_message` gains a `DEVICE STATE:` block after `WORLD STATE`:

```
DEVICE STATE:
  lamp: on=false, power=0.0W, last_agent_command=12s ago (off), override_lockout=false
  fan:  on=true,  power=18.2W, last_agent_command=4m ago (on), override_lockout=true (until 3m32s)
```

(Formatted from `world.lamp_state` / `fan_state`.)

---

## 6. Runtime wiring (main.py)

- The existing plug-poll loop (around `main.py:607`) already iterates lamp/fan aliases. Add a call to `world.update_from_plug_poll(alias, plug_state)` inside it so WorldState's DeviceState stays current.
- `DecisionEngine` constructor now takes `world: WorldState` (in addition to `plugs`, `speaker`) so it can read DeviceState and call `record_agent_command`.
- **Empty-room watcher.** A small watcher (in `main.py` or `WorldState`) tracks the person count from each WorldState snapshot. When count transitions ≥1 → 0 and stays at 0 for 3 seconds, it emits a synthetic event of type `room_empty_confirmed`. The watcher is re-armed (and any pending fire is cancelled) the instant person count goes back to ≥1, so a quick step-out/step-back-in does not fire C1.
- Add `"room_empty_confirmed"` to `config.REASONER_ALWAYS` so it bypasses Observer escalation and goes straight to the Reasoner.

---

## 7. Edge cases (call-outs)

| Case | Handling |
|---|---|
| Track-ID churn looks like "new arrival" | A3 (and any "person returned" reasoning) is gated on **Observer narration** confirming occupancy change, not raw track-ID deltas. Already a documented Observer-limitation Claude is trained on. |
| Brightness ambiguous | Prompt rule: lean conservative. If Claude isn't confident the room is dim, command null. |
| User toggles plug *while agent is mid-command* | The agent's `record_agent_command` precedes the PlugManager call. If the next poll shows `is_on` ≠ `last_agent_command_intent` *and* > 10s have passed since the command, it's an override. 10s is generous to absorb network jitter. |
| Plug becomes unreachable mid-session | DecisionEngine returns False from PlugManager; we surface to broadcaster and stop trying that plug for that toggle. Next Reasoner call re-evaluates. No retry loop. |
| Speech false positive triggers R1 incorrectly | YAMNet temporal smoothing already requires ≥2 consecutive windows. R1 acts on the smoothed signal, not raw windows. If R1 misfires anyway, the 60s cooldown limits the damage to one toggle. |
| Reasoner emits both lamp and fan ON in the same call | Both run through DecisionEngine independently with separate cooldowns. Fine. |
| Speaker is None (dev runs) | DecisionEngine already logs narrations instead of speaking. No change. |

---

## 8. Out of scope (deferred)

- Wall-clock / time-of-day rules (e.g. "don't blast fan at night").
- "Natural light arrived" scenario (C2, dropped) — requires reliable brightness comparison over time; revisit when we have a per-frame brightness scalar in WorldState.
- "Work session heats up" (A2, dropped) — no thermal signal.
- Fan speed / lamp dimming.
- Cross-device coordination (e.g. lamp dim when fan high).
- Persistent learned preferences across sessions.

---

## 9. Resolved decisions

1. **Cooldown = 60s, override lockout = 5min.** Confirmed.
2. **A3 trusts `session_narrative`** for the "returning person" framing. No explicit `last_departure_at` field. Confirmed.
3. **R2 only resumes the fan if the agent owned the fan-on state before R1.** Confirmed.
4. **R3 acknowledgment narration is generated by Claude** on the next Reasoner call after the override is detected. In-character over mechanical. Confirmed.
5. **C1 latency ≈ 5–6s** via a 3s empty-room debounce that emits a synthetic `room_empty_confirmed` event routed through `REASONER_ALWAYS`. See §3 (C1) and §6.

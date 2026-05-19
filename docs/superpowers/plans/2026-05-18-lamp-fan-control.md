# Lamp & Fan Control Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the Reasoner (Claude Sonnet 4.6) actually decide when to turn the lamp and fan on or off, with code-enforced guardrails that prevent flicker, respect manual user overrides, and avoid acting when the room is empty or a plug is unreachable.

**Architecture:** Two-tier decision split. The Reasoner produces `lamp`/`fan` commands (`"on"`/`"off"`/`null`) in its existing JSON output. `DecisionEngine` applies code-enforced guardrails (idempotency, 60s cooldown, 5-min manual-override lockout, no-person-ON guard, plug-unreachable check) before calling `PlugManager`. `WorldState.DeviceState` is extended to track agent-command history and override lockouts. A new `EmptyRoomWatcher` synthesizes a `room_empty_confirmed` event 3s after the room goes empty so the cleanup scenario (C1) fires within ~5–6s of departure.

**Tech Stack:** Python 3.11, Pydantic (existing ReasonerOutput schema), anthropic SDK (existing Claude call), python-kasa via the existing `PlugManager` wrapper. No new dependencies. Tests follow the existing `tests/smoke_*.py` convention — plain Python scripts with `assert`, runnable as `python tests/test_*.py`. No pytest dependency added.

**Spec:** `docs/superpowers/specs/2026-05-18-lamp-fan-control-design.md`. Read it before starting — every task implements a piece of that spec.

---

## File map

| Path | Change |
|---|---|
| `agents/world_state.py` | Extend `DeviceState` with agent-command + lockout fields; modify `update_devices`; add `record_agent_command(alias, intent)`; add `device_state(alias)`; expand `snapshot_for_reasoner` device block. |
| `agents/decisions.py` | Major rewrite. New constructor param `world: WorldState`. Add idempotency, cooldown, lockout, no-person-ON, plug-unreachable guards. Publish accepted/refused decisions to broadcaster. |
| `agents/reasoner.py` | Update `ReasonerOutput` schema (lamp/fan no longer "always null"; add `lamp_reason`/`fan_reason`). Rewrite `<action_rules>` and `<output_format>` blocks. Add 3 in-prompt examples (A1, R1, R3). Update `_build_user_message` to inject a `DEVICE STATE:` block. |
| `agents/empty_room_watcher.py` | **NEW.** Pure-logic 3s debounce. Caller feeds it person counts; it fires a callback when count has been 0 for ≥3s after being ≥1. Injectable `now_fn` for testability. |
| `config.py` | Add `room_empty_confirmed` to `REASONER_ALWAYS`. Add tunables: `DEVICE_COOLDOWN_S = 60.0`, `MANUAL_OVERRIDE_LOCKOUT_S = 300.0`, `EMPTY_ROOM_DEBOUNCE_S = 3.0`, `AGENT_COMMAND_GRACE_S = 10.0`. |
| `main.py` | Pass `world` to `DecisionEngine`. Construct and start `EmptyRoomWatcher`. On each tick, feed it the current person count. When it fires, push synthetic `room_empty_confirmed` work to `reasoner_worker`. |
| `tests/test_world_state_devices.py` | **NEW.** Asserts for `DeviceState` extension, `record_agent_command`, `update_devices` override detection. |
| `tests/test_decision_engine.py` | **NEW.** Asserts for every DecisionEngine guard using fakes for `PlugManager` and `Speaker`. |
| `tests/test_empty_room_watcher.py` | **NEW.** Asserts for the 3s debounce, including cancel-on-return. |
| `DESIGN.md` | Update per CLAUDE.md rule: DecisionEngine flipped from "plugs disabled" to "active two-tier policy"; data flow includes new DEVICE STATE block to Reasoner; new EmptyRoomWatcher module. |
| `LEARNING.md` | One new entry: two-tier decision split, why we put guardrails in code rather than the prompt, and why the override-lockout is 5 min. |

---

## Conventions

- **Tests** are plain Python scripts following the existing `tests/smoke_*.py` pattern. Each file has a `main()` that runs `assert`-based checks and prints `OK <test name>` per case. Run with `python tests/test_<name>.py` (venv activated). Time-dependent code accepts an injectable `now_fn` so tests don't sleep.
- **Commits** are small and labeled `feat:`, `refactor:`, `test:`, `docs:` to match recent log style (`fine-tune reasoner prompt and update UI` etc. — the repo is not strict about conventional commits; the labels are just for clarity here).
- **Imports.** Stick to absolute imports (`from agents.world_state import WorldState`) — matches existing files.
- **Logging.** Use the existing `log = logging.getLogger(__name__)` pattern; never `print()` outside of `main.py`'s status lines.
- **Don't touch `actuators/smart_plug.py`.** It's an unused stub — DecisionEngine talks to `PlugManager` directly. Removing the stub is out of scope.

---

# Tasks

## Task 1 — Extend `DeviceState` and add agent-command tracking to `WorldState`

**Files:**
- Modify: `agents/world_state.py:76-79` (DeviceState dataclass)
- Modify: `agents/world_state.py:247-257` (update_devices method)
- Modify: `agents/world_state.py` snapshot_for_reasoner (around line 343) — expand device block
- Create: `tests/test_world_state_devices.py`

### What & why

The existing `DeviceState` only stores `alias`, `on`, `power_w`. We need to track when the agent last commanded the device (so we can detect manual overrides) and how long the device is locked out. We also extend `update_devices` so every plug poll runs override detection automatically — no separate code path. The Reasoner reads these via `snapshot_for_reasoner` and the new `DEVICE STATE:` prompt block (Task 3).

`WorldState` is thread-safe via `self._lock`; preserve that — every read/write touches `_devices` under the lock.

### Steps

- [ ] **Step 1: Write the failing test**

Create `tests/test_world_state_devices.py`:

```python
"""Unit tests for WorldState device-state extensions (lamp/fan control)."""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from unittest.mock import MagicMock

import config
from agents.world_state import WorldState, DeviceState


def _fake_plug_state(is_on, power_w=1.0):
    m = MagicMock()
    m.is_on = is_on
    m.power_w = power_w
    return m


def _fake_plugs(lamp_on, fan_on):
    m = MagicMock()
    def state(alias):
        if alias == config.LAMP_PLUG_ALIAS:
            return _fake_plug_state(lamp_on)
        if alias == config.FAN_PLUG_ALIAS:
            return _fake_plug_state(fan_on)
        return None
    m.state.side_effect = state
    return m


def test_device_state_defaults_have_new_fields():
    ds = DeviceState(alias="light")
    assert ds.last_agent_command_at is None
    assert ds.last_agent_command_intent is None
    assert ds.last_manual_override_at is None
    assert ds.lockout_until is None
    print("OK test_device_state_defaults_have_new_fields")


def test_record_agent_command_sets_intent_and_ts():
    w = WorldState()
    now = time.monotonic()
    w.record_agent_command(config.LAMP_PLUG_ALIAS, intent=True, now=now)
    ds = w.device_state(config.LAMP_PLUG_ALIAS)
    assert ds is not None
    assert ds.last_agent_command_intent is True
    assert ds.last_agent_command_at == now
    print("OK test_record_agent_command_sets_intent_and_ts")


def test_update_devices_no_override_when_state_matches_intent():
    w = WorldState()
    t0 = 1000.0
    w.record_agent_command(config.LAMP_PLUG_ALIAS, intent=True, now=t0)
    # Plug reports ON, matching intent → no override
    w.update_devices(_fake_plugs(lamp_on=True, fan_on=False), now=t0 + 1.0)
    ds = w.device_state(config.LAMP_PLUG_ALIAS)
    assert ds.last_manual_override_at is None
    assert ds.lockout_until is None
    print("OK test_update_devices_no_override_when_state_matches_intent")


def test_update_devices_no_override_within_grace_window():
    w = WorldState()
    t0 = 1000.0
    w.record_agent_command(config.LAMP_PLUG_ALIAS, intent=True, now=t0)
    # Plug still reports OFF only 2s later — within AGENT_COMMAND_GRACE_S, not an override
    w.update_devices(_fake_plugs(lamp_on=False, fan_on=False), now=t0 + 2.0)
    ds = w.device_state(config.LAMP_PLUG_ALIAS)
    assert ds.last_manual_override_at is None
    assert ds.lockout_until is None
    print("OK test_update_devices_no_override_within_grace_window")


def test_update_devices_detects_override_after_grace():
    w = WorldState()
    t0 = 1000.0
    w.record_agent_command(config.LAMP_PLUG_ALIAS, intent=True, now=t0)
    # 15s later, plug still OFF despite agent commanding ON → manual override
    w.update_devices(_fake_plugs(lamp_on=False, fan_on=False), now=t0 + 15.0)
    ds = w.device_state(config.LAMP_PLUG_ALIAS)
    assert ds.last_manual_override_at == t0 + 15.0
    assert ds.lockout_until == t0 + 15.0 + config.MANUAL_OVERRIDE_LOCKOUT_S
    print("OK test_update_devices_detects_override_after_grace")


def test_update_devices_detects_override_when_no_prior_command():
    """User toggles the plug with the agent never having commanded it."""
    w = WorldState()
    # First poll: lamp is off. Just baseline, no override.
    w.update_devices(_fake_plugs(lamp_on=False, fan_on=False), now=1000.0)
    assert w.device_state(config.LAMP_PLUG_ALIAS).lockout_until is None
    # Second poll: lamp is on. No agent command ever issued → override.
    w.update_devices(_fake_plugs(lamp_on=True, fan_on=False), now=1001.0)
    ds = w.device_state(config.LAMP_PLUG_ALIAS)
    assert ds.last_manual_override_at == 1001.0
    print("OK test_update_devices_detects_override_when_no_prior_command")


def test_snapshot_for_reasoner_includes_lockout_info():
    w = WorldState()
    t0 = 1000.0
    w.record_agent_command(config.LAMP_PLUG_ALIAS, intent=True, now=t0)
    w.update_devices(_fake_plugs(lamp_on=False, fan_on=False), now=t0 + 15.0)
    snap = w.snapshot_for_reasoner()
    devices = snap.get("devices", {})
    lamp = devices.get(config.LAMP_PLUG_ALIAS)
    assert lamp is not None
    assert lamp["on"] is False
    assert lamp["lockout_active"] is True
    print("OK test_snapshot_for_reasoner_includes_lockout_info")


def main():
    test_device_state_defaults_have_new_fields()
    test_record_agent_command_sets_intent_and_ts()
    test_update_devices_no_override_when_state_matches_intent()
    test_update_devices_no_override_within_grace_window()
    test_update_devices_detects_override_after_grace()
    test_update_devices_detects_override_when_no_prior_command()
    test_snapshot_for_reasoner_includes_lockout_info()
    print("\nAll WorldState device tests passed.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the test, confirm it fails**

```bash
source venv/bin/activate
python tests/test_world_state_devices.py
```

Expected: `AttributeError` (no `last_agent_command_at` on `DeviceState`) or `AttributeError` for `record_agent_command`.

- [ ] **Step 3: Add the tunables to `config.py`**

Append near the existing plug/reasoner config (after the `REASONER_*` block, before `TTS_VOICE`):

```python
# -- Lamp/fan control (DecisionEngine + override detection) --
DEVICE_COOLDOWN_S = 60.0           # min seconds between agent toggles on same device
MANUAL_OVERRIDE_LOCKOUT_S = 300.0  # how long to leave a manually-overridden device alone
AGENT_COMMAND_GRACE_S = 10.0       # window after agent command before mismatched state counts as override
EMPTY_ROOM_DEBOUNCE_S = 3.0        # seconds the room must stay empty before C1 fires
```

And add to the `REASONER_ALWAYS` set (modify the existing literal):

```python
REASONER_ALWAYS: set[str] = {
    "new_person",
    "lost_person",
    "power_anomaly",
    "security_event",
    "periodic_refresh_minutely",
    "periodic_refresh_hourly",
    "room_empty_confirmed",   # synthetic event from EmptyRoomWatcher
}
```

- [ ] **Step 4: Extend `DeviceState` in `agents/world_state.py`**

Replace the existing dataclass:

```python
@dataclass
class DeviceState:
    """One smart plug's state, including agent-command history and lockout."""
    alias: str = ""
    on: bool = False
    power_w: float = 0.0

    # Agent-command history (used to detect manual overrides)
    last_agent_command_at: float | None = None       # monotonic ts of last DE-issued toggle
    last_agent_command_intent: bool | None = None    # what the agent tried to set it to

    # Manual override / lockout tracking
    last_manual_override_at: float | None = None     # monotonic ts of last detected override
    lockout_until: float | None = None               # monotonic ts; if > now, no agent toggles
```

- [ ] **Step 5: Add `record_agent_command` and `device_state` accessors**

Add these methods to `WorldState` (insert near `update_devices`, around line 247):

```python
def record_agent_command(self, alias: str, intent: bool, now: float | None = None) -> None:
    """Called by DecisionEngine immediately before issuing a turn_on/off.

    Stores the agent's intent and timestamp so update_devices() can later
    distinguish 'state still settling' from 'user just overrode us'.
    """
    if now is None:
        now = time.monotonic()
    with self._lock:
        ds = self._devices.get(alias)
        if ds is None:
            ds = DeviceState(alias=alias)
            self._devices[alias] = ds
        ds.last_agent_command_at = now
        ds.last_agent_command_intent = intent

def device_state(self, alias: str) -> DeviceState | None:
    """Return a copy of the DeviceState for this plug, or None if unknown."""
    with self._lock:
        ds = self._devices.get(alias)
        if ds is None:
            return None
        # Return a copy so callers can read without holding the lock
        return DeviceState(
            alias=ds.alias, on=ds.on, power_w=ds.power_w,
            last_agent_command_at=ds.last_agent_command_at,
            last_agent_command_intent=ds.last_agent_command_intent,
            last_manual_override_at=ds.last_manual_override_at,
            lockout_until=ds.lockout_until,
        )

def people_count(self) -> int:
    """Current number of tracked persons (thread-safe accessor)."""
    with self._lock:
        return self._people_count
```

- [ ] **Step 6: Modify `update_devices` to detect overrides**

Replace the existing method (around line 247-257):

```python
def update_devices(self, plugs: PlugManager, now: float | None = None) -> None:
    """Copy latest plug states into the world model, detecting manual overrides.

    Override detection: if the actual is_on differs from our last agent-command
    intent AND the command was issued more than AGENT_COMMAND_GRACE_S ago (or
    we have no record of commanding it at all), the user touched it.
    Set last_manual_override_at and lockout_until.
    """
    if now is None:
        now = time.monotonic()
    with self._lock:
        for alias in (config.LAMP_PLUG_ALIAS, config.FAN_PLUG_ALIAS):
            st = plugs.state(alias)
            if st is None:
                continue
            ds = self._devices.get(alias)
            if ds is None:
                ds = DeviceState(alias=alias)
                self._devices[alias] = ds

            new_on = bool(st.is_on)
            # Override detection — only when:
            #   (a) the new state contradicts our last intent, AND
            #   (b) we're outside the grace window OR we never commanded the device
            intent = ds.last_agent_command_intent
            cmd_at = ds.last_agent_command_at
            outside_grace = cmd_at is None or (now - cmd_at) > config.AGENT_COMMAND_GRACE_S
            state_contradicts_intent = (intent is not None) and (new_on != intent)
            state_changed_unprompted = (intent is None) and (new_on != ds.on)

            if outside_grace and (state_contradicts_intent or state_changed_unprompted):
                ds.last_manual_override_at = now
                ds.lockout_until = now + config.MANUAL_OVERRIDE_LOCKOUT_S

            ds.on = new_on
            ds.power_w = float(st.power_w)
```

- [ ] **Step 7: Expand the device block in `snapshot_for_reasoner`**

Find the existing device serialization (around line 388):

```python
"devices": {
    alias: {"on": d.on, "power_w": round(d.power_w, 1)}
    for alias, d in self._devices.items()
},
```

Replace with a helper-driven version. Add this helper method to `WorldState`:

```python
def _device_for_reasoner(self, d: DeviceState, now: float) -> dict:
    lockout_remaining = 0.0
    if d.lockout_until is not None and d.lockout_until > now:
        lockout_remaining = d.lockout_until - now
    last_cmd_age = None
    if d.last_agent_command_at is not None:
        last_cmd_age = round(now - d.last_agent_command_at, 1)
    return {
        "on": d.on,
        "power_w": round(d.power_w, 1),
        "lockout_active": lockout_remaining > 0,
        "lockout_remaining_s": round(lockout_remaining, 1),
        "last_agent_command_age_s": last_cmd_age,
        "last_agent_command_intent": d.last_agent_command_intent,
    }
```

And replace the snapshot dict's `"devices"` line with:

```python
"devices": {
    alias: self._device_for_reasoner(d, time.monotonic())
    for alias, d in self._devices.items()
},
```

- [ ] **Step 8: Run the test, confirm it passes**

```bash
python tests/test_world_state_devices.py
```

Expected: 7 OK lines + `All WorldState device tests passed.`

- [ ] **Step 9: Commit**

```bash
git add config.py agents/world_state.py tests/test_world_state_devices.py
git commit -m "feat: extend DeviceState with agent-command history and manual-override detection"
```

---

## Task 2 — Rewrite `DecisionEngine` with guardrails

**Files:**
- Modify: `agents/decisions.py` (full rewrite of the class body)
- Create: `tests/test_decision_engine.py`

### What & why

`DecisionEngine` currently just forwards `output.lamp`/`output.fan` straight to PlugManager. It needs five guards (idempotency, cooldown, lockout, no-person-ON, plug-unreachable), the constructor needs to take `world: WorldState`, and every accepted/refused decision needs to publish to `broadcaster` so the dashboard can show *why* nothing happened.

The decision flow per device:
1. If Reasoner output is `None` → skip silently.
2. If `world.device_state(alias).on` already matches the requested state → idempotent skip, log debug, publish `accepted=False, reason="idempotent"`.
3. If `last_agent_command_at` is within `DEVICE_COOLDOWN_S` → refuse, reason `"cooldown"`.
4. If `lockout_until > now` → refuse, reason `"override_lockout"`.
5. If `state == "on"` and `world` shows zero people → refuse, reason `"no_person_for_on"`.
6. If `plugs.is_available(alias)` is False → refuse, reason `"plug_unreachable"`.
7. Otherwise: `world.record_agent_command(alias, intent)` → `plugs.turn_on/off(alias)` → publish `accepted=True, reason="ok"`. If PlugManager returns False, publish `accepted=False, reason="plug_call_failed"` and log.

### Steps

- [ ] **Step 1: Write the failing test**

Create `tests/test_decision_engine.py`:

```python
"""Unit tests for DecisionEngine guardrails (lamp/fan control)."""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from unittest.mock import MagicMock

import config
from agents.world_state import WorldState
from agents.reasoner import ReasonerOutput
from agents.decisions import DecisionEngine


def _fake_plugs(lamp_available=True, fan_available=True, lamp_on=False, fan_on=False, command_ok=True):
    m = MagicMock()
    def state(alias):
        s = MagicMock()
        if alias == config.LAMP_PLUG_ALIAS:
            s.is_on = lamp_on
        elif alias == config.FAN_PLUG_ALIAS:
            s.is_on = fan_on
        s.power_w = 0.0
        return s
    m.state.side_effect = state
    m.is_available.side_effect = lambda a: {
        config.LAMP_PLUG_ALIAS: lamp_available,
        config.FAN_PLUG_ALIAS: fan_available,
    }.get(a, False)
    m.turn_on.return_value = command_ok
    m.turn_off.return_value = command_ok
    return m


def _world_with_person(count=1):
    """Create a WorldState reporting `count` people. Avoid touching YOLO."""
    w = WorldState()
    # snapshot_for_reasoner reads _people_count; the easiest way to set it is
    # directly under the lock since YOLO update is too heavy for a unit test.
    with w._lock:
        w._people_count = count
    return w


def _output(lamp=None, fan=None):
    return ReasonerOutput(
        narration="", lamp=lamp, fan=fan, alert=False, speak=True, reasoning="",
        activity_label="unknown", session_narrative="",
    )


def test_null_lamp_skips_silently():
    plugs = _fake_plugs()
    world = _world_with_person(1)
    de = DecisionEngine(plugs=plugs, speaker=None, world=world)
    de.execute(_output(lamp=None, fan=None))
    plugs.turn_on.assert_not_called()
    plugs.turn_off.assert_not_called()
    print("OK test_null_lamp_skips_silently")


def test_lamp_on_when_off_and_person_present():
    plugs = _fake_plugs(lamp_on=False)
    world = _world_with_person(1)
    de = DecisionEngine(plugs=plugs, speaker=None, world=world)
    de.execute(_output(lamp="on"))
    plugs.turn_on.assert_called_once_with(config.LAMP_PLUG_ALIAS)
    # WorldState recorded the intent
    assert world.device_state(config.LAMP_PLUG_ALIAS).last_agent_command_intent is True
    print("OK test_lamp_on_when_off_and_person_present")


def test_lamp_on_idempotent_when_already_on():
    plugs = _fake_plugs(lamp_on=True)
    world = _world_with_person(1)
    # Sync WorldState to reflect lamp already on
    world.update_devices(plugs)
    de = DecisionEngine(plugs=plugs, speaker=None, world=world)
    de.execute(_output(lamp="on"))
    plugs.turn_on.assert_not_called()
    print("OK test_lamp_on_idempotent_when_already_on")


def test_lamp_on_refused_during_cooldown():
    plugs = _fake_plugs(lamp_on=False)
    world = _world_with_person(1)
    # Pretend the agent just turned the lamp off 5s ago
    world.record_agent_command(config.LAMP_PLUG_ALIAS, intent=False, now=time.monotonic() - 5.0)
    de = DecisionEngine(plugs=plugs, speaker=None, world=world)
    de.execute(_output(lamp="on"))
    plugs.turn_on.assert_not_called()
    print("OK test_lamp_on_refused_during_cooldown")


def test_lamp_on_allowed_after_cooldown():
    plugs = _fake_plugs(lamp_on=False)
    world = _world_with_person(1)
    world.record_agent_command(
        config.LAMP_PLUG_ALIAS, intent=False,
        now=time.monotonic() - (config.DEVICE_COOLDOWN_S + 1.0),
    )
    de = DecisionEngine(plugs=plugs, speaker=None, world=world)
    de.execute(_output(lamp="on"))
    plugs.turn_on.assert_called_once()
    print("OK test_lamp_on_allowed_after_cooldown")


def test_lamp_refused_during_override_lockout():
    plugs = _fake_plugs(lamp_on=False)
    world = _world_with_person(1)
    # Simulate: agent commanded ON 30s ago, but plug actually OFF — override detected
    t = time.monotonic()
    world.record_agent_command(config.LAMP_PLUG_ALIAS, intent=True, now=t - 30.0)
    world.update_devices(plugs, now=t)  # mismatch + outside grace → override
    de = DecisionEngine(plugs=plugs, speaker=None, world=world)
    de.execute(_output(lamp="on"))
    plugs.turn_on.assert_not_called()
    print("OK test_lamp_refused_during_override_lockout")


def test_lamp_on_refused_when_no_person():
    plugs = _fake_plugs(lamp_on=False)
    world = _world_with_person(0)
    de = DecisionEngine(plugs=plugs, speaker=None, world=world)
    de.execute(_output(lamp="on"))
    plugs.turn_on.assert_not_called()
    print("OK test_lamp_on_refused_when_no_person")


def test_lamp_off_allowed_when_no_person():
    """Turning OFF doesn't need a person present — that's the cleanup case."""
    plugs = _fake_plugs(lamp_on=True)
    world = _world_with_person(0)
    world.update_devices(plugs)
    de = DecisionEngine(plugs=plugs, speaker=None, world=world)
    de.execute(_output(lamp="off"))
    plugs.turn_off.assert_called_once_with(config.LAMP_PLUG_ALIAS)
    print("OK test_lamp_off_allowed_when_no_person")


def test_lamp_refused_when_plug_unavailable():
    plugs = _fake_plugs(lamp_available=False, lamp_on=False)
    world = _world_with_person(1)
    de = DecisionEngine(plugs=plugs, speaker=None, world=world)
    de.execute(_output(lamp="on"))
    plugs.turn_on.assert_not_called()
    print("OK test_lamp_refused_when_plug_unavailable")


def test_lamp_and_fan_independent():
    plugs = _fake_plugs(lamp_on=False, fan_on=True)
    world = _world_with_person(1)
    world.update_devices(plugs)
    de = DecisionEngine(plugs=plugs, speaker=None, world=world)
    de.execute(_output(lamp="on", fan="off"))
    plugs.turn_on.assert_called_once_with(config.LAMP_PLUG_ALIAS)
    plugs.turn_off.assert_called_once_with(config.FAN_PLUG_ALIAS)
    print("OK test_lamp_and_fan_independent")


def main():
    test_null_lamp_skips_silently()
    test_lamp_on_when_off_and_person_present()
    test_lamp_on_idempotent_when_already_on()
    test_lamp_on_refused_during_cooldown()
    test_lamp_on_allowed_after_cooldown()
    test_lamp_refused_during_override_lockout()
    test_lamp_on_refused_when_no_person()
    test_lamp_off_allowed_when_no_person()
    test_lamp_refused_when_plug_unavailable()
    test_lamp_and_fan_independent()
    print("\nAll DecisionEngine tests passed.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run, confirm failure**

```bash
python tests/test_decision_engine.py
```

Expected: `TypeError: __init__() got an unexpected keyword argument 'world'` (or similar).

- [ ] **Step 3: Rewrite `agents/decisions.py`**

Replace the entire file:

```python
"""
Action decision engine — Beat 2 actuation with code-enforced guardrails.

Takes a ReasonerOutput and dispatches concrete actions: control smart plugs
(lamp/fan via PlugManager), enqueue TTS narration (via Speaker), and log
security alerts to the dashboard broadcaster.

The Reasoner produces the *judgment* ("turn the lamp on"). DecisionEngine
applies *guardrails* before any plug call:
  1. Idempotency — don't re-toggle a device already in the requested state.
  2. Cooldown   — at most one agent toggle per device per DEVICE_COOLDOWN_S.
  3. Override lockout — if the user touched the plug manually, leave it
     alone for MANUAL_OVERRIDE_LOCKOUT_S.
  4. No-person ON guard — never command ON when WorldState shows 0 people.
  5. Plug-unreachable — refuse if PlugManager.is_available() is False.

Every accepted or refused decision is published to the broadcaster so the
dashboard can show what happened and why.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Optional

import config
from agents.reasoner import ReasonerOutput
from agents.world_state import WorldState
from server.broadcaster import broadcaster

if TYPE_CHECKING:
    from actuators.speaker import Speaker
    from perception.plugs import PlugManager

log = logging.getLogger(__name__)


class DecisionEngine:
    """Translates ReasonerOutput into physical actions with guardrails."""

    def __init__(
        self,
        plugs: Optional["PlugManager"],
        speaker: Optional["Speaker"],
        world: WorldState,
    ) -> None:
        self._plugs = plugs
        self._speaker = speaker
        self._world = world

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(self, output: ReasonerOutput) -> None:
        """Run all actions from a ReasonerOutput in fixed order:
        1. Lamp control, 2. Fan control, 3. Alert, 4. Beat 2 TTS.
        """
        self._handle_device(config.LAMP_PLUG_ALIAS, output.lamp, output.lamp_reason)
        self._handle_device(config.FAN_PLUG_ALIAS, output.fan, output.fan_reason)
        self._handle_alert(output.alert, output.reasoning)
        self._handle_narration(output.narration, output.speak)

    # ------------------------------------------------------------------
    # Device control with guardrails
    # ------------------------------------------------------------------

    def _handle_device(self, alias: str, state: Optional[str], reason: str) -> None:
        if state is None or self._plugs is None:
            return
        intent = (state == "on")
        ok, refusal = self._evaluate_guards(alias, intent)
        if not ok:
            self._publish(alias, intent, accepted=False, reason=refusal, agent_reason=reason)
            log.info("DecisionEngine: %s → %s REFUSED (%s)", alias, state.upper(), refusal)
            return

        # Record intent BEFORE issuing the command so a fast subsequent poll
        # sees the in-flight command and doesn't mistake settling-time for an override.
        self._world.record_agent_command(alias, intent=intent)
        try:
            sent_ok = self._plugs.turn_on(alias) if intent else self._plugs.turn_off(alias)
        except Exception as exc:
            log.error("DecisionEngine: plug call %s=%s raised: %s", alias, state, exc)
            self._publish(alias, intent, accepted=False, reason="plug_call_failed",
                          agent_reason=reason)
            return

        if not sent_ok:
            self._publish(alias, intent, accepted=False, reason="plug_call_failed",
                          agent_reason=reason)
            log.warning("DecisionEngine: %s → %s plug call returned False", alias, state.upper())
            return

        log.info("DecisionEngine: %s → %s (reason: %s)", alias, state.upper(), reason or "—")
        self._publish(alias, intent, accepted=True, reason="ok", agent_reason=reason)

    def _evaluate_guards(self, alias: str, intent: bool) -> tuple[bool, str]:
        """Return (accepted, refusal_reason). refusal_reason is "" when accepted."""
        ds = self._world.device_state(alias)
        now = time.monotonic()

        # 1. Idempotency
        if ds is not None and ds.on == intent:
            return False, "idempotent"

        # 2. Cooldown
        if ds is not None and ds.last_agent_command_at is not None:
            age = now - ds.last_agent_command_at
            if age < config.DEVICE_COOLDOWN_S:
                return False, f"cooldown ({config.DEVICE_COOLDOWN_S - age:.1f}s left)"

        # 3. Override lockout
        if ds is not None and ds.lockout_until is not None and ds.lockout_until > now:
            return False, f"override_lockout ({ds.lockout_until - now:.1f}s left)"

        # 4. No-person ON guard
        if intent and self._world.people_count() <= 0:
            return False, "no_person_for_on"

        # 5. Plug-unreachable
        if not self._plugs.is_available(alias):
            return False, "plug_unreachable"

        return True, ""

    # ------------------------------------------------------------------
    # Alert + narration (unchanged behavior, modernized)
    # ------------------------------------------------------------------

    def _handle_alert(self, alert: bool, reasoning: str) -> None:
        if not alert:
            return
        log.warning("SECURITY ALERT — %s", reasoning[:300])
        broadcaster.publish_narration("reasoner_alert", {
            "alert": True,
            "reasoning": reasoning,
            "ts": time.time(),
        })

    def _handle_narration(self, narration: str, speak: bool) -> None:
        if not narration or not speak:
            return
        if self._speaker is not None:
            self._speaker.enqueue_beat2(narration)
        else:
            log.info("DecisionEngine: Beat 2 (no TTS): %s", narration)

    # ------------------------------------------------------------------
    # Broadcaster
    # ------------------------------------------------------------------

    def _publish(
        self,
        alias: str,
        intent: bool,
        *,
        accepted: bool,
        reason: str,
        agent_reason: str = "",
    ) -> None:
        broadcaster.publish_narration("decision_engine", {
            "alias": alias,
            "intent": "on" if intent else "off",
            "accepted": accepted,
            "refusal_reason": "" if accepted else reason,
            "agent_reason": agent_reason,
            "ts": time.time(),
        })
```

- [ ] **Step 4: Add `lamp_reason` / `fan_reason` to `ReasonerOutput` (minimal change, full prompt rewrite is Task 3)**

In `agents/reasoner.py`, in the `ReasonerOutput` class, add right after the `fan` field:

```python
    lamp_reason: str = Field(default="", description="Short internal justification for the lamp action. Not spoken. Empty when lamp is null.")
    fan_reason: str = Field(default="", description="Short internal justification for the fan action. Not spoken. Empty when fan is null.")
```

(Full prompt rewrite happens in Task 3 — for now this just keeps the schema in sync so DecisionEngine can read the fields.)

- [ ] **Step 5: Run the test, confirm it passes**

```bash
python tests/test_decision_engine.py
```

Expected: 10 OK lines + `All DecisionEngine tests passed.`

- [ ] **Step 6: Commit**

```bash
git add agents/decisions.py agents/reasoner.py tests/test_decision_engine.py
git commit -m "feat: DecisionEngine guardrails (idempotency, cooldown, lockout, no-person, unreachable)"
```

---

## Task 3 — Rewrite the Reasoner prompt to authorize lamp/fan commands

**Files:**
- Modify: `agents/reasoner.py` `_SYSTEM_PROMPT` (the `<action_rules>` block, `<output_format>` block, and `<examples>` block)
- Modify: `agents/reasoner.py` `_build_user_message` (add `DEVICE STATE:` block)

### What & why

The system prompt currently says `"Lamp and fan: return null for both. Device control is currently disabled."` Every in-prompt example also emits `null`. This task replaces those with the v1 policy from spec §5 and adds three new examples (A1, R1, R3). It also injects a `DEVICE STATE:` block into the user message so Claude can see lockout/cooldown state and choose not to bother trying.

This task is prompt-only — there's no clean unit test for prompt content (LLM behavior tests need real API calls and are flaky). Validation happens through Task 6's end-to-end smoke run.

### Steps

- [ ] **Step 1: Update the schema field descriptions for `lamp` and `fan`**

In `agents/reasoner.py`, replace the `lamp` and `fan` Field definitions:

```python
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
```

- [ ] **Step 2: Replace the `<action_rules>` block in `_SYSTEM_PROMPT`**

Find the current block (starts with `<action_rules>`, contains `"Lamp and fan: return null for both. Device control is currently disabled."`). Replace the entire `<action_rules>...</action_rules>` block with:

```
<action_rules>
Lamp and fan: you may command "on", "off", or null (no change). Null is the
default — only command a toggle when one of these patterns is clearly present.
DecisionEngine enforces hard guardrails (cooldown, override lockout, no-person
guard, plug reachability) and will refuse toggles that violate them, but you
should not knowingly emit a refused command either — check DEVICE STATE first.

LAMP
  - Turn ON when a person is in the room AND the frame looks visibly dim AND
    the lamp is currently off. A person who has just returned from an absence
    that triggered a cleanup powerdown is a strong case for ON.
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
```

- [ ] **Step 3: Replace the `<output_format>` block**

Find the existing `<output_format>...</output_format>` block and replace with:

```
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
```

- [ ] **Step 4: Replace example outputs and add A1 / R1 / R3 examples**

For EVERY existing example in the `<examples>` block, replace `"lamp": null, "fan": null` with `"lamp": null, "fan": null, "lamp_reason": "", "fan_reason": ""` (no policy reason; nothing was commanded).

Then append three new examples at the end of the `<examples>` block, before the closing `</examples>`:

```
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
```

- [ ] **Step 5: Inject DEVICE STATE block into `_build_user_message`**

Find `_build_user_message` in `agents/reasoner.py`. After the `WORLD STATE:` lines (and before the `IMPORTANT: Respond with ONLY valid JSON…` reminder), add:

```python
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
```

(Insert this block immediately before the final `lines.append("")` + `lines.append("IMPORTANT: Respond with ONLY valid JSON...")` calls. Order matters: DEVICE STATE goes after WORLD STATE, before the JSON reminder.)

- [ ] **Step 6: Smoke check — instantiate Reasoner and inspect the built prompt**

A quick offline check that the user-message builder doesn't crash and produces the new block:

```bash
python -c "
from agents.world_state import WorldState
from agents.reasoner import Reasoner
from unittest.mock import MagicMock
w = WorldState()
w.record_agent_command('light', intent=True)
r = Reasoner(w, MagicMock())
content = r._build_user_message(
    {'narration': 'x', 'escalate': False, 'escalate_reason': ''},
    ['new_person'],
    w.snapshot_for_reasoner(),
    None,
)
text = content[-1]['text']
print(text)
assert 'DEVICE STATE:' in text, 'DEVICE STATE block missing'
print('--- OK ---')
"
```

Expected: prints the assembled prompt, ends with `--- OK ---`.

- [ ] **Step 7: Commit**

```bash
git add agents/reasoner.py
git commit -m "feat: authorize Reasoner lamp/fan commands; rewrite action_rules + add A1/R1/R3 examples"
```

---

## Task 4 — `EmptyRoomWatcher` with 3-second debounce

**Files:**
- Create: `agents/empty_room_watcher.py`
- Create: `tests/test_empty_room_watcher.py`

### What & why

C1 needs to fire ~5–6s after the room actually empties. The Observer's `lost_person` events are noisy (track-ID churn). We add a small synchronous helper that the main loop pokes with the current person count on every tick. When the count transitions from ≥1 → 0 and stays at 0 for `EMPTY_ROOM_DEBOUNCE_S`, the watcher fires a callback exactly once. If the count goes back to ≥1 before the timer fires, the watcher resets and never fires for that transition.

This is pure logic — fully unit-testable with an injectable `now_fn`.

### Steps

- [ ] **Step 1: Write the failing test**

Create `tests/test_empty_room_watcher.py`:

```python
"""Unit tests for EmptyRoomWatcher."""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from agents.empty_room_watcher import EmptyRoomWatcher


class _Clock:
    def __init__(self, t=1000.0): self.t = t
    def __call__(self): return self.t
    def advance(self, dt): self.t += dt


def _record_calls():
    calls = []
    def cb(): calls.append(True)
    return cb, calls


def test_does_not_fire_when_room_never_emptied():
    clk = _Clock()
    cb, calls = _record_calls()
    w = EmptyRoomWatcher(cb, debounce_s=3.0, now_fn=clk)
    for _ in range(10):
        w.update(person_count=1)
        clk.advance(1.0)
    assert calls == []
    print("OK test_does_not_fire_when_room_never_emptied")


def test_fires_after_debounce_when_room_goes_empty():
    clk = _Clock()
    cb, calls = _record_calls()
    w = EmptyRoomWatcher(cb, debounce_s=3.0, now_fn=clk)
    w.update(person_count=1)
    clk.advance(1.0)
    w.update(person_count=0)   # transition — start timer
    clk.advance(2.9)
    w.update(person_count=0)
    assert calls == [], "should not fire before debounce elapses"
    clk.advance(0.2)            # now 3.1s after the transition
    w.update(person_count=0)
    assert calls == [True]
    print("OK test_fires_after_debounce_when_room_goes_empty")


def test_does_not_refire_while_still_empty():
    clk = _Clock()
    cb, calls = _record_calls()
    w = EmptyRoomWatcher(cb, debounce_s=3.0, now_fn=clk)
    w.update(person_count=1); clk.advance(1.0)
    w.update(person_count=0); clk.advance(4.0)
    w.update(person_count=0)
    assert calls == [True]
    clk.advance(10.0)
    w.update(person_count=0)
    assert calls == [True], "should fire once per emptying, not continuously"
    print("OK test_does_not_refire_while_still_empty")


def test_resets_if_person_returns_before_debounce():
    clk = _Clock()
    cb, calls = _record_calls()
    w = EmptyRoomWatcher(cb, debounce_s=3.0, now_fn=clk)
    w.update(person_count=1); clk.advance(1.0)
    w.update(person_count=0); clk.advance(1.5)
    w.update(person_count=1)   # came back — cancel
    clk.advance(5.0)
    w.update(person_count=1)
    assert calls == [], "transient emptiness should not fire"
    print("OK test_resets_if_person_returns_before_debounce")


def test_fires_again_after_second_emptying():
    clk = _Clock()
    cb, calls = _record_calls()
    w = EmptyRoomWatcher(cb, debounce_s=3.0, now_fn=clk)
    w.update(person_count=1); clk.advance(1.0)
    w.update(person_count=0); clk.advance(4.0)
    w.update(person_count=0)            # fire #1
    w.update(person_count=1); clk.advance(2.0)
    w.update(person_count=0); clk.advance(4.0)
    w.update(person_count=0)            # fire #2
    assert calls == [True, True]
    print("OK test_fires_again_after_second_emptying")


def main():
    test_does_not_fire_when_room_never_emptied()
    test_fires_after_debounce_when_room_goes_empty()
    test_does_not_refire_while_still_empty()
    test_resets_if_person_returns_before_debounce()
    test_fires_again_after_second_emptying()
    print("\nAll EmptyRoomWatcher tests passed.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run, confirm failure**

```bash
python tests/test_empty_room_watcher.py
```

Expected: `ModuleNotFoundError: No module named 'agents.empty_room_watcher'`.

- [ ] **Step 3: Implement the watcher**

Create `agents/empty_room_watcher.py`:

```python
"""
Empty-room watcher — fires a synthetic event when the room has been empty
for EMPTY_ROOM_DEBOUNCE_S.

Driven by the main loop, which calls update(person_count) every tick.
The watcher tracks two pieces of state:
  - whether the room is currently considered "in an empty stretch", and
  - the monotonic timestamp the empty stretch began.

When the empty stretch reaches the debounce threshold, the callback fires
exactly once. If a person reappears before the threshold, the watcher resets
and waits for the next ≥1 → 0 transition.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Optional

log = logging.getLogger(__name__)


class EmptyRoomWatcher:
    def __init__(
        self,
        on_empty: Callable[[], None],
        debounce_s: float,
        now_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self._on_empty = on_empty
        self._debounce_s = debounce_s
        self._now_fn = now_fn

        self._was_occupied: bool = False    # last-seen person_count >= 1
        self._empty_since: Optional[float] = None
        self._fired_for_current_stretch: bool = False

    def update(self, person_count: int) -> None:
        now = self._now_fn()
        if person_count >= 1:
            # Person present — reset everything
            self._was_occupied = True
            self._empty_since = None
            self._fired_for_current_stretch = False
            return

        # person_count == 0
        if self._was_occupied or self._empty_since is None:
            # Just transitioned to empty — start the debounce timer.
            # (Or we've been empty since startup and want to baseline.)
            self._empty_since = now
            self._was_occupied = False
            self._fired_for_current_stretch = False
            return

        # Still empty. Check the debounce.
        if self._fired_for_current_stretch:
            return
        if now - self._empty_since >= self._debounce_s:
            self._fired_for_current_stretch = True
            try:
                self._on_empty()
            except Exception as exc:
                log.warning("EmptyRoomWatcher callback raised: %s", exc)
```

- [ ] **Step 4: Run the test, confirm it passes**

```bash
python tests/test_empty_room_watcher.py
```

Expected: 5 OK lines + `All EmptyRoomWatcher tests passed.`

- [ ] **Step 5: Commit**

```bash
git add agents/empty_room_watcher.py tests/test_empty_room_watcher.py
git commit -m "feat: EmptyRoomWatcher with 3s debounce for synthetic room_empty_confirmed event"
```

---

## Task 5 — Wire everything into `main.py`

**Files:**
- Modify: `agents/__init__.py` (export EmptyRoomWatcher — optional, only if other agents-prefix imports require it; otherwise skip)
- Modify: `main.py` — DecisionEngine construction, EmptyRoomWatcher construction + tick, synthetic event push

### What & why

Two integration changes in `main.py`:

1. `DecisionEngine(plugs, speaker)` becomes `DecisionEngine(plugs, speaker, world)`.
2. Construct an `EmptyRoomWatcher` whose callback pushes a `room_empty_confirmed` synthetic event to the `reasoner_worker`. Call `watcher.update(world._people_count)` once per tick. (Reading `_people_count` directly is consistent with how the existing status print does it, but a cleaner accessor is fine too — see Step 3.)

The synthetic event uses the same `reasoner_worker.push_work(obs_result, event_types, frame)` API as the minutely summary so it flows through the existing pipeline. `room_empty_confirmed` is in `REASONER_ALWAYS` (added in Task 1), so `should_call_reasoner` will fire it through.

### Steps

- [ ] **Step 1: Update the DecisionEngine call site in `main.py`**

Find (around `main.py:335`):

```python
    decisions = DecisionEngine(plugs=plugs, speaker=speaker)
```

Replace with:

```python
    decisions = DecisionEngine(plugs=plugs, speaker=speaker, world=world)
```

- [ ] **Step 2: Construct and wire the EmptyRoomWatcher in `main.py`**

Add the import near the other `from agents.…` imports (around `main.py:47`):

```python
from agents.empty_room_watcher import EmptyRoomWatcher
```

After the DecisionEngine is constructed (a few lines below `main.py:335`), construct the watcher. Build the callback as a closure over `reasoner_worker` and `camera`:

```python
    def _on_room_empty() -> None:
        log.info("EmptyRoomWatcher: room_empty_confirmed (debounce elapsed)")
        if not config.REASONER_ENABLED:
            return
        reasoner_worker.push_work(
            obs_result={
                "narration": "",
                "escalate": False,
                "escalate_reason": "room empty for debounce window",
            },
            event_types=["room_empty_confirmed"],
            frame=camera.latest_frame(),
        )

    empty_room_watcher = EmptyRoomWatcher(
        on_empty=_on_room_empty,
        debounce_s=config.EMPTY_ROOM_DEBOUNCE_S,
    )
```

- [ ] **Step 3: Tick the watcher each main loop iteration**

Inside the main `while True:` loop, after the existing world-state updates (just before the routing/event-dispatch block — look for where `world.update_from_yolo(...)` is called and add this immediately after the world updates), add:

```python
            empty_room_watcher.update(world.people_count())
```

- [ ] **Step 4: Make sure `update_devices` is being called every loop**

Find the existing call to `world.update_devices(plugs)` in `main.py`. It should already exist; if it isn't being called every tick, move it into the loop so override detection runs at the plug poll cadence. Use the existing plug-poll call site — don't introduce a new poll.

(If `world.update_devices` is missing entirely from the main loop, add it immediately before `empty_room_watcher.update(...)`:)

```python
            if plugs is not None:
                world.update_devices(plugs)
            empty_room_watcher.update(world.people_count())
```

- [ ] **Step 5: Smoke-run the system (no Claude calls needed for this verification)**

```bash
source venv/bin/activate
python main.py
```

Expected within ~10s:
- No tracebacks.
- Log lines confirm plug discovery (or fall back gracefully if hardware unavailable).
- Walking out of frame for ≥3s logs: `EmptyRoomWatcher: room_empty_confirmed (debounce elapsed)`.

If no hardware is available, the run should still come up; if it crashes because `plugs is None`, guard the watcher wiring with `if plugs is not None:`. Otherwise leave it as-is.

Stop with Ctrl-C.

- [ ] **Step 6: Commit**

```bash
git add main.py
git commit -m "feat: wire EmptyRoomWatcher + DecisionEngine(world=) into main loop"
```

---

## Task 6 — Update `DESIGN.md` and `LEARNING.md`

**Files:**
- Modify: `DESIGN.md` (data-flow diagram + module list)
- Modify: `LEARNING.md` (append one entry)

### What & why

CLAUDE.md requires DESIGN.md to be updated in the same commit as any module that flips from stub to functional, and a LEARNING.md entry when we work around a hard problem or change our minds. DecisionEngine just went from passthrough to a real two-tier policy, and EmptyRoomWatcher is a new module — both qualify.

### Steps

- [ ] **Step 1: Update `DESIGN.md`**

Two edits:

1. In the data-flow diagram, the Reasoner → DecisionEngine arrow now carries `lamp`/`fan`/`lamp_reason`/`fan_reason` (previously always null). Update the contract description accordingly.
2. Add `EmptyRoomWatcher` to the module list, noting its single responsibility (debounce empty-room transitions into a synthetic Reasoner event), and update DecisionEngine's description to mention the new guardrails (idempotency, cooldown, override lockout, no-person ON, plug unreachable).

Keep edits surgical — don't rewrite untouched sections.

- [ ] **Step 2: Update `ARCHITECTURE_AND_BUILD_PLAN copy.md` if it conflicts**

If that doc still claims plug control is disabled or that the Reasoner doesn't actuate, update the affected sentence(s) only. Don't rewrite the section.

- [ ] **Step 3: Append a `LEARNING.md` entry**

Use the template at the bottom of `LEARNING.md`. The entry should cover:

- **Block:** Lamp & fan control (Day 2/3 boundary).
- **Tag:** architecture / decision-making.
- **What I thought:** that the Reasoner prompt alone was enough — let Claude decide and trust it.
- **What actually happened:** prompts can't enforce "don't toggle twice in 5s" or "the user just touched it — back off" reliably. Soft policy in the prompt drifts; hard policy in code stays exact. The split (Claude does judgment, DecisionEngine does guardrails) is what makes the agent *feel* respectful instead of bossy.
- **Lesson:** when an LLM is your decider, code-level guardrails aren't a fallback — they're the *contract* between the model and the physical world. Keep the rules small (5 guards), keep them named, and surface refusals to the dashboard so you can debug *why* nothing happened.

- [ ] **Step 4: Commit**

```bash
git add DESIGN.md "ARCHITECTURE_AND_BUILD_PLAN copy.md" LEARNING.md
git commit -m "docs: update DESIGN.md + LEARNING.md for lamp/fan control rollout"
```

---

## Final verification

After Task 6 commits, run all three test scripts in a single command:

```bash
source venv/bin/activate
python tests/test_world_state_devices.py && \
python tests/test_decision_engine.py && \
python tests/test_empty_room_watcher.py
```

Expected: each script ends with `All … tests passed.` Total ~22 OK lines.

Then start the system for a 60-second sanity run:

```bash
python main.py
```

While it runs, manually verify:
1. **A1:** walk into a dim room → lamp turns on within a few seconds, narration says something like *"Got the light for you."*
2. **R3:** physically toggle the lamp off via the plug button → within one Reasoner cycle, agent narrates an acknowledgment and stops trying to control the lamp for 5 minutes (check logs for `REFUSED (override_lockout …)`).
3. **C1:** leave the room → within ~5–6s, lamp + fan turn off, narration confirms.

Stop with Ctrl-C.

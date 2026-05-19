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

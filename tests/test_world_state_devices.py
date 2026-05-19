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
    # Pass the same synthetic time reference so lockout_until (1315.0) is in the future
    snap = w.snapshot_for_reasoner(now=t0 + 16.0)
    devices = snap.get("devices", {})
    lamp = devices.get(config.LAMP_PLUG_ALIAS)
    assert lamp is not None
    assert lamp["on"] is False
    assert lamp["lockout_active"] is True
    print("OK test_snapshot_for_reasoner_includes_lockout_info")


def test_update_devices_no_override_on_first_contact_when_plug_already_on():
    """Plug is already ON when agent boots — must NOT flag as override."""
    w = WorldState()
    # First-ever poll sees lamp ON. No prior command.
    w.update_devices(_fake_plugs(lamp_on=True, fan_on=False), now=1000.0)
    ds = w.device_state(config.LAMP_PLUG_ALIAS)
    assert ds.on is True
    assert ds.last_manual_override_at is None
    assert ds.lockout_until is None
    print("OK test_update_devices_no_override_on_first_contact_when_plug_already_on")


def main():
    test_device_state_defaults_have_new_fields()
    test_record_agent_command_sets_intent_and_ts()
    test_update_devices_no_override_when_state_matches_intent()
    test_update_devices_no_override_within_grace_window()
    test_update_devices_detects_override_after_grace()
    test_update_devices_detects_override_when_no_prior_command()
    test_snapshot_for_reasoner_includes_lockout_info()
    test_update_devices_no_override_on_first_contact_when_plug_already_on()
    print("\nAll WorldState device tests passed.")


if __name__ == "__main__":
    main()

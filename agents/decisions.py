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

    def execute(self, output: ReasonerOutput) -> None:
        """Run all actions from a ReasonerOutput in fixed order:
        1. Lamp control, 2. Fan control, 3. Alert, 4. Beat 2 TTS.
        """
        self._handle_device(config.LAMP_PLUG_ALIAS, output.lamp, output.lamp_reason)
        self._handle_device(config.FAN_PLUG_ALIAS, output.fan, output.fan_reason)
        self._handle_alert(output.alert, output.reasoning)
        self._handle_narration(output.narration, output.speak)

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

"""
Action decision engine.

Takes a ReasonerOutput and dispatches concrete actions: control smart plugs
(lamp/fan via PlugManager), enqueue TTS narration (via Speaker), and log
security alerts to the dashboard broadcaster.

DecisionEngine is intentionally thin — it translates the Reasoner's intent
into physical side effects and logs. No reasoning happens here.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Optional

import config
from agents.reasoner import ReasonerOutput
from server.broadcaster import broadcaster

if TYPE_CHECKING:
    from actuators.speaker import Speaker
    from perception.plugs import PlugManager

log = logging.getLogger(__name__)


class DecisionEngine:
    """Translates ReasonerOutput into physical actions.

    Called from the main loop after a successful Reasoner poll. All plug
    control is non-blocking at the PlugManager level (it runs its own async
    event loop). Speaker.enqueue_beat2() is also non-blocking.
    """

    def __init__(
        self,
        plugs: Optional[PlugManager],
        speaker: Optional[Speaker],
    ) -> None:
        self._plugs = plugs
        self._speaker = speaker

    def execute(self, output: ReasonerOutput) -> None:
        """Execute all actions from a ReasonerOutput in order:
        1. Lamp control
        2. Fan control
        3. Alert logging
        4. Beat 2 TTS narration
        """
        self._handle_lamp(output.lamp)
        self._handle_fan(output.fan)
        self._handle_alert(output.alert, output.reasoning)
        self._handle_narration(output.narration, output.speak)

    # ----- private handlers -----

    def _handle_lamp(self, state: Optional[str]) -> None:
        if state is None or self._plugs is None:
            return
        if state == "on":
            self._plugs.turn_on(config.LAMP_PLUG_ALIAS)
            log.info("DecisionEngine: lamp → ON")
        elif state == "off":
            self._plugs.turn_off(config.LAMP_PLUG_ALIAS)
            log.info("DecisionEngine: lamp → OFF")

    def _handle_fan(self, state: Optional[str]) -> None:
        if state is None or self._plugs is None:
            return
        if state == "on":
            self._plugs.turn_on(config.FAN_PLUG_ALIAS)
            log.info("DecisionEngine: fan → ON")
        elif state == "off":
            self._plugs.turn_off(config.FAN_PLUG_ALIAS)
            log.info("DecisionEngine: fan → OFF")

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
            # TTS not available — just log so the narration isn't silently lost
            log.info("DecisionEngine: Beat 2 (no TTS): %s", narration)

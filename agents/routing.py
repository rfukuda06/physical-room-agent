"""
Hybrid routing policy for the Reasoner.

The Observer fires on every structural event + periodic refresh. The Reasoner
is gated by two rules:
  1. Event type is in config.REASONER_ALWAYS → always fire.
  2. Observer's output sets `escalate=true` → fire.
Otherwise: Beat 1 only, no Claude call, no Beat 2.

See ARCHITECTURE_AND_BUILD_PLAN copy.md §2.5 for the rationale.
"""

import config


def should_call_reasoner(event_type: str, observer_output: dict) -> bool:
    if not config.REASONER_ENABLED:
        return False
    if event_type in config.REASONER_ALWAYS:
        return True
    if observer_output.get("escalate") is True:
        return True
    return False

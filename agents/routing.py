"""
Hybrid routing policy for the Reasoner.

The Observer fires on every structural event + periodic refresh. The Reasoner
is gated by two rules:
  1. Event type is in config.REASONER_ALWAYS → always fire (with churn exception).
  2. Observer's output sets `escalate=true` → fire.
Otherwise: Beat 1 only, no Claude call, no Beat 2.

Churn exception: new_person and lost_person are in REASONER_ALWAYS but are
frequently BoT-SORT track ID re-assignments of the same physical person, not
real arrivals or departures. If the Observer confirms "room unchanged" and
didn't escalate, we skip the Reasoner to avoid burning Claude calls on noise.

See ARCHITECTURE_AND_BUILD_PLAN copy.md §2.5 for the rationale.
"""

import config

# Event types prone to track ID churn that require Observer confirmation
_CHURN_PRONE = frozenset({"new_person", "lost_person"})


def should_call_reasoner(event_type: str, observer_output: dict) -> bool:
    if not config.REASONER_ENABLED:
        return False
    if event_type in config.REASONER_ALWAYS:
        # For churn-prone events, require Observer confirmation of a real change.
        # If Observer said "room unchanged" and didn't escalate, it's track churn.
        if event_type in _CHURN_PRONE:
            narration = observer_output.get("narration", "").lower()
            escalated = observer_output.get("escalate", False)
            if not escalated and "room unchanged" in narration:
                return False
        return True
    if observer_output.get("escalate") is True:
        return True
    return False

"""
Empty-room watcher — fires a synthetic event when the room has been empty
for EMPTY_ROOM_DEBOUNCE_S after a confirmed ≥1 → 0 transition.

Driven by the main loop, which calls update(person_count) every tick.
The watcher tracks two pieces of state:
  - whether the room is currently considered "in an empty stretch", and
  - the monotonic timestamp the empty stretch began.

Strict ≥1 → 0 semantics: the debounce timer only starts after at least one
person has been observed (person_count >= 1) and then the count drops to 0.
Cold-start-empty rooms — where the room is empty when the agent starts and no
person has ever appeared — are silently ignored; the watcher waits for the
first person to appear before it can fire on their departure.

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
        if not self._was_occupied and self._empty_since is None:
            # Cold start, never seen a person yet — ignore until someone appears
            return

        if self._empty_since is None:
            # Just transitioned from occupied → empty: start debounce timer
            self._empty_since = now
            self._was_occupied = False
            self._fired_for_current_stretch = False
            return

        # Still empty (and we did see someone earlier). Check the debounce.
        if self._fired_for_current_stretch:
            return
        if now - self._empty_since >= self._debounce_s:
            self._fired_for_current_stretch = True
            try:
                self._on_empty()
            except Exception as exc:
                log.warning("EmptyRoomWatcher callback raised: %s", exc)

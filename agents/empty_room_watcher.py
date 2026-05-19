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

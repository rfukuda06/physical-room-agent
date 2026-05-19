"""Unit tests for EmptyRoomWatcher."""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


def test_does_not_fire_on_cold_start_empty():
    """Room empty at startup (no one ever seen) must NOT fire — wait for >=1->0 transition."""
    clk = _Clock()
    cb, calls = _record_calls()
    w = EmptyRoomWatcher(cb, debounce_s=3.0, now_fn=clk)
    # Many empty ticks, never a person — must not fire
    for _ in range(20):
        w.update(person_count=0)
        clk.advance(1.0)
    assert calls == []
    print("OK test_does_not_fire_on_cold_start_empty")


def main():
    test_does_not_fire_when_room_never_emptied()
    test_fires_after_debounce_when_room_goes_empty()
    test_does_not_refire_while_still_empty()
    test_resets_if_person_returns_before_debounce()
    test_fires_again_after_second_emptying()
    test_does_not_fire_on_cold_start_empty()
    print("\nAll EmptyRoomWatcher tests passed.")


if __name__ == "__main__":
    main()

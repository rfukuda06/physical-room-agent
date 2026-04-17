"""
DashboardBroadcaster — bridge from the sync perception loop to async FastAPI.

The main loop (and all its helper threads) produces data synchronously: YOLO
ticks, WorldState snapshots, Observer narrations. FastAPI's WebSocket
handlers are async coroutines living on a single asyncio event loop.
This class is the single place those two worlds meet.

How it works:

  * `bind_loop()` is called once from FastAPI's startup hook. It captures a
    reference to the running asyncio loop so producers (on any thread) can
    schedule work onto it via `run_coroutine_threadsafe`.
  * Producers call `publish_*` methods from *any* thread. The broadcaster
    enqueues the message onto every connected client's asyncio.Queue, using
    run_coroutine_threadsafe to cross the thread boundary safely.
  * Each WebSocket handler calls `register()` to get its own queue, then
    loops: `msg = await queue.get(); await ws.send_json(msg)`.
  * For video, the main loop calls `publish_frame(jpeg_bytes)`; the MJPEG
    endpoint waits on a condition variable and yields each new frame.

Message shapes (JSON sent over the WebSocket):
  {"kind": "snapshot",  "data": <world_state.snapshot()>}
  {"kind": "event",     "data": {type, ts, track_id, zones, payload}}
  {"kind": "narration", "agent": "observer"|"reasoner", "data": {...}}
  {"kind": "routing",   "data": {trigger, fired, reason}}

Backpressure: each client queue is bounded (maxsize=512). If a slow client
falls behind, oldest messages are dropped. The video stream is independent
of the WebSocket queue.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Optional

log = logging.getLogger(__name__)


class DashboardBroadcaster:
    """Thread-safe fan-out for dashboard data.  Singleton — import `broadcaster`."""

    _QUEUE_MAXSIZE = 512

    def __init__(self) -> None:
        # Thread lock protects the client set and the JPEG slot.
        # Kept tight — never held across an I/O call.
        self._lock = threading.Lock()

        # Captured in bind_loop() — the FastAPI event loop.
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Connected WS clients. Each client gets its own asyncio.Queue so
        # slow clients can't block fast ones.
        self._clients: set[asyncio.Queue] = set()

        # MJPEG stream state. We store the latest already-encoded JPEG so
        # the HTTP endpoint can hand it out with zero work. A Condition lets
        # the generator wait until a new frame arrives instead of polling.
        self._jpeg_cond = threading.Condition()
        self._latest_jpeg: Optional[bytes] = None
        self._latest_jpeg_seq: int = 0

    # ------------------------------------------------------------------
    # Setup / teardown
    # ------------------------------------------------------------------

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Attach the FastAPI event loop. Call once from startup."""
        self._loop = loop
        log.info("DashboardBroadcaster bound to asyncio loop")

    def has_clients(self) -> bool:
        """Cheap check for skipping expensive work (like JPEG encoding)
        when nobody is watching. Approximate — clients may connect between
        this call and the subsequent publish."""
        with self._lock:
            return bool(self._clients)

    # ------------------------------------------------------------------
    # Producer API (called from any thread in the perception loop)
    # ------------------------------------------------------------------

    def publish_snapshot(self, snap: dict) -> None:
        self._broadcast({"kind": "snapshot", "data": snap})

    def publish_event(self, event: dict) -> None:
        self._broadcast({"kind": "event", "data": event})

    def publish_narration(self, agent: str, data: dict) -> None:
        self._broadcast({"kind": "narration", "agent": agent, "data": data})

    def publish_routing(self, data: dict) -> None:
        self._broadcast({"kind": "routing", "data": data})

    def publish_frame(self, jpeg_bytes: bytes) -> None:
        """Store the latest annotated JPEG and wake up MJPEG readers."""
        with self._jpeg_cond:
            self._latest_jpeg = jpeg_bytes
            self._latest_jpeg_seq += 1
            self._jpeg_cond.notify_all()

    # ------------------------------------------------------------------
    # Consumer API (called from inside FastAPI async handlers)
    # ------------------------------------------------------------------

    def register(self) -> asyncio.Queue:
        """Create a new per-client queue and add it to the fan-out set.
        Must be called from the asyncio loop thread."""
        q: asyncio.Queue = asyncio.Queue(maxsize=self._QUEUE_MAXSIZE)
        with self._lock:
            self._clients.add(q)
        log.info("WS client registered (total=%d)", len(self._clients))
        return q

    def unregister(self, q: asyncio.Queue) -> None:
        with self._lock:
            self._clients.discard(q)
        log.info("WS client unregistered (total=%d)", len(self._clients))

    def wait_for_frame(self, last_seq: int, timeout: float = 1.0
                       ) -> tuple[Optional[bytes], int]:
        """Block until a newer frame than `last_seq` is available.
        Returns (jpeg_bytes, new_seq). If timeout elapses, returns the
        current frame (possibly None, possibly the same seq)."""
        with self._jpeg_cond:
            if self._latest_jpeg_seq <= last_seq:
                self._jpeg_cond.wait(timeout=timeout)
            return self._latest_jpeg, self._latest_jpeg_seq

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _broadcast(self, msg: dict) -> None:
        """Thread-safe: enqueue msg to every connected client's queue.

        If the loop isn't bound yet (server not started), silently drop.
        If a client's queue is full, drop the oldest message to make room —
        losing a stale snapshot is better than blocking the perception loop.
        """
        loop = self._loop
        if loop is None:
            return
        with self._lock:
            clients = list(self._clients)
        if not clients:
            return
        for q in clients:
            try:
                asyncio.run_coroutine_threadsafe(self._put(q, msg), loop)
            except RuntimeError:
                # Loop has been closed — ignore.
                pass

    async def _put(self, q: asyncio.Queue, msg: dict) -> None:
        """Enqueue with drop-oldest policy if full."""
        try:
            q.put_nowait(msg)
        except asyncio.QueueFull:
            try:
                q.get_nowait()  # drop oldest
            except asyncio.QueueEmpty:
                pass
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                pass  # give up


# Module-level singleton. Import and use this everywhere.
broadcaster = DashboardBroadcaster()

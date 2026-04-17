"""
FastAPI app for the Newton-for-a-Room dashboard.

Endpoints:
  GET /                — health check ("ok")
  GET /config          — static dashboard config (zones, frame size, thresholds)
  GET /video/stream    — multipart MJPEG of the latest annotated YOLO frame
  WS  /ws/state        — live world-state, events, and agent narrations

The app is created once at import time. `run_server_in_thread()` starts
uvicorn on a daemon thread so the perception loop (main.py) can keep
driving. On startup, it binds the event loop to the DashboardBroadcaster
singleton so producers on other threads can fan out messages safely.

**FastAPI in one minute (for first-timers):**

FastAPI is a Python web framework built on Starlette + Pydantic. A function
decorated with `@app.get("/path")` becomes an HTTP handler; type-annotated
parameters get parsed/validated automatically. Async handlers (`async def`)
run on an event loop, which is the same concurrency model Node.js uses —
one thread juggles many connections via `await`. StreamingResponse lets
you yield bytes incrementally, which is how MJPEG and SSE work. WebSocket
is a separate handler type that supports bidirectional messages.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, StreamingResponse

import config
from server.broadcaster import broadcaster

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Newton-for-a-Room Dashboard", version="0.1.0")

# Next.js dev server typically runs on :3000; allow all origins during dev.
# In production we'd lock this down to the dashboard host.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _on_startup() -> None:
    """Attach the running event loop to the broadcaster so cross-thread
    producers can schedule messages onto it."""
    loop = asyncio.get_running_loop()
    broadcaster.bind_loop(loop)


# ---------------------------------------------------------------------------
# HTTP endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_class=PlainTextResponse)
async def root() -> str:
    return "Newton-for-a-Room dashboard backend — ok"


@app.get("/config")
async def get_config() -> dict:
    """One-shot config the frontend needs at page load.

    Zones are in camera-pixel coords; the frontend overlays them as SVG
    polygons over the video element, scaling by the rendered element's size.
    """
    return {
        "camera": {
            "width": config.CAMERA_CAPTURE_WIDTH,
            "height": config.CAMERA_CAPTURE_HEIGHT,
            "fps": config.CAMERA_CAPTURE_FPS,
        },
        "zones": {
            name: [list(pt) for pt in pts]
            for name, pts in config.ZONES.items()
        },
        "thresholds": {
            "audio_spike_db": config.AUDIO_SPIKE_DB_THRESHOLD,
            "yamnet_min_conf": config.YAMNET_MIN_CONFIDENCE,
        },
        "agents": {
            "observer_enabled": config.OBSERVER_ENABLED,
            "observer_model": config.GEMINI_MODEL,
            "reasoner_enabled": config.REASONER_ENABLED,
        },
        "calibration_seconds": config.CALIBRATION_SECONDS,
    }


# ---------------------------------------------------------------------------
# MJPEG stream
# ---------------------------------------------------------------------------
#
# MJPEG (Motion JPEG) is the simplest possible live-video format: an endless
# multipart HTTP response where each "part" is a single JPEG image. Browsers
# render it in a plain <img> tag — no JS required. It's inefficient vs. real
# video codecs, but perfect for an internal dashboard.
#
# Format (multipart/x-mixed-replace):
#   --frame\r\n
#   Content-Type: image/jpeg\r\n
#   Content-Length: <n>\r\n
#   \r\n
#   <jpeg bytes>\r\n
#   --frame\r\n
#   ...
#
# We wait on the broadcaster's condition variable so we emit exactly one
# frame per new publish — no polling, no duplicates.

_MJPEG_BOUNDARY = b"--frame"


async def _mjpeg_generator():
    """Yield MJPEG parts as new frames arrive on the broadcaster."""
    last_seq = -1
    loop = asyncio.get_running_loop()
    try:
        while True:
            # wait_for_frame is blocking (threading.Condition.wait), so run
            # it in the default executor to avoid stalling the event loop.
            jpeg, seq = await loop.run_in_executor(
                None, broadcaster.wait_for_frame, last_seq, 1.0
            )
            if jpeg is None or seq == last_seq:
                # No new frame within timeout — loop and wait again. This
                # keeps the connection alive while the perception loop warms
                # up (e.g., during calibration).
                continue
            last_seq = seq
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                + jpeg + b"\r\n"
            )
    except asyncio.CancelledError:
        # Client disconnected — clean exit.
        return


@app.get("/video/stream")
async def video_stream() -> StreamingResponse:
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ---------------------------------------------------------------------------
# WebSocket — state, events, narrations
# ---------------------------------------------------------------------------

@app.websocket("/ws/state")
async def ws_state(ws: WebSocket) -> None:
    """One WebSocket per connected dashboard client.

    After accepting, we register a queue with the broadcaster; every
    publish_* call from the perception loop gets enqueued here. We loop,
    pulling messages and sending them as JSON. On disconnect, we
    unregister so the broadcaster stops queuing for this client.
    """
    await ws.accept()
    queue = broadcaster.register()
    try:
        while True:
            msg = await queue.get()
            await ws.send_json(msg)
    except WebSocketDisconnect:
        pass
    except Exception as e:  # catch-all to ensure cleanup
        log.warning("WS state handler exited with: %r", e)
    finally:
        broadcaster.unregister(queue)


# ---------------------------------------------------------------------------
# Programmatic uvicorn launcher
# ---------------------------------------------------------------------------

def run_server_in_thread(
    host: str = "127.0.0.1",
    port: int = 8000,
    log_level: str = "warning",
) -> threading.Thread:
    """Start uvicorn on a daemon thread so main.py can continue.

    uvicorn is the ASGI server that actually speaks HTTP/WebSocket — FastAPI
    only defines the routes. We construct a Config + Server explicitly
    (instead of `uvicorn.run`) so we can control log level and run inside
    a thread. `asyncio.run(server.serve())` spins up the event loop that
    FastAPI lives in for this process.
    """
    import uvicorn

    cfg = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level=log_level,
        access_log=False,
    )
    server = uvicorn.Server(cfg)

    def _run() -> None:
        asyncio.run(server.serve())

    t = threading.Thread(target=_run, daemon=True, name="dashboard-server")
    t.start()
    log.info("Dashboard server thread started on http://%s:%d", host, port)
    return t

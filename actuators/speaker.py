"""
Text-to-speech output (edge-tts).

Owns the two-beat TTS queue: Beat 1 plays immediately, Beat 2 queues after
Beat 1 finishes. Never talks over itself.

How it works:
-------------
A single daemon thread ("tts-worker") serializes all audio output. Items are
enqueued as (kind, text) tuples. The worker processes them FIFO:

  - "beat1": synthesize → play → signal _beat1_done
  - "beat2": wait for _beat1_done → synthesize → play

In practice, Beat 1 is always enqueued before Beat 2 (Observer finishes ~1s
before Reasoner), so they naturally play in sequence. The _beat1_done Event
is a safety net for any edge-case race condition.

Audio synthesis: edge-tts calls Microsoft's cloud TTS API and returns MP3
audio. The audio is saved to a temp file and played with `afplay` (macOS).
On non-macOS systems, `ffplay` is tried as a fallback.

If TTS_ENABLED is False in config, all public methods are no-ops.
"""

from __future__ import annotations

import asyncio
import logging
import os
import queue
import subprocess
import tempfile
import threading
from typing import Optional

import config

log = logging.getLogger(__name__)


class Speaker:
    """Two-beat TTS queue using edge-tts + afplay.

    Beat 1 (Observer narration): enqueue_beat1() — plays as soon as possible.
    Beat 2 (Reasoner narration): enqueue_beat2() — plays after Beat 1 finishes.

    Both methods are non-blocking. The daemon thread handles synthesis + playback.
    """

    def __init__(self) -> None:
        self._queue: queue.Queue = queue.Queue()
        # Signals when Beat 1 has finished playing.
        # Initially set (True) — no Beat 1 currently playing.
        self._beat1_done = threading.Event()
        self._beat1_done.set()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def start(self) -> None:
        """Start the TTS worker daemon thread."""
        self._thread = threading.Thread(
            target=self._run, name="tts-worker", daemon=True,
        )
        self._thread.start()
        log.info("Speaker (TTS) started (voice=%s)", config.TTS_VOICE)

    def stop(self) -> None:
        """Signal the worker to stop. Returns quickly — audio may still be playing."""
        self._stop.set()
        self._queue.put(None)  # unblock queue.get()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            log.info("Speaker (TTS) stopped")

    def enqueue_beat1(self, text: str) -> None:
        """Enqueue Beat 1 narration (from the Observer). Non-blocking.

        Clears _beat1_done so that any pending Beat 2 waits until this plays.
        """
        if not config.TTS_ENABLED or not text.strip():
            return
        self._beat1_done.clear()
        self._queue.put(("beat1", text.strip()))

    def enqueue_beat2(self, text: str) -> None:
        """Enqueue Beat 2 narration (from the Reasoner). Non-blocking.

        The worker will wait for Beat 1 to finish before playing this.
        """
        if not config.TTS_ENABLED or not text.strip():
            return
        self._queue.put(("beat2", text.strip()))

    # ----- worker thread -----

    def _run(self) -> None:
        """Worker thread: drain queue, synthesize and play audio in order."""
        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if item is None:  # stop sentinel
                break

            kind, text = item

            if kind == "beat2":
                # Wait for Beat 1 to finish before starting Beat 2.
                # Timeout of 15s ensures Beat 2 eventually plays even if Beat 1
                # somehow got stuck (should never happen for a 40-word narration).
                self._beat1_done.wait(timeout=15.0)

            self._synthesize_and_play(text)

            if kind == "beat1":
                self._beat1_done.set()

    def _synthesize_and_play(self, text: str) -> None:
        """Synthesize with edge-tts, play with afplay. Blocks until done."""
        try:
            path = asyncio.run(self._synthesize(text))
        except Exception as exc:
            log.warning("Speaker: TTS synthesis failed: %s", exc)
            return

        if path is None:
            return

        try:
            # afplay is macOS-native and plays MP3 natively. Blocks until done.
            subprocess.run(["afplay", path], check=False, timeout=60)
        except FileNotFoundError:
            # Not macOS — try ffplay as a cross-platform fallback
            try:
                subprocess.run(
                    ["ffplay", "-nodisp", "-autoexit", path],
                    check=False,
                    timeout=60,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception as exc2:
                log.warning("Speaker: audio playback failed (no afplay or ffplay): %s", exc2)
        except Exception as exc:
            log.warning("Speaker: afplay failed: %s", exc)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    @staticmethod
    async def _synthesize(text: str) -> Optional[str]:
        """Synthesize text to a temp .mp3 file via edge-tts. Returns file path."""
        try:
            import edge_tts
        except ImportError:
            log.error("edge-tts not installed. Run: pip install edge-tts")
            return None

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            path = f.name

        try:
            communicate = edge_tts.Communicate(text, voice=config.TTS_VOICE)
            await communicate.save(path)
            return path
        except Exception as exc:
            log.warning("Speaker: edge-tts synthesis error: %s", exc)
            try:
                os.unlink(path)
            except OSError:
                pass
            return None

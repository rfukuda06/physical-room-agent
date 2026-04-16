"""
Audio perception: mic streaming + dB monitoring + YAMNet classification.

Streams from the microphone via ``sounddevice``, computes rolling dB levels,
detects volume spikes, and classifies sounds using Google's YAMNet model on
rolling 1-second windows every ~500 ms.  Filters classifications to a
room-relevant whitelist of ~35 AudioSet classes with temporal smoothing
(a class must persist for N consecutive windows before we report it).

--- Quick primer on the concepts here (for first-time readers) ---

* **sounddevice** is a Python wrapper around PortAudio.  We open an
  ``InputStream`` which fires a *callback* function on PortAudio's own thread
  every time a new chunk of audio samples is ready (~64 ms at 16 kHz with a
  1024-sample block).  The callback must be fast — it just copies samples into
  a ring buffer and computes a quick RMS dB.  Heavy work (YAMNet inference)
  runs on a separate thread that wakes periodically.

* **YAMNet** ("Yet Another Mobile Net") is a pre-trained audio classifier
  from Google, shipped via TensorFlow Hub.  It maps a 1-second waveform into
  scores over 521 AudioSet classes (speech, music, alarms, animals, etc.).
  Internally it computes a log-mel spectrogram and runs a MobileNet backbone.
  Inference takes ~15-25 ms on CPU — fast enough for our 500 ms cycle.

* **RMS dB** is a standard loudness measure.  RMS = sqrt(mean(samples²)),
  then dB = 20·log₁₀(RMS).  sounddevice delivers float32 samples normalised
  to [-1, 1], so 0 dB = digital full-scale.  A quiet room is roughly -55 to
  -45 dB; speech is -35 to -20 dB; a clap near the mic is -15 to -5 dB.

* **Temporal smoothing** prevents a single noisy YAMNet frame from triggering
  an event.  Each class tracks a *streak* of consecutive detection windows.
  Only after ``YAMNET_PERSISTENCE_WINDOWS`` consecutive windows (default 2 =
  1 second) do we consider the class confirmed.  When a class drops below
  confidence for one window, its streak resets to zero.  This is the same idea
  as the hysteresis in ``event_detector.py`` for pose changes.

* **Speech transitions** follow the EventDetector pattern: we emit events on
  *transitions* (silence → speech, speech → silence), not on steady state.
  Speech continuing is tracked silently via a ``speech_active`` flag in
  ``AudioState``.  The "off" transition uses the same persistence threshold
  to avoid flickering on brief pauses between sentences.

--- Threading model ---

    ┌ MainThread ─────── orchestrator: audio_monitor.tick(), latest_state()
    ├ [PortAudio cb] ──── sounddevice-managed: appends chunks, computes dB
    └ audio-classify ──── our daemon thread: YAMNet, smoothing, events

AudioClassifier is an abstract interface so YAMNet can be swapped for
CLAP / BEATs / EfficientAT later without rewiring the monitor.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from perception.event_detector import Event

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sounddevice callback block size.  1024 samples at 16 kHz ≈ 64 ms per
# callback invocation — small enough for responsive dB metering, large
# enough to keep callback overhead negligible.
# ---------------------------------------------------------------------------
_BLOCKSIZE = 1024

# ---------------------------------------------------------------------------
# Room-relevant AudioSet class whitelist.  YAMNet outputs 521 classes; we
# only care about these ~35.  Names must match the ``display_name`` column
# in YAMNet's bundled ``yamnet_class_map.csv`` — validated at load time.
# ---------------------------------------------------------------------------
ROOM_RELEVANT_CLASSES: frozenset[str] = frozenset({
    # Speech / voice
    "Speech", "Conversation", "Shout", "Whispering", "Laughter",
    # Human non-speech
    "Clapping", "Cough", "Sneeze", "Breathing", "Snoring",
    # Doors / entry
    "Door", "Doorbell", "Knock",
    # Movement
    "Walk, footsteps",
    # Breaking / impact
    "Glass", "Shatter",
    # Alarms
    "Alarm", "Alarm clock", "Smoke detector, smoke alarm", "Siren",
    # Music / media
    "Music", "Singing", "Television",
    # Appliances / room
    "Mechanical fan", "Air conditioning", "Computer keyboard",
    # Phone
    "Telephone", "Telephone bell ringing",
    # Safety-critical
    "Screaming", "Crying, sobbing",
    # Note: "Silence" and "White noise" deliberately excluded — they're
    # ambient defaults, not events worth reporting.
})

SPEECH_CLASSES: frozenset[str] = frozenset({
    "Speech", "Conversation", "Narration, monologue",
})

AUDIO_EVENT_TYPES: tuple[str, ...] = (
    "unusual_sound_class",
    "audio_spike",
    "speech_start",
    "speech_end",
)


# ---------------------------------------------------------------------------
# AudioState — snapshot consumed by WorldState / Observer / dashboard
# ---------------------------------------------------------------------------

@dataclass
class AudioState:
    """Point-in-time audio perception snapshot.

    Produced every classification cycle (~500 ms) by the classify thread;
    read by the main thread via ``AudioMonitor.latest_state()``.
    """
    audio_level_db: float                        # current RMS dB
    top_classes: list[tuple[str, float]]          # filtered + smoothed, descending
    dominant_class: str                           # top label or "speech" / "silence"
    speech_active: bool                           # True while speech is ongoing
    recent_spike: bool                            # True if a spike occurred recently
    spike_magnitude_db: float                     # delta dB of last spike (0 if none)
    timestamp: float                              # time.monotonic()


# ---------------------------------------------------------------------------
# AudioClassifier — abstract interface for swappable backends
# ---------------------------------------------------------------------------

class AudioClassifier(ABC):
    """Interface for audio classification models.

    ``load()`` is called once from ``AudioMonitor.start()``.  ``classify()``
    is called every ~500 ms from the background classify thread.
    """

    @abstractmethod
    def load(self) -> None:
        """Load model weights, warm up inference.  May be slow (seconds)."""

    @abstractmethod
    def classify(
        self, waveform: np.ndarray, sample_rate: int
    ) -> list[tuple[str, float]]:
        """Classify a waveform, return ``[(class_name, confidence), ...]``.

        Sorted by confidence descending.  ``waveform`` is 1-D float32 mono.
        """

    @abstractmethod
    def all_class_names(self) -> list[str]:
        """Return the full list of class display names the model knows."""


# ---------------------------------------------------------------------------
# YamNetClassifier — concrete TF Hub implementation
# ---------------------------------------------------------------------------

class YamNetClassifier(AudioClassifier):
    """YAMNet via TensorFlow Hub.

    TensorFlow and tensorflow_hub are imported lazily inside ``load()`` so
    that ``import perception.audio`` doesn't pay a multi-second TF startup
    tax.  The Hub model is downloaded on first run (~15 MB) and cached
    locally for subsequent loads.
    """

    def __init__(self) -> None:
        self._model = None
        self._class_names: list[str] = []

    def load(self) -> None:
        import csv
        import tensorflow_hub as hub

        log.info("Loading YAMNet model from TensorFlow Hub …")
        t0 = time.monotonic()
        self._model = hub.load("https://tfhub.dev/google/yamnet/1")

        class_map_path = self._model.class_map_path().numpy().decode("utf-8")
        with open(class_map_path) as f:
            reader = csv.DictReader(f)
            self._class_names = [row["display_name"] for row in reader]
        log.info(
            "YAMNet loaded: %d classes in %.1fs",
            len(self._class_names),
            time.monotonic() - t0,
        )

        dummy = np.zeros(16000, dtype=np.float32)
        self._model(dummy)
        log.info("YAMNet warmup complete")

    def classify(
        self, waveform: np.ndarray, sample_rate: int
    ) -> list[tuple[str, float]]:
        import tensorflow as tf

        scores, _embeddings, _spectrogram = self._model(
            tf.cast(waveform, tf.float32)
        )
        # Mean across time patches. Consider reduce_max if transient
        # sounds (clap, knock) get diluted by surrounding silence.
        peak_scores = tf.reduce_mean(scores, axis=0).numpy()

        pairs = [
            (self._class_names[i], float(peak_scores[i]))
            for i in range(len(self._class_names))
        ]
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs

    def all_class_names(self) -> list[str]:
        return list(self._class_names)


# ---------------------------------------------------------------------------
# AudioMonitor — the main class
# ---------------------------------------------------------------------------

class AudioMonitor:
    """Background audio capture + classification with event emission.

    Lifecycle mirrors ``CameraCapture`` and ``YoloEngine``:
    ``start()`` / ``stop()`` bracket the active period.  The orchestrator
    calls ``tick()`` each loop iteration to drain accumulated events, and
    ``latest_state()`` to read the current audio snapshot.
    """

    def __init__(
        self,
        *,
        device_index: int,
        sample_rate: int,
        window_seconds: float,
        classify_interval: float,
        min_confidence: float,
        persistence_windows: int,
        spike_db_threshold: float,
        spike_cooldown: float,
        db_rolling_window: float,
        classifier: Optional[AudioClassifier] = None,
    ) -> None:
        self._device_index = device_index
        self._sample_rate = sample_rate
        self._window_seconds = window_seconds
        self._classify_interval = classify_interval
        self._min_confidence = min_confidence
        self._persistence_windows = max(1, persistence_windows)
        self._spike_db_threshold = spike_db_threshold
        self._spike_cooldown = spike_cooldown
        self._db_rolling_window = db_rolling_window
        self._classifier = classifier or YamNetClassifier()

        # --- Audio ring buffer (written by PortAudio callback) ---
        max_chunks = int(
            sample_rate * (window_seconds + 0.5) / _BLOCKSIZE
        ) + 2
        self._chunks: deque[np.ndarray] = deque(maxlen=max_chunks)
        self._buffer_lock = threading.Lock()

        # --- dB tracking (written by callback, read by classify thread) ---
        self._current_db: float = -100.0
        self._db_history: deque[tuple[float, float]] = deque()
        self._db_lock = threading.Lock()

        # --- Published state + pending events ---
        self._latest_state: Optional[AudioState] = None
        self._pending_events: list[Event] = []
        self._state_lock = threading.Lock()

        # --- Classification state (owned by classify thread, no lock) ---
        self._class_streak: dict[str, int] = {}
        self._class_off_streak: dict[str, int] = {}
        self._reported_classes: set[str] = set()
        self._speech_active: bool = False
        self._speech_start_ts: float = 0.0
        self._speech_off_streak: int = 0
        self._last_spike_ts: float = 0.0
        self._last_classify_ms: float = 0.0

        # --- Validated whitelists (populated in start()) ---
        self._valid_whitelist: frozenset[str] = frozenset()
        self._valid_speech: frozenset[str] = frozenset()

        # --- Lifecycle ---
        self._stream = None  # sounddevice.InputStream
        self._classify_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._started = False

    # ---- lifecycle ----

    def start(self) -> None:
        """Load the classifier, open the mic stream, start classify thread."""
        if self._started:
            raise RuntimeError("AudioMonitor already started")

        import sounddevice as sd

        self._classifier.load()

        actual_names = set(self._classifier.all_class_names())
        self._valid_whitelist = ROOM_RELEVANT_CLASSES & actual_names
        self._valid_speech = SPEECH_CLASSES & actual_names
        missing = ROOM_RELEVANT_CLASSES - actual_names
        if missing:
            log.warning(
                "Whitelist classes not found in model (%d): %s",
                len(missing),
                sorted(missing),
            )
        log.info(
            "Whitelist validated: %d / %d classes matched",
            len(self._valid_whitelist),
            len(ROOM_RELEVANT_CLASSES),
        )

        self._stream = sd.InputStream(
            device=self._device_index,
            samplerate=self._sample_rate,
            channels=1,
            dtype="float32",
            blocksize=_BLOCKSIZE,
            callback=self._audio_callback,
        )
        self._stream.start()

        self._stop_event.clear()
        self._classify_thread = threading.Thread(
            target=self._classify_loop, name="audio-classify", daemon=True
        )
        self._classify_thread.start()
        self._started = True
        log.info(
            "AudioMonitor started: device=%d, rate=%d Hz, classify every %.1fs",
            self._device_index,
            self._sample_rate,
            self._classify_interval,
        )

    def stop(self) -> None:
        """Stop capture and classification."""
        self._stop_event.set()
        if self._classify_thread is not None:
            self._classify_thread.join(timeout=3.0)
            self._classify_thread = None
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._started = False

    def __enter__(self) -> "AudioMonitor":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    # ---- consumer API (thread-safe) ----

    def tick(self) -> list[Event]:
        """Drain and return all events accumulated since the last tick.

        Same pull pattern as ``EventDetector.tick()`` — the orchestrator
        calls this each loop iteration.  Events are never lost: the classify
        thread appends, tick() atomically swaps with an empty list.
        """
        with self._state_lock:
            events = self._pending_events
            self._pending_events = []
            return events

    def latest_state(self) -> Optional[AudioState]:
        """Return the most recent AudioState, or None before first classify."""
        with self._state_lock:
            return self._latest_state

    def stats(self) -> dict:
        """Diagnostics for logging and the smoke test."""
        with self._buffer_lock:
            n_chunks = len(self._chunks)
        with self._db_lock:
            db = self._current_db
        return {
            "buffer_chunks": n_chunks,
            "current_db": round(db, 1),
            "classify_latency_ms": round(self._last_classify_ms, 1),
            "whitelist_matched": len(self._valid_whitelist),
            "started": self._started,
        }

    # ---- sounddevice callback (runs on PortAudio thread) ----

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        """Append audio chunk + update dB.  Must be fast, never block."""
        if status:
            log.debug("sounddevice status: %s", status)

        chunk = indata[:, 0].copy()  # (frames,) float32 mono

        rms = float(np.sqrt(np.mean(chunk ** 2)))
        db = 20.0 * math.log10(rms + 1e-10)

        with self._buffer_lock:
            self._chunks.append(chunk)

        ts = time.monotonic()
        with self._db_lock:
            self._current_db = db
            self._db_history.append((ts, db))
            cutoff = ts - self._db_rolling_window
            while self._db_history and self._db_history[0][0] < cutoff:
                self._db_history.popleft()

    # ---- classify thread ----

    def _classify_loop(self) -> None:
        """Background loop: wake every classify_interval, run YAMNet."""
        while not self._stop_event.is_set():
            t0 = time.monotonic()
            try:
                self._classify_once()
            except Exception:
                log.exception("Audio classification error")
            elapsed = time.monotonic() - t0
            self._last_classify_ms = elapsed * 1000
            remaining = max(0.01, self._classify_interval - elapsed)
            self._stop_event.wait(remaining)

    def _classify_once(self) -> None:
        """One classification cycle: assemble window, run model, emit events."""
        # Assemble last window_seconds of audio from the chunk buffer.
        with self._buffer_lock:
            if not self._chunks:
                return
            waveform = np.concatenate(list(self._chunks))

        target_samples = int(self._sample_rate * self._window_seconds)
        if len(waveform) < target_samples // 2:
            return  # not enough audio yet
        if len(waveform) > target_samples:
            waveform = waveform[-target_samples:]

        ts = time.monotonic()

        # Run classification
        all_classes = self._classifier.classify(waveform, self._sample_rate)

        # Filter to whitelist + confidence threshold
        filtered = [
            (name, conf)
            for name, conf in all_classes
            if name in self._valid_whitelist and conf >= self._min_confidence
        ]

        with self._db_lock:
            current_db = self._current_db

        # --- Temporal smoothing + event emission ---
        #
        # Two-sided hysteresis for all classes:
        #   ON:  class must appear N consecutive windows to start reporting
        #   OFF: class must be absent N consecutive windows to stop reporting
        #
        # This prevents repeated events for sustained sounds (typing) and
        # avoids flicker on brief gaps.  Speech uses a longer off-threshold
        # (6 windows = 3s) to survive natural pauses between sentences.
        present_names = {name for name, _ in filtered}
        conf_map = dict(filtered)
        events: list[Event] = []

        # Update on-streaks and off-streaks for every tracked class
        for name in list(self._class_streak.keys()):
            if name not in present_names:
                self._class_streak[name] = 0
                # Increment off-streak; only remove from reported after threshold
                self._class_off_streak[name] = (
                    self._class_off_streak.get(name, 0) + 1
                )
                if (
                    name in self._reported_classes
                    and self._class_off_streak[name] >= self._persistence_windows
                ):
                    self._reported_classes.discard(name)
            else:
                self._class_off_streak[name] = 0

        for name, conf in filtered:
            self._class_streak[name] = self._class_streak.get(name, 0) + 1
            streak = self._class_streak[name]

            if streak >= self._persistence_windows and name not in self._reported_classes:
                self._reported_classes.add(name)
                self._class_off_streak[name] = 0

                if name not in self._valid_speech:
                    events.append(Event(
                        type="unusual_sound_class",
                        ts=ts,
                        track_id=None,
                        zones=[],
                        confidence=conf,
                        payload={
                            "class_name": name,
                            "confidence": round(conf, 3),
                            "db_level": round(current_db, 1),
                        },
                    ))

        # --- Speech transitions ---
        # Longer off-threshold: 4 windows (2s) to ride through natural
        # pauses between sentences without flipping speech_end/speech_start.
        _SPEECH_OFF_WINDOWS = 4

        speech_reported = bool(self._reported_classes & self._valid_speech)

        if speech_reported:
            self._speech_off_streak = 0
            if not self._speech_active:
                self._speech_active = True
                self._speech_start_ts = ts
                best_speech_conf = max(
                    (conf_map.get(c, 0.0) for c in self._valid_speech), default=0.0
                )
                events.append(Event(
                    type="speech_start",
                    ts=ts,
                    track_id=None,
                    zones=[],
                    confidence=best_speech_conf,
                    payload={
                        "confidence": round(best_speech_conf, 3),
                        "db_level": round(current_db, 1),
                    },
                ))
        elif self._speech_active:
            self._speech_off_streak += 1
            if self._speech_off_streak >= _SPEECH_OFF_WINDOWS:
                self._speech_active = False
                duration = ts - self._speech_start_ts
                events.append(Event(
                    type="speech_end",
                    ts=ts,
                    track_id=None,
                    zones=[],
                    confidence=0.0,
                    payload={"duration_seconds": round(duration, 1)},
                ))

        # --- dB spike detection ---
        with self._db_lock:
            n_readings = len(self._db_history)
            if n_readings > 10:
                mean_db = sum(db for _, db in self._db_history) / n_readings
            else:
                mean_db = current_db

        delta = current_db - mean_db
        if (
            n_readings > 10
            and delta >= self._spike_db_threshold
            and ts - self._last_spike_ts >= self._spike_cooldown
        ):
            self._last_spike_ts = ts
            events.append(Event(
                type="audio_spike",
                ts=ts,
                track_id=None,
                zones=[],
                confidence=1.0,
                payload={
                    "current_db": round(current_db, 1),
                    "baseline_db": round(mean_db, 1),
                    "delta_db": round(delta, 1),
                },
            ))

        spike_recent = (ts - self._last_spike_ts) < self._spike_cooldown * 2
        spike_mag = delta if spike_recent else 0.0

        # --- Build AudioState ---
        non_speech = [
            (n, round(c, 3))
            for n, c in filtered
            if n not in self._valid_speech
        ][:5]
        dominant = (
            non_speech[0][0]
            if non_speech
            else ("speech" if self._speech_active else "silence")
        )

        state = AudioState(
            audio_level_db=round(current_db, 1),
            top_classes=non_speech,
            dominant_class=dominant,
            speech_active=self._speech_active,
            recent_spike=spike_recent,
            spike_magnitude_db=round(spike_mag, 1),
            timestamp=ts,
        )

        with self._state_lock:
            self._latest_state = state
            self._pending_events.extend(events)


# ---------------------------------------------------------------------------
# Standalone preview: ``python -m perception.audio``
# ---------------------------------------------------------------------------

def _preview_main() -> None:
    """Console-based live audio monitor for testing and threshold tuning."""
    import config

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s  %(message)s",
    )

    monitor = AudioMonitor(
        device_index=config.AUDIO_DEVICE_INDEX,
        sample_rate=config.AUDIO_SAMPLE_RATE,
        window_seconds=config.AUDIO_WINDOW_SECONDS,
        classify_interval=config.YAMNET_CLASSIFY_INTERVAL_SECONDS,
        min_confidence=config.YAMNET_MIN_CONFIDENCE,
        persistence_windows=config.YAMNET_PERSISTENCE_WINDOWS,
        spike_db_threshold=config.AUDIO_SPIKE_DB_THRESHOLD,
        spike_cooldown=config.AUDIO_SPIKE_COOLDOWN_SECONDS,
        db_rolling_window=config.AUDIO_DB_ROLLING_WINDOW_SECONDS,
    )

    print("Starting AudioMonitor — loading YAMNet (first run downloads ~15 MB) …")
    monitor.start()

    # Let the buffer fill for a moment before displaying.
    time.sleep(1.0)

    recent_events: list[tuple[float, Event]] = []
    t_start = time.monotonic()
    max_event_log = 12

    print("\033[2J")  # clear screen

    try:
        while True:
            events = monitor.tick()
            state = monitor.latest_state()
            st = monitor.stats()

            for ev in events:
                elapsed = ev.ts - t_start
                recent_events.append((elapsed, ev))
            if len(recent_events) > max_event_log:
                recent_events = recent_events[-max_event_log:]

            # Build display
            lines: list[str] = []
            lines.append("Audio Monitor  (Ctrl+C to quit)")
            lines.append("─" * 56)

            if state is not None:
                # dB meter bar
                db = state.audio_level_db
                bar_len = 30
                # Map dB from [-80, 0] to [0, bar_len]
                fill = max(0, min(bar_len, int((db + 80) / 80 * bar_len)))
                bar = "█" * fill + "░" * (bar_len - fill)
                lines.append(f"  dB: {db:6.1f}  {bar}")
                lines.append("")

                # Top classes
                if state.top_classes:
                    cls_strs = [
                        f"{name}({conf:.2f})" for name, conf in state.top_classes[:3]
                    ]
                    lines.append(f"  Top: {', '.join(cls_strs)}")
                else:
                    lines.append("  Top: (none above threshold)")

                # Speech / spike status
                speech_str = "active" if state.speech_active else "inactive"
                spike_str = (
                    f"+{state.spike_magnitude_db:.0f}dB"
                    if state.recent_spike
                    else "none"
                )
                lines.append(
                    f"  Speech: {speech_str}  │  "
                    f"Spike: {spike_str}  │  "
                    f"Classify: {st['classify_latency_ms']:.0f}ms"
                )
            else:
                lines.append("  (waiting for first classification …)")

            lines.append("")
            lines.append("  Events:")
            if recent_events:
                for elapsed, ev in recent_events[-8:]:
                    p = ev.payload
                    if ev.type == "unusual_sound_class":
                        detail = f"{p['class_name']}  conf={p['confidence']}"
                    elif ev.type == "audio_spike":
                        detail = f"delta={p['delta_db']}dB  now={p['current_db']}dB"
                    elif ev.type == "speech_start":
                        detail = f"conf={p['confidence']}  db={p['db_level']}"
                    elif ev.type == "speech_end":
                        detail = f"duration={p['duration_seconds']}s"
                    else:
                        detail = str(p)
                    lines.append(f"    {elapsed:6.1f}s  {ev.type:24s} {detail}")
            else:
                lines.append("    (none yet)")

            lines.append("")
            lines.append(
                f"  Buffer: {st['buffer_chunks']} chunks  │  "
                f"Whitelist: {st['whitelist_matched']} classes"
            )

            # Render: move cursor to top-left and overwrite
            output = "\033[H" + "\n".join(lines) + "\033[J"
            print(output, end="", flush=True)

            time.sleep(0.2)

    except KeyboardInterrupt:
        pass
    finally:
        monitor.stop()
        elapsed = time.monotonic() - t_start
        print(f"\n\nStopped after {elapsed:.1f}s — {len(recent_events)} events total")
        for t, ev in recent_events:
            print(f"  {t:6.1f}s  {ev.type}  {ev.payload}")


if __name__ == "__main__":
    _preview_main()

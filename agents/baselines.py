"""
Calibration: learn what "normal" looks like for this specific room.

Runs for ~30 seconds at startup (configurable via config.CALIBRATION_SECONDS).
Collects audio noise floor (mean dB, std), idle power for lamp/fan, typical
occupancy, and the set of YAMNet classes that are persistently present so they
can be suppressed from Observer/Reasoner context.

**How it works:**

CalibrationCollector.run() blocks the main thread for the calibration duration.
All Layer 0 systems (camera, YOLO, audio, plugs) are already running in their
own threads by the time calibration starts.  The collector simply polls their
published state at regular intervals, accumulates samples, and at the end
computes summary statistics that become the room's Baselines.

During calibration, the collector also keeps WorldState warm — calling
update_from_yolo/audio/devices each tick so the state is already populated
when monitoring begins.  This prevents spurious "new_person" events at the
transition.

**What it produces:**

A Baselines dataclass (defined in world_state.py) with:
  - audio_mean_db / audio_std_db  — ambient noise floor
  - typical_occupancy             — median person count
  - power_idle_lamp_w / power_idle_fan_w  — device power baselines
  - ambient_audio_classes         — YAMNet classes present >=30% of the time
  - calibrated = True
"""

from __future__ import annotations

import logging
import statistics
import time
from collections import Counter
from typing import TYPE_CHECKING, Callable, Optional

import cv2

import config
from agents.world_state import Baselines, WorldState
from perception.audio import SPEECH_CLASSES, AudioMonitor
from perception.event_detector import EventDetector
from perception.plugs import PlugManager
from perception.yolo_engine import YoloEngine

if TYPE_CHECKING:
    import numpy as np

log = logging.getLogger(__name__)


class CalibrationCollector:
    """Collects sensor samples for CALIBRATION_SECONDS, then computes baselines.

    Usage (in main.py)::

        cal = CalibrationCollector(world, engine, detector, audio, plugs)
        baselines = cal.run(overlay_callback=my_draw_fn)
        # world.set_baselines() is called automatically inside run()
    """

    def __init__(
        self,
        world: WorldState,
        engine: YoloEngine,
        detector: EventDetector,
        audio: AudioMonitor,
        plugs: Optional[PlugManager],
        duration: float = config.CALIBRATION_SECONDS,
    ) -> None:
        self._world = world
        self._engine = engine
        self._detector = detector
        self._audio = audio
        self._plugs = plugs
        self._duration = duration

        # Accumulators
        self._db_samples: list[float] = []          # all dB readings
        self._db_no_speech: list[float] = []         # dB readings excluding speech
        self._class_counter: Counter[str] = Counter()  # class name → window count
        self._class_windows: int = 0                 # total classification windows
        self._occupancy_samples: list[int] = []
        self._power_lamp: list[float] = []
        self._power_fan: list[float] = []

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        overlay_callback: Optional[Callable[["np.ndarray", float, float], None]] = None,
    ) -> Baselines:
        """Block for ``self._duration`` seconds, collecting sensor samples.

        While blocking, this method:
        - Ticks the YOLO event detector and drains audio events (keeps them
          healthy and prevents stale state from building up).
        - Updates WorldState each tick so it's warm when monitoring begins.
        - Samples audio, occupancy, and power at appropriate intervals.
        - Calls ``overlay_callback(frame, elapsed, duration)`` if provided,
          so the video window can show calibration progress.
        - Listens for 'q' keypress to allow early exit during development.

        At the end, computes baselines and stores them via
        ``world.set_baselines()``.

        Returns the computed Baselines instance.
        """
        t_start = time.monotonic()
        last_audio_sample = 0.0
        last_occupancy_sample = 0.0
        last_power_sample = 0.0
        last_status_print = 0.0

        audio_interval = 0.5    # match YAMNet classify interval
        occupancy_interval = 1.0
        power_interval = 5.0    # match plug poll interval
        status_interval = 5.0

        log.info(
            "Calibration started — collecting samples for %ds", int(self._duration)
        )

        while True:
            now = time.monotonic()
            elapsed = now - t_start
            if elapsed >= self._duration:
                break

            # -- Tick sensors (keeps event detector + audio healthy) --
            result = self._engine.latest_result()
            self._detector.tick(result)
            self._audio.tick()  # drain events so they don't accumulate

            # -- Keep WorldState warm --
            self._world.update_from_yolo(result, self._detector)
            audio_st = self._audio.latest_state()
            if audio_st:
                self._world.update_audio(audio_st)
            if self._plugs:
                self._world.update_devices(self._plugs)

            # -- Sample audio (~every 500ms) --
            if elapsed - last_audio_sample >= audio_interval:
                self._collect_audio_sample()
                last_audio_sample = elapsed

            # -- Sample occupancy (~every 1s) --
            if elapsed - last_occupancy_sample >= occupancy_interval:
                self._collect_occupancy_sample()
                last_occupancy_sample = elapsed

            # -- Sample power (~every 5s) --
            if elapsed - last_power_sample >= power_interval:
                self._collect_power_sample()
                last_power_sample = elapsed

            # -- Console status (~every 5s) --
            if elapsed - last_status_print >= status_interval:
                self._print_status(elapsed)
                last_status_print = elapsed

            # -- Video overlay + display --
            if result is not None and result.annotated_frame is not None:
                frame = result.annotated_frame
                if overlay_callback is not None:
                    overlay_callback(frame, elapsed, self._duration)
                cv2.imshow("Newton-for-a-Room  (q=quit)", frame)

            # -- Keep cv2 window responsive + early exit --
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                log.info("Early exit requested during calibration (q pressed)")
                break

        # -- Compute and store baselines --
        baselines = self._compute_baselines()
        self._world.set_baselines(baselines)

        log.info(
            "Calibration complete: audio=%.1f +/- %.1f dB, occ=%d, "
            "lamp=%.1fW, fan=%.1fW, ambient=%s",
            baselines.audio_mean_db,
            baselines.audio_std_db,
            baselines.typical_occupancy,
            baselines.power_idle_lamp_w,
            baselines.power_idle_fan_w,
            baselines.ambient_audio_classes,
        )
        return baselines

    # ------------------------------------------------------------------
    # Sample collectors
    # ------------------------------------------------------------------

    def _collect_audio_sample(self) -> None:
        """Sample current audio state: dB level and active YAMNet classes."""
        state = self._audio.latest_state()
        if state is None:
            return

        db = state.audio_level_db
        self._db_samples.append(db)

        # Optionally exclude speech periods from noise floor
        if not (config.CALIBRATION_EXCLUDE_SPEECH_FROM_FLOOR and state.speech_active):
            self._db_no_speech.append(db)

        # Count class occurrences for ambient detection
        if state.top_classes:
            self._class_windows += 1
            for class_name, _conf in state.top_classes:
                self._class_counter[class_name] += 1

    def _collect_occupancy_sample(self) -> None:
        """Sample current person count from EventDetector."""
        count = len(self._detector.active_track_ids())
        self._occupancy_samples.append(count)

    def _collect_power_sample(self) -> None:
        """Sample current plug power readings (handles missing plugs)."""
        if self._plugs is None:
            return

        lamp = self._plugs.state(config.LAMP_PLUG_ALIAS)
        if lamp is not None:
            self._power_lamp.append(lamp.power_w)

        fan = self._plugs.state(config.FAN_PLUG_ALIAS)
        if fan is not None:
            self._power_fan.append(fan.power_w)

    # ------------------------------------------------------------------
    # Compute final baselines
    # ------------------------------------------------------------------

    def _compute_baselines(self) -> Baselines:
        """Reduce accumulated samples into a single Baselines instance."""

        # Audio noise floor — prefer speech-excluded samples, fall back to all
        db_source = self._db_no_speech if self._db_no_speech else self._db_samples
        if len(db_source) >= 2:
            audio_mean = statistics.mean(db_source)
            audio_std = statistics.stdev(db_source)
        elif db_source:
            audio_mean = db_source[0]
            audio_std = 5.0  # can't compute std from 1 sample
        else:
            audio_mean = -50.0
            audio_std = 5.0

        # Typical occupancy — median, robust to brief tracker drops
        if self._occupancy_samples:
            typical_occ = round(statistics.median(self._occupancy_samples))
        else:
            typical_occ = 0

        # Idle power — median (robust to outlier readings during discovery)
        power_lamp = (
            statistics.median(self._power_lamp) if self._power_lamp else 0.0
        )
        power_fan = (
            statistics.median(self._power_fan) if self._power_fan else 0.0
        )

        # Ambient audio classes — classes appearing in >= threshold of windows,
        # excluding speech classes (speech is always event-worthy, never ambient)
        ambient: list[str] = []
        if self._class_windows > 0:
            threshold = self._class_windows * config.CALIBRATION_AMBIENT_CLASS_MIN_RATIO
            for cls_name, count in self._class_counter.most_common():
                if count >= threshold and cls_name not in SPEECH_CLASSES:
                    ambient.append(cls_name)

        return Baselines(
            audio_mean_db=round(audio_mean, 1),
            audio_std_db=round(audio_std, 1),
            typical_occupancy=typical_occ,
            power_idle_lamp_w=round(power_lamp, 1),
            power_idle_fan_w=round(power_fan, 1),
            ambient_audio_classes=ambient,
            calibrated=True,
        )

    # ------------------------------------------------------------------
    # Console feedback
    # ------------------------------------------------------------------

    def _print_status(self, elapsed: float) -> None:
        """Print a one-line progress update to the console."""
        n_audio = len(self._db_samples)
        n_occ = len(self._occupancy_samples)
        n_lamp = len(self._power_lamp)
        n_fan = len(self._power_fan)

        avg_db = statistics.mean(self._db_samples) if self._db_samples else -99.0

        print(
            f"  [calibrate] {elapsed:4.0f}/{self._duration:.0f}s  "
            f"audio={avg_db:.1f}dB ({n_audio} samples)  "
            f"occ={n_occ} samples  "
            f"plugs=lamp:{n_lamp} fan:{n_fan}  "
            f"classes={len(self._class_counter)}"
        )

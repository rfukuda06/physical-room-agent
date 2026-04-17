"""
python-kasa wrapper: plug discovery, power reading, on/off control.

Architecture
------------
python-kasa is 100% async.  The rest of this project is synchronous.
PlugManager bridges the gap by owning a private asyncio event loop that
runs in a daemon thread ("kasa-loop").  All public methods are synchronous:
they submit coroutines to the background loop and block until they resolve.

Discovery
---------
Startup calls `discover()`, which broadcasts on the local network using
TP-Link's KLAP auth (KP125M requires a TP-Link account).  If broadcast
doesn't surface both expected aliases, we fall back to connecting directly
to the IP hints in config.py.

Power polling
-------------
After discovery, a background async task fires every PLUG_POLL_INTERVAL
seconds, calls `device.update()` on each found plug, and writes a fresh
PlugState into `_states`.  The main loop can call `state(alias)` at any
time to get the latest reading without blocking on I/O.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from kasa import (
    Credentials, Device, DeviceConfig, DeviceConnectionParameters,
    DeviceEncryptionType, DeviceFamily, Discover, Module, KasaException,
)

import config

log = logging.getLogger(__name__)

PLUG_POLL_INTERVAL = 5.0     # seconds between background power reads
DISCOVER_BROADCAST_TIMEOUT = 10  # seconds for UDP broadcast
_COMMAND_TIMEOUT = 6.0       # seconds to wait for turn_on/off to complete


@dataclass
class PlugState:
    alias: str
    is_on: bool
    power_w: float       # current draw in watts
    voltage_v: float     # volts (0 if unavailable)
    current_a: float     # amps (0 if unavailable)
    ts: float = field(default_factory=time.monotonic)


class PlugManager:
    """
    Thread-safe, synchronous wrapper around python-kasa's async API.

    Usage
    -----
        mgr = PlugManager()
        mgr.start()                       # spins up the background loop
        found = mgr.discover(timeout=15)  # blocks until plugs are found
        mgr.turn_on("light")
        print(mgr.state("light").power_w)
        mgr.stop()
    """

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._devices: dict[str, Device] = {}   # alias -> Device
        self._states: dict[str, PlugState] = {}
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._poll_task: Optional[asyncio.Task] = None
        self._credentials = Credentials(config.KASA_USERNAME, config.KASA_PASSWORD)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the private asyncio event loop in a daemon thread."""
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="kasa-loop"
        )
        self._thread.start()
        log.debug("PlugManager: background loop started")

    def stop(self) -> None:
        """Cancel polling and shut down the event loop."""
        self._running = False
        if self._poll_task is not None:
            self._loop.call_soon_threadsafe(self._poll_task.cancel)
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=4.0)

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _submit(self, coro, timeout: float = _COMMAND_TIMEOUT):
        """Submit a coroutine to the background loop and block for result."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def discover(self, timeout: float = 15.0) -> bool:
        """
        Find both plugs (lamp + fan) by alias.  Blocks up to `timeout` seconds.
        Returns True if both were found, False if only partial or none.

        Strategy:
          1. Broadcast UDP discover (finds everything on the subnet).
          2. For any alias not found via broadcast, try connecting directly
             to the IP hint from config.py.
        """
        try:
            found = self._submit(self._discover_async(), timeout=timeout)
        except Exception as exc:
            log.error("PlugManager.discover failed: %s", exc)
            return False

        expected = {config.LAMP_PLUG_ALIAS, config.FAN_PLUG_ALIAS}
        missing = expected - set(self._devices.keys())
        if missing:
            log.warning("PlugManager: plugs not found after discovery: %s", missing)
        else:
            log.info("PlugManager: all plugs found — starting power polling")
            self._submit(self._start_polling_async(), timeout=2.0)

        return len(missing) == 0

    async def _discover_async(self) -> None:
        """Broadcast discovery + IP-hint fallback."""
        target_aliases = {config.LAMP_PLUG_ALIAS, config.FAN_PLUG_ALIAS}
        ip_hints = {
            config.LAMP_PLUG_ALIAS: config.LAMP_PLUG_IP_HINT,
            config.FAN_PLUG_ALIAS: config.FAN_PLUG_IP_HINT,
        }

        log.info("PlugManager: broadcasting UDP discover (timeout=%ds)…",
                 DISCOVER_BROADCAST_TIMEOUT)
        try:
            found: dict[str, Device] = await Discover.discover(
                credentials=self._credentials,
                discovery_timeout=DISCOVER_BROADCAST_TIMEOUT,
            )
        except Exception as exc:
            log.warning("PlugManager: broadcast discover error: %s", exc)
            found = {}

        for ip, device in found.items():
            try:
                await device.update()  # alias is None until update() is called
            except Exception as exc:
                log.warning("  update failed for %s: %s", ip, exc)
                continue
            if device.alias in target_aliases:
                log.info("  discovered %s at %s (broadcast)", device.alias, ip)
                with self._lock:
                    self._devices[device.alias] = device

        # Fallback: direct IP for any alias still missing.
        # KP125M uses KLAP over HTTP port 80 — must specify connection_type explicitly,
        # otherwise DeviceConfig defaults to the legacy SmartHome protocol on port 9999.
        klap_params = DeviceConnectionParameters(
            device_family=DeviceFamily.SmartKasaPlug,
            encryption_type=DeviceEncryptionType.Klap,
            login_version=2,
        )
        for alias, ip in ip_hints.items():
            if alias not in self._devices:
                log.info("  %s not found via broadcast — trying IP hint %s (KLAP/80)", alias, ip)
                try:
                    cfg = DeviceConfig(
                        host=ip,
                        credentials=self._credentials,
                        connection_type=klap_params,
                        port_override=80,
                    )
                    device = await Device.connect(config=cfg)
                    await device.update()
                    log.info("  connected to %s at %s (IP hint)", device.alias, ip)
                    with self._lock:
                        self._devices[device.alias] = device
                except Exception as exc:
                    log.warning("  IP-hint connect for %s failed: %s", alias, exc)

    async def _start_polling_async(self) -> None:
        self._poll_task = asyncio.create_task(self._poll_loop())

    async def _poll_loop(self) -> None:
        """Background coroutine: refresh all plug states every PLUG_POLL_INTERVAL."""
        while self._running:
            with self._lock:
                devices = dict(self._devices)

            for alias, device in devices.items():
                try:
                    await device.update()
                    state = _extract_state(alias, device)
                    with self._lock:
                        self._states[alias] = state
                    log.debug("PlugManager: polled %s → %.1fW on=%s",
                              alias, state.power_w, state.is_on)
                except Exception as exc:
                    log.warning("PlugManager: poll error for %s: %s", alias, exc)

            await asyncio.sleep(PLUG_POLL_INTERVAL)

    # ------------------------------------------------------------------
    # Public synchronous API
    # ------------------------------------------------------------------

    def state(self, alias: str) -> Optional[PlugState]:
        """Return the most recently polled state for this plug, or None."""
        with self._lock:
            return self._states.get(alias)

    def all_states(self) -> dict[str, PlugState]:
        with self._lock:
            return dict(self._states)

    def is_available(self, alias: str) -> bool:
        with self._lock:
            return alias in self._devices

    def turn_on(self, alias: str) -> bool:
        """Send turn-on command.  Returns True if successful."""
        return self._toggle(alias, on=True)

    def turn_off(self, alias: str) -> bool:
        """Send turn-off command.  Returns True if successful."""
        return self._toggle(alias, on=False)

    def _toggle(self, alias: str, *, on: bool) -> bool:
        with self._lock:
            device = self._devices.get(alias)
        if device is None:
            log.warning("PlugManager._toggle: %s not found — skipping", alias)
            return False
        try:
            self._submit(self._toggle_async(device, on=on))
            return True
        except Exception as exc:
            log.error("PlugManager._toggle %s=%s failed: %s", alias, on, exc)
            return False

    async def _toggle_async(self, device: Device, *, on: bool) -> None:
        if on:
            await device.turn_on()
        else:
            await device.turn_off()
        # Refresh state immediately after toggling
        await device.update()
        alias = device.alias
        state = _extract_state(alias, device)
        with self._lock:
            self._states[alias] = state


# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------

def _extract_state(alias: str, device: Device) -> PlugState:
    """Pull is_on + energy readings from an already-updated Device."""
    power_w = 0.0
    voltage_v = 0.0
    current_a = 0.0

    if Module.Energy in device.modules:
        energy = device.modules[Module.Energy]
        try:
            power_w = float(energy.current_consumption or 0)
        except Exception:
            pass
        try:
            voltage_v = float(energy.voltage or 0)
        except Exception:
            pass
        try:
            current_a = float(energy.current or 0)
        except Exception:
            pass

    return PlugState(
        alias=alias,
        is_on=bool(device.is_on),
        power_w=power_w,
        voltage_v=voltage_v,
        current_a=current_a,
    )

"""Observational, producer-owned raw AC-energy evidence for PowerMind.

Only SHA-256 digests of physical identity inputs enter HA events. Identity uses
canonical JSON of configured host/port/unit and discovered SunSpec common
manufacturer/model/serial/device address. Capability uses canonical JSON of
the supported inverter DIDs, model length, AC-energy registers, raw bounds,
and the PowerMind-only signed scale-factor interpretation. These inputs are
intentionally fixed and independent of HA entity and sensor state.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from importlib.metadata import version
from uuid import uuid4

_LOGGER = logging.getLogger(__name__)

ACQUISITION_EVENT = "powermind_solaredge_ac_energy_acquisition"
FAILURE_EVENT = "powermind_solaredge_ac_energy_failure"
PRODUCER_VERSION = "4.0.3-powermind-acquisition.1"
ADAPTER_REVISION = "solaredge-raw-ac-v1"
PROFILE_REVISION = "solaredge-profile-v1"

DEFAULT_CAPABILITY = {
    "sunspec_inverter_dids": [101, 102, 103],
    "sunspec_inverter_length": 50,
    "register_wh": 40093,
    "register_sf": 40095,
    "raw_wh_max": 2**32 - 2,
    "sf_encoding": "signed-u16-twos-complement",
    "sf_min": -10,
    "sf_max": 10,
}


def _fingerprint(value: dict) -> str:
    canonical = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def normalize_scale_factor(value: object) -> int | None:
    """Interpret only PowerMind's current unsigned 16-bit register as signed."""
    if type(value) is not int or not 0 <= value <= 0xFFFF:
        return None
    return value - 0x10000 if value >= 0x8000 else value


def _runtime_versions() -> tuple[str, str, str]:
    return (version("modbus-connection"), version("tmodbus"), version("homeassistant"))


class PowerMindEvidenceProducer:
    """One source/hub lifecycle, one never-reused epoch and monotonic generation."""

    def __init__(
        self,
        hass,
        *,
        host: str,
        port: int,
        inverter_units,
        clock=None,
        epoch_factory=None,
        versions=None,
        capability: dict | None = None,
    ) -> None:
        self.hass = hass
        self.host = host
        self.port = port
        self.inverter_units = tuple(inverter_units)
        self.enabled = (
            len(self.inverter_units) == 1
            and type(self.inverter_units[0]) is int
            and self.inverter_units[0] > 0
        )
        self.epoch_id = str((epoch_factory or uuid4)())
        self.generation = 0
        self.clock = clock or (lambda: datetime.now(UTC))
        self.versions = versions or _runtime_versions
        self.capability = {**DEFAULT_CAPABILITY, **(capability or {})}
        self.capability_fingerprint = _fingerprint(self.capability)
        if not self.enabled:
            _LOGGER.warning(
                "PowerMind AC-energy evidence disabled: exactly one inverter unit is required"
            )

    def capture_time(self) -> datetime | None:
        """Capture a read boundary without interrupting an ordinary refresh."""
        try:
            return self.clock()
        except Exception:
            _LOGGER.exception("PowerMind AC-energy clock failed")
            return None

    def begin(self) -> tuple[int, datetime | None] | None:
        if not self.enabled:
            return None
        self.generation += 1
        return self.generation, self.capture_time()

    def _identity(self, inverter) -> tuple[str, bool]:
        unit = getattr(inverter, "inverter_unit_id", None)
        data = {
            "host": self.host,
            "port": self.port,
            "inverter_unit": unit,
            "manufacturer": getattr(inverter, "manufacturer", None),
            "model": getattr(inverter, "model", None),
            "serial": getattr(inverter, "serial", None),
            "device_address": getattr(inverter, "device_address", None),
        }
        valid = (
            type(unit) is int
            and unit > 0
            and unit == self.inverter_units[0]
            and isinstance(self.host, str)
            and bool(self.host)
            and type(self.port) is int
            and 0 < self.port <= 65535
            and all(
                isinstance(data[key], str) and bool(data[key].strip())
                for key in ("manufacturer", "model", "serial")
            )
        )
        return _fingerprint(data), valid

    @staticmethod
    def _iso(value: datetime | None) -> str | None:
        if not isinstance(value, datetime) or value.tzinfo is None:
            return None
        if value.utcoffset() is None:
            return None
        return value.astimezone(UTC).isoformat()

    def _publish(
        self,
        attempt,
        inverter,
        *,
        completed_at: datetime | None,
        failure_class: str,
        raw_wh=None,
        raw_sf=None,
        did=None,
        length=None,
    ) -> None:
        if attempt is None:
            return
        try:
            generation, started_at = attempt
            identity_fingerprint, identity_valid = self._identity(inverter)
            started = self._iso(started_at)
            completed = self._iso(completed_at)
            if started is None or completed is None:
                failure_class = "TIMING_ERROR"
                # A failure event still needs parseable UTC timestamps. These
                # fallbacks are not used for a valid acquisition.
                fallback = datetime.now(UTC).isoformat()
                started = started or fallback
                completed = completed or fallback
            elif completed_at < started_at:
                failure_class = "TIMING_ERROR"
            if not identity_valid and failure_class == "NONE":
                failure_class = "IDENTITY_ERROR"
            if failure_class == "NONE" and (did not in (101, 102, 103) or length != 50):
                failure_class = "IDENTITY_ERROR"
            if failure_class == "NONE" and (
                type(raw_wh) is not int
                or not 0 <= raw_wh <= 2**32 - 2
                or type(raw_sf) is not int
                or not -10 <= raw_sf <= 10
            ):
                failure_class = "RAW_FIELD_ERROR"
            modbus_version, tmodbus_version, ha_version = self.versions()
            payload = {
                "source_id": "solaredge_pv",
                "epoch_id": self.epoch_id,
                "generation": generation,
                "source_identity_fingerprint": identity_fingerprint,
                "capability_fingerprint": self.capability_fingerprint,
                "producer_version": PRODUCER_VERSION,
                "modbus_connection_version": modbus_version,
                "tmodbus_version": tmodbus_version,
                "ha_version": ha_version,
                "adapter_revision": ADAPTER_REVISION,
                "profile_revision": PROFILE_REVISION,
                "raw_ac_energy_wh": raw_wh,
                "raw_ac_energy_sf": raw_sf,
                "sunspec_did": did,
                "sunspec_length": length,
                "inverter_unit": getattr(inverter, "inverter_unit_id", None),
                "read_started_at": started,
                "read_completed_at": completed,
                "acquisition_valid": failure_class == "NONE",
                "failure_class": failure_class,
            }
            event = ACQUISITION_EVENT if failure_class == "NONE" else FAILURE_EVENT
            self.hass.bus.async_fire(event, payload)
        except Exception:
            _LOGGER.exception("PowerMind AC-energy evidence publication failed")

    def publish_acquisition(
        self, attempt, inverter, component, *, completed_at=None
    ) -> None:
        """Publish only values from the just-completed InverterData update."""
        if attempt is None:
            return
        completed_at = completed_at if completed_at is not None else self.capture_time()
        try:
            raw_wh = component.AC_Energy_WH
            raw_sf = normalize_scale_factor(component.AC_Energy_WH_SF)
            did = component.C_SunSpec_DID
            length = component.C_SunSpec_Length
        except Exception:  # noqa: BLE001 - evidence must never break a refresh
            self._publish(
                attempt,
                inverter,
                completed_at=completed_at,
                failure_class="RAW_FIELD_ERROR",
            )
            return
        self._publish(
            attempt,
            inverter,
            completed_at=completed_at,
            failure_class="NONE",
            raw_wh=raw_wh,
            raw_sf=raw_sf,
            did=did,
            length=length,
        )

    def publish_failure(
        self,
        attempt,
        inverter,
        failure_class: str,
        *,
        completed_at=None,
        component=None,
    ) -> None:
        """Never reuse a counter or scale factor from a preceding read."""
        if attempt is None:
            return
        completed_at = completed_at if completed_at is not None else self.capture_time()
        try:
            did = getattr(component, "C_SunSpec_DID", None)
            length = getattr(component, "C_SunSpec_Length", None)
        except Exception:
            _LOGGER.exception("PowerMind AC-energy identity snapshot failed")
            did = length = None
        self._publish(
            attempt,
            inverter,
            completed_at=completed_at,
            failure_class=failure_class,
            did=did,
            length=length,
        )

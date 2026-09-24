"""PowerMind raw AC-energy producer contract tests."""

from __future__ import annotations

import ast
import importlib.util
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from modbus_connection.exceptions import (
    ModbusConnectionError,
    ModbusProtocolError,
    ModbusTimeoutError,
)

MODULE = (
    Path(__file__).parents[1]
    / "custom_components/solaredge_modbus_multi/powermind_evidence.py"
)
FIELDS = {
    "source_id",
    "epoch_id",
    "generation",
    "source_identity_fingerprint",
    "capability_fingerprint",
    "producer_version",
    "modbus_connection_version",
    "tmodbus_version",
    "ha_version",
    "adapter_revision",
    "profile_revision",
    "raw_ac_energy_wh",
    "raw_ac_energy_sf",
    "sunspec_did",
    "sunspec_length",
    "inverter_unit",
    "read_started_at",
    "read_completed_at",
    "acquisition_valid",
    "failure_class",
}
START = datetime(2026, 9, 24, 10, 0, tzinfo=UTC)


@pytest.fixture
def evidence_module():
    assert MODULE.is_file(), "PowerMind evidence producer module is missing"
    spec = importlib.util.spec_from_file_location("powermind_evidence", MODULE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def producer_factory(evidence_module):
    def make(
        *, identity=None, clock=None, units=(1,), epoch="epoch-1", capability=None
    ):
        events = []
        hass = SimpleNamespace(
            bus=SimpleNamespace(
                async_fire=lambda event_type, data: events.append((event_type, data))
            )
        )
        inverter = SimpleNamespace(
            inverter_unit_id=1,
            manufacturer="Synthetic Manufacturer",
            model="Synthetic Model",
            serial="SYNTHETIC-SERIAL",
            device_address="synthetic-address",
        )
        if identity:
            for key, value in identity.items():
                setattr(inverter, key, value)
        producer = evidence_module.PowerMindEvidenceProducer(
            hass,
            host="example.invalid",
            port=1502,
            inverter_units=units,
            clock=clock or iter([START, START + timedelta(milliseconds=10)]).__next__,
            epoch_factory=lambda: epoch,
            versions=lambda: ("4.10.0", "0.6.2", "2026.9.3"),
            capability=capability,
        )
        return producer, inverter, events

    return make


def acquire(producer, inverter, *, wh=0, sf=0, did=101, length=50):
    attempt = producer.begin()
    component = SimpleNamespace(
        AC_Energy_WH=wh,
        AC_Energy_WH_SF=sf,
        C_SunSpec_DID=did,
        C_SunSpec_Length=length,
    )
    producer.publish_acquisition(attempt, inverter, component)


@pytest.mark.parametrize(
    "encoded,normalized",
    [
        (0x0000, 0),
        (0x0001, 1),
        (0xFFFF, -1),
        (0xFFFE, -2),
        (0x8000, -32768),
    ],
)
def test_signed_scale_factor_normalization(evidence_module, encoded, normalized):
    assert evidence_module.normalize_scale_factor(encoded) == normalized


@pytest.mark.parametrize("sf", [0, 1, 0xFFFF, 0xFFFE, 10, 0xFFF6])
def test_valid_raw_scale_factors_publish_success(producer_factory, sf):
    producer, inverter, events = producer_factory()
    acquire(producer, inverter, sf=sf)
    event_type, payload = events[0]
    assert event_type == "powermind_solaredge_ac_energy_acquisition"
    assert payload["raw_ac_energy_sf"] == (sf if sf <= 10 else sf - 0x10000)
    assert payload["acquisition_valid"] is True
    assert payload["failure_class"] == "NONE"
    assert set(payload) == FIELDS
    assert not {"entity_id", "state", "last_updated", "sensor_value"} & payload.keys()


@pytest.mark.parametrize("sf", [0x8000, 11, 0xFFF5, None, -32768])
def test_invalid_scale_factors_publish_raw_field_failure(producer_factory, sf):
    producer, inverter, events = producer_factory()
    acquire(producer, inverter, sf=sf)
    event_type, payload = events[0]
    assert event_type == "powermind_solaredge_ac_energy_failure"
    assert payload["failure_class"] == "RAW_FIELD_ERROR"
    assert payload["acquisition_valid"] is False
    assert set(payload) == FIELDS


@pytest.mark.parametrize("wh", [0, 2**32 - 2])
def test_raw_energy_boundary_is_accepted_without_scaling(producer_factory, wh):
    producer, inverter, events = producer_factory()
    acquire(producer, inverter, wh=wh, sf=0xFFFE)
    assert events[0][1]["raw_ac_energy_wh"] == wh
    assert events[0][1]["raw_ac_energy_sf"] == -2
    assert events[0][1]["acquisition_valid"] is True


@pytest.mark.parametrize("wh", [2**32 - 1, -1, None, 1.5])
def test_invalid_raw_energy_fails_closed(producer_factory, wh):
    producer, inverter, events = producer_factory()
    acquire(producer, inverter, wh=wh)
    assert events[0][0] == "powermind_solaredge_ac_energy_failure"
    assert events[0][1]["failure_class"] == "RAW_FIELD_ERROR"


@pytest.mark.parametrize("did", [101, 102, 103])
def test_supported_sunspec_models_are_accepted(producer_factory, did):
    producer, inverter, events = producer_factory()
    acquire(producer, inverter, did=did)
    assert events[0][1]["sunspec_did"] == did
    assert events[0][1]["acquisition_valid"] is True


@pytest.mark.parametrize("did,length", [(100, 50), (101, 49), (101, 51)])
def test_invalid_sunspec_identity_fails_closed(producer_factory, did, length):
    producer, inverter, events = producer_factory()
    acquire(producer, inverter, did=did, length=length)
    assert events[0][0] == "powermind_solaredge_ac_energy_failure"
    assert events[0][1]["failure_class"] == "IDENTITY_ERROR"


def test_identity_and_capability_fingerprints_are_deterministic(producer_factory):
    first, inverter, first_events = producer_factory()
    acquire(first, inverter)
    second, inverter, second_events = producer_factory()
    acquire(second, inverter)
    for field in ("source_identity_fingerprint", "capability_fingerprint"):
        assert first_events[0][1][field] == second_events[0][1][field]
        assert first_events[0][1][field].startswith("sha256:")
        assert len(first_events[0][1][field]) == 71
    changed, inverter, changed_events = producer_factory(identity={"serial": "OTHER"})
    acquire(changed, inverter)
    assert (
        changed_events[0][1]["source_identity_fingerprint"]
        != first_events[0][1]["source_identity_fingerprint"]
    )
    changed, inverter, changed_events = producer_factory(
        capability={"register_wh": 40094}
    )
    acquire(changed, inverter)
    assert (
        changed_events[0][1]["capability_fingerprint"]
        != first_events[0][1]["capability_fingerprint"]
    )
    assert "SYNTHETIC-SERIAL" not in str(first_events)
    assert "example.invalid" not in str(first_events)


def test_failed_attempt_consumes_generation_and_never_reuses_raw(producer_factory):
    times = iter(START + timedelta(seconds=i) for i in range(6))
    producer, inverter, events = producer_factory(clock=times.__next__)
    acquire(producer, inverter, wh=100)
    attempt = producer.begin()
    producer.publish_failure(attempt, inverter, "READ_ERROR")
    acquire(producer, inverter, wh=101)
    assert [payload["generation"] for _, payload in events] == [1, 2, 3]
    assert events[0][1]["epoch_id"] == events[2][1]["epoch_id"]
    assert events[1][1]["raw_ac_energy_wh"] is None
    assert events[1][1]["raw_ac_energy_sf"] is None
    assert events[1][1]["failure_class"] == "READ_ERROR"
    assert set(events[1][1]) == FIELDS


def test_new_producer_uses_new_epoch(evidence_module):
    first = evidence_module.PowerMindEvidenceProducer(
        SimpleNamespace(bus=SimpleNamespace(async_fire=lambda *_: None)),
        host="example.invalid",
        port=1502,
        inverter_units=(1,),
    )
    second = evidence_module.PowerMindEvidenceProducer(
        SimpleNamespace(bus=SimpleNamespace(async_fire=lambda *_: None)),
        host="example.invalid",
        port=1502,
        inverter_units=(1,),
    )
    assert first.epoch_id != second.epoch_id


def test_invalid_timing_publishes_typed_failure(producer_factory):
    times = iter([START, START - timedelta(seconds=1)])
    producer, inverter, events = producer_factory(clock=times.__next__)
    acquire(producer, inverter)
    assert events[0][0] == "powermind_solaredge_ac_energy_failure"
    assert events[0][1]["failure_class"] == "TIMING_ERROR"


def test_clock_exception_publishes_timing_failure_without_interrupting_read(
    producer_factory,
):
    calls = iter([RuntimeError("clock unavailable"), START])

    def clock():
        value = next(calls)
        if isinstance(value, Exception):
            raise value
        return value

    producer, inverter, events = producer_factory(clock=clock)
    acquire(producer, inverter)
    assert events[0][0] == "powermind_solaredge_ac_energy_failure"
    assert events[0][1]["failure_class"] == "TIMING_ERROR"
    assert datetime.fromisoformat(events[0][1]["read_started_at"]).tzinfo is not None


def test_runtime_versions_are_read_from_package_metadata(evidence_module, monkeypatch):
    seen = []

    def package_version(name):
        seen.append(name)
        return {
            "modbus-connection": "4.10.0",
            "tmodbus": "0.6.2",
            "homeassistant": "2026.9.3",
        }[name]

    monkeypatch.setattr(evidence_module, "version", package_version)
    assert evidence_module._runtime_versions() == ("4.10.0", "0.6.2", "2026.9.3")
    assert seen == ["modbus-connection", "tmodbus", "homeassistant"]


def test_publication_error_is_contained(producer_factory):
    producer, inverter, events = producer_factory()

    def broken_fire(*_):
        raise RuntimeError("bus unavailable")

    producer.hass.bus.async_fire = broken_fire
    acquire(producer, inverter)
    assert events == []


def test_failure_identity_snapshot_error_is_contained(producer_factory):
    producer, inverter, events = producer_factory()

    class BrokenComponent:
        @property
        def C_SunSpec_DID(self):
            raise RuntimeError("field unreadable")

    attempt = producer.begin()
    producer.publish_failure(
        attempt, inverter, "IDENTITY_ERROR", component=BrokenComponent()
    )
    assert events[0][1]["failure_class"] == "IDENTITY_ERROR"
    assert events[0][1]["sunspec_did"] is None


def test_multi_inverter_config_does_not_publish_single_source_evidence(
    producer_factory,
):
    producer, inverter, events = producer_factory(units=(1, 2))
    acquire(producer, inverter)
    assert events == []


def test_generic_scale_factor_decoder_remains_unsigned():
    components = MODULE.with_name("components.py").read_text(encoding="utf-8")
    assert "AC_Energy_WH_SF = integer(40095, signed=False)" in components


@pytest.fixture
def inverter_read_method():
    """Execute the real hub method with only unrelated HA imports isolated."""
    tree = ast.parse(MODULE.with_name("hub.py").read_text(encoding="utf-8"))
    inverter_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SolarEdgeInverter"
    )
    method = next(
        node
        for node in inverter_class.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "read_modbus_data"
    )
    source = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))

    class DeviceInvalid(Exception):
        pass

    namespace = {
        "_LOGGER": logging.getLogger(__name__),
        "_log_component_fields": lambda *_: None,
        "SunSpecNotImpl": SimpleNamespace(UINT16=0xFFFF),
        "DeviceInvalid": DeviceInvalid,
        "ModbusConnectionError": ModbusConnectionError,
        "ModbusProtocolError": ModbusProtocolError,
        "ModbusTimeoutError": ModbusTimeoutError,
    }
    exec(  # noqa: S102 - execute only the checked-in method under test
        compile(source, str(MODULE.with_name("hub.py")), "exec"), namespace
    )
    return namespace["read_modbus_data"], DeviceInvalid


def make_hook_inverter(producer, component_update):
    inverter = SimpleNamespace(
        inverter_unit_id=1,
        manufacturer="Synthetic Manufacturer",
        model="Synthetic Model",
        serial="SYNTHETIC-SERIAL",
        device_address="synthetic-address",
        inverter_common=object(),
        inverter_data=SimpleNamespace(
            AC_Energy_WH=100,
            AC_Energy_WH_SF=0xFFFF,
            C_SunSpec_DID=101,
            C_SunSpec_Length=50,
        ),
        use_mmppt_units=False,
    )
    inverter.hub = SimpleNamespace(
        powermind_evidence=producer,
        component_update=component_update,
        option_detect_extras=False,
        option_site_limit_control=False,
        option_storage_control=False,
    )
    return inverter


@pytest.mark.asyncio
async def test_hub_hook_surrounds_current_update_and_does_not_write(
    producer_factory,
    inverter_read_method,
):
    trace = []
    timestamps = iter([START, START + timedelta(milliseconds=10)])

    def clock():
        trace.append("clock")
        return next(timestamps)

    producer, _, events = producer_factory(clock=clock)

    async def component_update(_, component):
        trace.append("common" if component is inverter.inverter_common else "data")
        if component is inverter.inverter_data:
            component.AC_Energy_WH = 123

    inverter = make_hook_inverter(producer, component_update)
    read_method, _ = inverter_read_method
    await read_method(inverter)
    assert trace == ["common", "clock", "data", "clock"]
    assert events[0][1]["raw_ac_energy_wh"] == 123
    assert events[0][1]["read_started_at"] == START.isoformat()
    assert (
        events[0][1]["read_completed_at"]
        == (START + timedelta(milliseconds=10)).isoformat()
    )


@pytest.mark.asyncio
async def test_hub_hook_failed_read_never_reuses_stale_raw(
    producer_factory,
    inverter_read_method,
):
    times = iter(START + timedelta(seconds=i) for i in range(4))
    producer, _, events = producer_factory(clock=times.__next__)
    original = ModbusConnectionError("link down")

    async def component_update(_, component):
        if component is inverter.inverter_data:
            raise original

    inverter = make_hook_inverter(producer, component_update)
    read_method, _ = inverter_read_method
    with pytest.raises(ModbusConnectionError) as raised:
        await read_method(inverter)
    assert raised.value.__cause__ is original
    assert len(events) == 1
    assert events[0][0] == "powermind_solaredge_ac_energy_failure"
    assert events[0][1]["raw_ac_energy_wh"] is None
    assert events[0][1]["raw_ac_energy_sf"] is None


@pytest.mark.asyncio
async def test_hub_hook_invalid_did_preserves_device_invalid(
    producer_factory,
    inverter_read_method,
):
    producer, _, events = producer_factory()

    async def component_update(*_):
        return None

    inverter = make_hook_inverter(producer, component_update)
    inverter.inverter_data.C_SunSpec_DID = 100
    read_method, device_invalid = inverter_read_method
    with pytest.raises(device_invalid):
        await read_method(inverter)
    assert events[0][1]["failure_class"] == "IDENTITY_ERROR"
    assert events[0][1]["acquisition_valid"] is False


@pytest.mark.asyncio
async def test_hub_hook_bus_failure_does_not_break_solar_refresh(
    producer_factory,
    inverter_read_method,
):
    producer, _, events = producer_factory()
    producer.hass.bus.async_fire = lambda *_: (_ for _ in ()).throw(
        RuntimeError("bus failure")
    )

    async def component_update(*_):
        return None

    inverter = make_hook_inverter(producer, component_update)
    read_method, _ = inverter_read_method
    await read_method(inverter)
    assert events == []

"""Tests for _check_dependency_versions()

Called from async_setup_entry via hass.async_add_executor_job before
modbus_connection or .hub are imported, so an old or missing install fails
with a ConfigEntryError message we can define.
"""

import asyncio
import importlib.metadata
import threading
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from homeassistant.const import CONF_HOST, CONF_NAME, CONF_PORT
from homeassistant.const import __version__ as HA_VERSION
from homeassistant.exceptions import ConfigEntryError

import custom_components.solaredge_modbus_multi as integration
from custom_components.solaredge_modbus_multi import _check_dependency_versions
from custom_components.solaredge_modbus_multi.const import DOMAIN, ConfName

_MISSING = object()


def _patched_versions(tmodbus, modbus_connection):
    versions = {"tmodbus": tmodbus, "modbus_connection": modbus_connection}

    def fake_version(package):
        if versions[package] is _MISSING:
            raise importlib.metadata.PackageNotFoundError(package)
        return versions[package]

    return patch("importlib.metadata.version", side_effect=fake_version)


class TestCheckDependencyVersions:
    def test_passes_when_both_versions_meet_the_floor(self):
        with _patched_versions(tmodbus="0.6.1", modbus_connection="4.10.0"):
            _check_dependency_versions()

    def test_passes_when_both_versions_are_newer(self):
        with _patched_versions(tmodbus="99.0.0", modbus_connection="99.0.0"):
            _check_dependency_versions()

    def test_old_tmodbus_raises(self):
        with (
            _patched_versions(tmodbus="0.6.0", modbus_connection="99.0.0"),
            pytest.raises(ConfigEntryError, match="tmodbus version must be at least"),
        ):
            _check_dependency_versions()

    def test_old_modbus_connection_raises(self):
        with (
            _patched_versions(tmodbus="99.0.0", modbus_connection="4.9.0"),
            pytest.raises(
                ConfigEntryError, match="modbus-connection version must be at least"
            ),
        ):
            _check_dependency_versions()

    def test_tmodbus_checked_before_modbus_connection(self):
        with (
            _patched_versions(tmodbus="0.1.0", modbus_connection="0.1.0"),
            pytest.raises(ConfigEntryError, match="tmodbus version must be at least"),
        ):
            _check_dependency_versions()

    def test_missing_tmodbus_raises(self):
        with (
            _patched_versions(tmodbus=_MISSING, modbus_connection="99.0.0"),
            pytest.raises(ConfigEntryError, match="tmodbus is not installed"),
        ):
            _check_dependency_versions()

    def test_missing_modbus_connection_raises(self):
        with (
            _patched_versions(tmodbus="99.0.0", modbus_connection=_MISSING),
            pytest.raises(ConfigEntryError, match="modbus-connection is not installed"),
        ):
            _check_dependency_versions()

    def test_missing_tmodbus_checked_before_modbus_connection(self):
        with (
            _patched_versions(tmodbus=_MISSING, modbus_connection=_MISSING),
            pytest.raises(ConfigEntryError, match="tmodbus is not installed"),
        ):
            _check_dependency_versions()


@pytest.mark.asyncio
async def test_setup_resolves_versions_in_executor_once_before_evidence_publication(
    monkeypatch,
    tmp_path,
):
    import modbus_connection.tmodbus as modbus_transport

    import custom_components.solaredge_modbus_multi.hub  # noqa: F401 - preload imports

    event_loop_thread = threading.get_ident()
    lookups = []
    installed = {"tmodbus": "0.6.2", "modbus_connection": "4.12.1"}

    def package_version(name):
        lookups.append((name, threading.get_ident()))
        return installed[name]

    monkeypatch.setattr(importlib.metadata, "version", package_version)

    class FakeConnection:
        def __init__(self, *_args, **_kwargs):
            pass

        def close(self):
            pass

    class FakeCoordinator:
        def __init__(self, *_args):
            pass

        async def async_config_entry_first_refresh(self):
            pass

    async def forward_entry_setups(*_args):
        pass

    class FakeHass:
        def __init__(self):
            self.data = {DOMAIN: {"yaml": {}}}
            self.config = SimpleNamespace(
                path=lambda *parts: str(tmp_path.joinpath(*parts))
            )
            self.events = []
            self.bus = SimpleNamespace(
                async_fire=lambda event_type, data: self.events.append(
                    (event_type, data)
                )
            )
            self.config_entries = SimpleNamespace(
                async_forward_entry_setups=forward_entry_setups
            )
            self.executor_jobs = 0

        async def async_add_executor_job(self, function):
            self.executor_jobs += 1
            return await asyncio.to_thread(function)

    monkeypatch.setattr(modbus_transport, "ModbusConnection", FakeConnection)
    monkeypatch.setattr(integration, "SolarEdgeCoordinator", FakeCoordinator)
    hass = FakeHass()
    entry = SimpleNamespace(
        entry_id="synthetic-entry",
        data={
            CONF_NAME: "SolarEdge",
            CONF_HOST: "example.invalid",
            CONF_PORT: 1502,
            ConfName.DEVICE_LIST: [1],
        },
        options={},
        async_on_unload=lambda _callback: None,
        add_update_listener=lambda _callback: lambda: None,
    )

    assert await integration.async_setup_entry(hass, entry) is True
    producer = hass.data[DOMAIN][entry.entry_id]["hub"].powermind_evidence
    assert producer is not None
    inverter = SimpleNamespace(
        inverter_unit_id=1,
        manufacturer="Synthetic Manufacturer",
        model="Synthetic Model",
        serial="SYNTHETIC-SERIAL",
        device_address="synthetic-address",
    )
    component = SimpleNamespace(
        AC_Energy_WH=123,
        AC_Energy_WH_SF=0,
        C_SunSpec_DID=101,
        C_SunSpec_Length=50,
    )
    times = iter(
        datetime(2026, 9, 24, tzinfo=UTC) + timedelta(seconds=i) for i in range(4)
    )
    producer.clock = times.__next__
    for _ in range(2):
        producer.publish_acquisition(producer.begin(), inverter, component)

    assert hass.executor_jobs == 2
    assert [name for name, _ in lookups] == ["tmodbus", "modbus_connection"]
    assert {thread_id for _, thread_id in lookups}.isdisjoint({event_loop_thread})
    assert len(hass.events) == 2
    replay = hass.data[DOMAIN][entry.entry_id]["evidence_replay"]
    assert not hasattr(replay, "append")
    page = replay.page(after=None)
    assert page["protocol_revision"] == "solaredge-evidence-replay-v1"
    assert [record["payload"]["generation"] for record in page["records"]] == [1, 2]
    for _, payload in hass.events:
        assert (
            payload["modbus_connection_version"],
            payload["tmodbus_version"],
            payload["ha_version"],
        ) == ("4.12.1", "0.6.2", HA_VERSION)

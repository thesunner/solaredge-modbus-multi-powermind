"""Exercise producer durability without Home Assistant test dependencies."""

from __future__ import annotations

import asyncio
import runpy
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
COMPONENT = ROOT / "custom_components" / "solaredge_modbus_multi"


def test_v404_producer_commits_journal_before_wake_event(tmp_path: Path) -> None:
    evidence = runpy.run_path(str(COMPONENT / "powermind_evidence.py"))
    journal_module = runpy.run_path(str(COMPONENT / "powermind_journal.py"))
    assert evidence["PRODUCER_VERSION"] == "4.0.4-powermind-acquisition.3"

    journal = journal_module["EvidenceJournal"].open(tmp_path / "journal.sqlite3")
    fired = []

    class Bus:
        def async_fire(self, event_type: str, payload: dict) -> None:
            page = journal.reader().page(after=None, limit=5)
            assert page["highwater"] == 1
            assert len(page["records"]) == 1
            record = page["records"][0]
            assert set(record) == {"cursor", "event_type", "payload"}
            assert record["event_type"] == event_type
            assert record["payload"] == payload
            fired.append(event_type)

    class Hass:
        bus = Bus()

        async def async_add_executor_job(self, function, *args):
            return function(*args)

    inverter = SimpleNamespace(
        inverter_unit_id=1,
        manufacturer="SolarEdge",
        model="qualified-model",
        serial="qualified-serial",
        device_address="1",
    )
    component = SimpleNamespace(
        AC_Energy_WH=123456,
        AC_Energy_WH_SF=0,
        C_SunSpec_DID=101,
        C_SunSpec_Length=50,
    )
    producer = evidence["PowerMindEvidenceProducer"](
        Hass(),
        host="192.0.2.1",
        port=1502,
        inverter_units=(1,),
        journal=journal,
        runtime_versions=("4.10.0", "0.6.2", "2026.9.3"),
    )

    async def publish() -> None:
        await producer.publish_acquisition(producer.begin(), inverter, component)

    try:
        asyncio.run(publish())
        assert fired == [evidence["ACQUISITION_EVENT"]]
    finally:
        journal.close()

    reopened = journal_module["EvidenceJournal"].open(tmp_path / "journal.sqlite3")
    try:
        page = reopened.reader().page(after=None, limit=5)
        assert page["protocol_revision"] == "solaredge-evidence-replay-v1"
        assert page["highwater"] == 1
        assert len(page["records"]) == 1
        assert set(page["records"][0]) == {"cursor", "event_type", "payload"}
        assert (
            page["records"][0]["payload"]["producer_version"]
            == evidence["PRODUCER_VERSION"]
        )
    finally:
        reopened.close()

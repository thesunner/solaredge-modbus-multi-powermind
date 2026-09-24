"""Durable producer replay is ordered, bounded, and fail closed."""

from __future__ import annotations

import importlib.util
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

MODULE = (
    Path(__file__).parents[1]
    / "custom_components/solaredge_modbus_multi/powermind_journal.py"
)
spec = importlib.util.spec_from_file_location("powermind_journal", MODULE)
assert spec is not None and spec.loader is not None
journal_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(journal_module)
EvidenceJournal = journal_module.EvidenceJournal
InvalidReplayCursor = journal_module.InvalidReplayCursor
JournalConflict = journal_module.JournalConflict
ReplayCursorExpired = journal_module.ReplayCursorExpired

ACQUISITION = "powermind_solaredge_ac_energy_acquisition"
FAILURE = "powermind_solaredge_ac_energy_failure"
START = datetime(2026, 9, 24, 10, 0, tzinfo=UTC)


def event(generation, *, epoch="epoch-one", failure=False, raw=100):
    at = (START + timedelta(minutes=generation)).isoformat()
    return {
        "source_id": "solaredge_pv",
        "epoch_id": epoch,
        "generation": generation,
        "source_identity_fingerprint": "sha256:" + "a" * 64,
        "capability_fingerprint": "sha256:" + "b" * 64,
        "producer_version": "4.0.3-powermind-acquisition.3",
        "modbus_connection_version": "4.10.0",
        "tmodbus_version": "0.6.2",
        "ha_version": "2026.9.3",
        "adapter_revision": "solaredge-raw-ac-v1",
        "profile_revision": "solaredge-profile-v1",
        "raw_ac_energy_wh": None if failure else raw,
        "raw_ac_energy_sf": None if failure else 0,
        "sunspec_did": 101,
        "sunspec_length": 50,
        "inverter_unit": 1,
        "read_started_at": at,
        "read_completed_at": at,
        "acquisition_valid": not failure,
        "failure_class": "READ_ERROR" if failure else "NONE",
    }


def test_generation_one_replays_without_any_bus_listener(tmp_path):
    journal = EvidenceJournal.open(tmp_path / "evidence.sqlite3")
    try:
        journal.start_epoch("epoch-one")
        journal.append(ACQUISITION, event(1))

        page = journal.page(after=None, limit=10)
        assert page["protocol_revision"] == "solaredge-evidence-replay-v1"
        assert page["current_epoch"] == "epoch-one"
        assert page["retained_floor"] == 0
        assert page["highwater"] == 1
        assert [record["payload"]["generation"] for record in page["records"]] == [1]
        assert page["records"][0]["event_type"] == ACQUISITION
        assert page["next_cursor"] == page["records"][0]["cursor"]
        assert page["has_more"] is False
    finally:
        journal.close()


def test_success_failure_success_replay_in_exact_generation_order(tmp_path):
    journal = EvidenceJournal.open(tmp_path / "evidence.sqlite3")
    try:
        journal.start_epoch("epoch-one")
        journal.append(ACQUISITION, event(1))
        journal.append(FAILURE, event(2, failure=True))
        journal.append(ACQUISITION, event(3, raw=101))

        page = journal.page(after=None, limit=2)
        assert [record["payload"]["generation"] for record in page["records"]] == [1, 2]
        assert [record["event_type"] for record in page["records"]] == [
            ACQUISITION,
            FAILURE,
        ]
        assert page["records"][1]["payload"]["raw_ac_energy_wh"] is None
        assert page["records"][1]["payload"]["raw_ac_energy_sf"] is None
        assert page["has_more"] is True

        continuation = journal.page(after=page["next_cursor"], limit=2)
        assert [
            record["payload"]["generation"] for record in continuation["records"]
        ] == [3]
        assert continuation["has_more"] is False
    finally:
        journal.close()


def test_restart_opens_new_epoch_without_losing_retained_history(tmp_path):
    path = tmp_path / "evidence.sqlite3"
    original = EvidenceJournal.open(path)
    original.start_epoch("epoch-one")
    original.append(ACQUISITION, event(1))
    original.close()

    reopened = EvidenceJournal.open(path)
    try:
        reopened.start_epoch("epoch-two")
        reopened.append(ACQUISITION, event(1, epoch="epoch-two"))
        page = reopened.page(after=None, limit=10)
        assert page["current_epoch"] == "epoch-two"
        assert [
            (r["payload"]["epoch_id"], r["payload"]["generation"])
            for r in page["records"]
        ] == [("epoch-one", 1), ("epoch-two", 1)]
        with pytest.raises(JournalConflict):
            reopened.start_epoch("epoch-one")
    finally:
        reopened.close()


def test_same_cursor_is_idempotent_and_invalid_cursor_is_rejected(tmp_path):
    journal = EvidenceJournal.open(tmp_path / "evidence.sqlite3")
    try:
        journal.start_epoch("epoch-one")
        journal.append(ACQUISITION, event(1))
        first = journal.page(after=None, limit=10)
        after = first["next_cursor"]
        assert journal.page(after=after, limit=10) == journal.page(
            after=after, limit=10
        )
        assert journal.page(after=after, limit=10)["records"] == []
        with pytest.raises(InvalidReplayCursor):
            journal.page(after="forged-cursor", limit=10)
    finally:
        journal.close()


def test_capacity_expiry_fails_closed_and_growth_is_bounded(tmp_path):
    journal = EvidenceJournal.open(tmp_path / "evidence.sqlite3", capacity=2)
    try:
        journal.start_epoch("epoch-one")
        journal.append(ACQUISITION, event(1))
        old_cursor = journal.page(after=None, limit=1)["next_cursor"]
        journal.append(ACQUISITION, event(2))
        journal.append(ACQUISITION, event(3))
        page = journal.page(after=old_cursor, limit=10)
        assert page["retained_floor"] == 1
        assert page["highwater"] == 3
        assert [record["payload"]["generation"] for record in page["records"]] == [2, 3]
        with pytest.raises(ReplayCursorExpired):
            journal.page(after=None, limit=10)
        journal.append(ACQUISITION, event(4))
        with pytest.raises(ReplayCursorExpired):
            journal.page(after=old_cursor, limit=10)
    finally:
        journal.close()


def test_conflicting_immutable_record_and_missing_generation_fail_closed(tmp_path):
    journal = EvidenceJournal.open(tmp_path / "evidence.sqlite3")
    try:
        journal.start_epoch("epoch-one")
        journal.append(ACQUISITION, event(1))
        with pytest.raises(JournalConflict):
            journal.append(ACQUISITION, event(1, raw=999))
        with pytest.raises(JournalConflict):
            journal.append(ACQUISITION, event(3))
        assert journal.page(after=None, limit=10)["highwater"] == 1
    finally:
        journal.close()


def test_records_expose_only_share_safe_event_fields(tmp_path):
    journal = EvidenceJournal.open(tmp_path / "evidence.sqlite3")
    try:
        journal.start_epoch("epoch-one")
        journal.append(ACQUISITION, event(1))
        page = journal.page(after=None, limit=10)
        payload = page["records"][0]["payload"]
        assert len(payload) == 20
        assert (
            not {"entity_id", "state", "serial", "host", "ip", "token"} & payload.keys()
        )
        assert "SYNTHETIC-SERIAL" not in str(page)
    finally:
        journal.close()


def test_nested_event_value_is_rejected_before_it_can_enter_replay(tmp_path):
    journal = EvidenceJournal.open(tmp_path / "evidence.sqlite3")
    try:
        journal.start_epoch("epoch-one")
        unsafe = event(1)
        unsafe["raw_ac_energy_wh"] = {"token": "secret"}
        with pytest.raises(JournalConflict):
            journal.append(FAILURE, unsafe)
        assert journal.page(after=None)["records"] == []
    finally:
        journal.close()


def test_string_raw_counter_is_rejected_before_it_can_enter_replay(tmp_path):
    journal = EvidenceJournal.open(tmp_path / "evidence.sqlite3")
    try:
        journal.start_epoch("epoch-one")
        unsafe = event(1)
        unsafe["raw_ac_energy_wh"] = "secret"
        with pytest.raises(JournalConflict):
            journal.append(FAILURE, unsafe)
        assert journal.page(after=None)["records"] == []
    finally:
        journal.close()


def test_missing_retained_sequence_fails_replay_closed(tmp_path):
    journal = EvidenceJournal.open(tmp_path / "evidence.sqlite3")
    try:
        journal.start_epoch("epoch-one")
        journal.append(ACQUISITION, event(1))
        journal.append(ACQUISITION, event(2))
        journal._db.execute("DELETE FROM records WHERE sequence = 1")
        journal._db.commit()
        with pytest.raises(JournalConflict):
            journal.page(after=None)
    finally:
        journal.close()

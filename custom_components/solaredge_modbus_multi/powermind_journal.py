"""Durable, bounded replay of producer-owned PowerMind evidence."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import sqlite3
from functools import wraps
from pathlib import Path
from threading import RLock

PROTOCOL_REVISION = "solaredge-evidence-replay-v1"
EVENT_TYPES = frozenset(
    {
        "powermind_solaredge_ac_energy_acquisition",
        "powermind_solaredge_ac_energy_failure",
    }
)
SAFE_FIELDS = frozenset(
    {
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
)
DEFAULT_CAPACITY = 10_000
MAX_PAGE_SIZE = 500


def _serialized(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapped


class JournalConflict(ValueError):
    """An epoch or immutable generation conflicts with retained history."""


class InvalidReplayCursor(ValueError):
    """The supplied cursor was not issued by this journal."""


class ReplayCursorExpired(ValueError):
    """The requested continuation predates the retained floor."""


class EvidenceReplay:
    """Read-only API exposed to consumers through hass.data."""

    def __init__(self, journal: EvidenceJournal) -> None:
        self._journal = journal

    def page(self, *, after: str | None = None, limit: int = 100) -> dict:
        return self._journal.page(after=after, limit=limit)


class EvidenceJournal:
    """SQLite writer with a persistent cursor key and global sequence."""

    def __init__(self, connection: sqlite3.Connection, capacity: int) -> None:
        self._db = connection
        self._lock = RLock()
        self.capacity = capacity
        self._key = bytes.fromhex(self._meta("cursor_key"))

    @classmethod
    def open(
        cls, path: str | Path, *, capacity: int = DEFAULT_CAPACITY
    ) -> EvidenceJournal:
        if type(capacity) is not int or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(path, check_same_thread=False)
        try:
            db.execute("PRAGMA synchronous=FULL")
            db.execute("PRAGMA auto_vacuum=FULL")
            db.execute("PRAGMA journal_mode=DELETE")
            with db:
                db.execute(
                    "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
                )
                db.execute(
                    "CREATE TABLE IF NOT EXISTS epochs (epoch_id TEXT PRIMARY KEY)"
                )
                db.execute(
                    "CREATE TABLE IF NOT EXISTS records ("
                    "sequence INTEGER PRIMARY KEY, epoch_id TEXT NOT NULL, "
                    "generation INTEGER NOT NULL, event_type TEXT NOT NULL, "
                    "payload TEXT NOT NULL, UNIQUE(epoch_id, generation))"
                )
                db.execute(
                    "INSERT OR IGNORE INTO meta VALUES ('cursor_key', ?)",
                    (secrets.token_hex(32),),
                )
                db.execute("INSERT OR IGNORE INTO meta VALUES ('highwater', '0')")
                db.execute("INSERT OR IGNORE INTO meta VALUES ('retained_floor', '0')")
            return cls(db, capacity)
        except Exception:
            db.close()
            raise

    @_serialized
    def close(self) -> None:
        self._db.close()

    def reader(self) -> EvidenceReplay:
        return EvidenceReplay(self)

    def _meta(self, key: str) -> str:
        row = self._db.execute(
            "SELECT value FROM meta WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            raise JournalConflict("journal metadata is incomplete")
        return row[0]

    @_serialized
    def start_epoch(self, epoch_id: str) -> None:
        if not isinstance(epoch_id, str) or not epoch_id:
            raise ValueError("epoch_id must be a nonempty string")
        with self._db:
            if self._db.execute(
                "SELECT 1 FROM epochs WHERE epoch_id = ?", (epoch_id,)
            ).fetchone():
                raise JournalConflict("epoch_id was already used")
            self._db.execute("INSERT INTO epochs VALUES (?)", (epoch_id,))
            self._db.execute(
                "INSERT OR REPLACE INTO meta VALUES ('current_epoch', ?)", (epoch_id,)
            )
            self._db.execute(
                "INSERT INTO meta VALUES (?, '0')", (f"generation:{epoch_id}",)
            )

    def _cursor(self, sequence: int) -> str:
        sequence_text = str(sequence)
        signature = hmac.new(
            self._key, sequence_text.encode(), hashlib.sha256
        ).hexdigest()
        return (
            base64.urlsafe_b64encode(f"{sequence_text}:{signature}".encode())
            .rstrip(b"=")
            .decode()
        )

    def _parse_cursor(self, cursor: str) -> int:
        if not isinstance(cursor, str) or len(cursor) > 160:
            raise InvalidReplayCursor("invalid replay cursor")
        try:
            raw = base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)).decode()
            sequence_text, _signature = raw.split(":", 1)
            sequence = int(sequence_text)
            if sequence < 1 or sequence_text != str(sequence):
                raise ValueError
            if not hmac.compare_digest(self._cursor(sequence), cursor):
                raise ValueError
            return sequence
        except (ValueError, UnicodeError, base64.binascii.Error) as exc:
            raise InvalidReplayCursor("invalid replay cursor") from exc

    @_serialized
    def append(self, event_type: str, payload: dict) -> str:
        if event_type not in EVENT_TYPES or not isinstance(payload, dict):
            raise JournalConflict("unsupported evidence record")
        if set(payload) != SAFE_FIELDS:
            raise JournalConflict("evidence fields do not match the share-safe schema")
        if any(
            type(value) not in (str, int, float, bool, type(None))
            for value in payload.values()
        ):
            raise JournalConflict("evidence values must be scalar")
        if any(
            value is not None and type(value) is not int
            for value in (payload["raw_ac_energy_wh"], payload["raw_ac_energy_sf"])
        ):
            raise JournalConflict("raw evidence values must be integers or null")
        epoch_id = self._meta("current_epoch")
        generation = payload["generation"]
        if (
            payload["epoch_id"] != epoch_id
            or type(generation) is not int
            or generation < 1
        ):
            raise JournalConflict("evidence epoch or generation is invalid")
        canonical = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        with self._db:
            existing = self._db.execute(
                "SELECT sequence, event_type, payload FROM records "
                "WHERE epoch_id = ? AND generation = ?",
                (epoch_id, generation),
            ).fetchone()
            if existing:
                if existing[1:] != (event_type, canonical):
                    raise JournalConflict("immutable evidence record conflicts")
                return self._cursor(existing[0])
            last = self._db.execute(
                "SELECT MAX(generation) FROM records WHERE epoch_id = ?", (epoch_id,)
            ).fetchone()[0]
            # A new epoch always begins at one. Retention may remove old rows,
            # so the per-epoch highwater is kept separately.
            last = int(self._meta(f"generation:{epoch_id}")) if last is None else last
            if generation != last + 1:
                raise JournalConflict("evidence generation is not contiguous")
            sequence = int(self._meta("highwater")) + 1
            self._db.execute(
                "INSERT INTO records VALUES (?, ?, ?, ?, ?)",
                (sequence, epoch_id, generation, event_type, canonical),
            )
            self._db.execute(
                "INSERT OR REPLACE INTO meta VALUES ('highwater', ?)", (str(sequence),)
            )
            self._db.execute(
                "INSERT OR REPLACE INTO meta VALUES (?, ?)",
                (f"generation:{epoch_id}", str(generation)),
            )
            oldest = self._db.execute(
                "SELECT sequence FROM records ORDER BY sequence DESC LIMIT 1 OFFSET ?",
                (self.capacity,),
            ).fetchone()
            if oldest:
                self._db.execute(
                    "DELETE FROM records WHERE sequence <= ?", (oldest[0],)
                )
                self._db.execute(
                    "INSERT OR REPLACE INTO meta VALUES ('retained_floor', ?)",
                    (str(oldest[0]),),
                )
        return self._cursor(sequence)

    @_serialized
    def page(self, *, after: str | None = None, limit: int = 100) -> dict:
        if type(limit) is not int or not 1 <= limit <= MAX_PAGE_SIZE:
            raise ValueError(f"limit must be between 1 and {MAX_PAGE_SIZE}")
        sequence = 0 if after is None else self._parse_cursor(after)
        floor = int(self._meta("retained_floor"))
        highwater = int(self._meta("highwater"))
        if sequence > highwater:
            raise InvalidReplayCursor("cursor is ahead of the journal")
        if sequence < floor:
            raise ReplayCursorExpired("cursor predates retained evidence")
        rows = self._db.execute(
            "SELECT sequence, event_type, payload FROM records "
            "WHERE sequence > ? ORDER BY sequence LIMIT ?",
            (sequence, limit + 1),
        ).fetchall()
        expected = sequence + 1
        for row in rows:
            if row[0] != expected:
                raise JournalConflict("retained evidence has a sequence gap")
            expected += 1
        if len(rows) <= limit and expected <= highwater:
            raise JournalConflict("retained evidence has a sequence gap")
        page_rows = rows[:limit]
        records = [
            {
                "cursor": self._cursor(seq),
                "event_type": event_type,
                "payload": json.loads(payload),
            }
            for seq, event_type, payload in page_rows
        ]
        return {
            "protocol_revision": PROTOCOL_REVISION,
            "current_epoch": self._meta("current_epoch"),
            "retained_floor": floor,
            "highwater": highwater,
            "records": records,
            "next_cursor": records[-1]["cursor"] if records else after,
            "has_more": len(rows) > limit,
        }

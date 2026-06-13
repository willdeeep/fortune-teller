"""SQLite-backed store for completed reading history.

Stores every finalised :class:`~fortune_teller.application.models.domain.Reading`
as a JSON payload with promoted metadata columns for fast listing. The design
mirrors :class:`~fortune_teller.application.stores.vector.VectorStore` — explicit
``open()``/``close()`` lifecycle, context-manager support, and a thread-safe
single connection guarded by a ``threading.Lock``.

Usage::

    with SQLiteStore(path) as store:
        store.save(reading)
        recent = store.list_recent()
        full = store.get(reading.id)
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from fortune_teller.application.models.domain import Reading, ReadingListItem

_SUMMARY_PREVIEW_LENGTH = 120

_CREATE_READINGS_TABLE = """\
CREATE TABLE IF NOT EXISTS readings (
    id          TEXT PRIMARY KEY,
    deck_id     TEXT NOT NULL,
    spread_id   TEXT NOT NULL,
    summary     TEXT NOT NULL,
    card_names  TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    payload     TEXT NOT NULL
)
"""

_CREATE_CREATED_AT_INDEX = """\
CREATE INDEX IF NOT EXISTS idx_readings_created_at ON readings (created_at DESC)
"""

_CREATE_SCHEMA_VERSION_TABLE = """\
CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)
"""

_INSERT_OR_REPLACE = """\
INSERT OR REPLACE INTO readings
    (id, deck_id, spread_id, summary, card_names, created_at, payload)
VALUES (?, ?, ?, ?, ?, ?, ?)
"""

_SELECT_RECENT = """\
SELECT id, deck_id, spread_id, card_names, summary, created_at
FROM readings
ORDER BY created_at DESC
LIMIT ?
"""

_SELECT_PAYLOAD_BY_ID = """\
SELECT payload FROM readings WHERE id = ?
"""

_DELETE_BY_ID = """\
DELETE FROM readings WHERE id = ?
"""


class SQLiteStore:
    """SQLite-backed store for completed readings.

    Thread-safe via a single connection guarded by a ``threading.Lock``.
    Use as a context manager to ensure the connection is closed::

        with SQLiteStore(path) as store:
            store.save(reading)
            recent = store.list_recent()

    Args:
        path: File path to the SQLite database. Parent directories are
              created automatically on ``open()``.
    """

    SCHEMA_VERSION = 1

    def __init__(self, path: str | Path) -> None:
        self._path = str(path)
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Context manager / lifecycle
    # ------------------------------------------------------------------

    def __enter__(self) -> SQLiteStore:
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def open(self) -> None:
        """Open the database connection and ensure the schema exists.

        Creates parent directories if needed, enables WAL mode for
        concurrent reads, and seeds the schema if the database is new.
        """
        Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()

    def close(self) -> None:
        """Close the database connection. Safe to call multiple times."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(self, reading: Reading) -> None:
        """Insert or replace a reading. Idempotent on ``Reading.id``.

        Args:
            reading: A fully-formed :class:`Reading` to persist.
        """
        with self._lock:
            self._require_conn().execute(
                _INSERT_OR_REPLACE,
                (
                    str(reading.id),
                    reading.deck_id,
                    reading.spread_id,
                    reading.summary,
                    ",".join(i.card_name for i in reading.per_card),
                    reading.created_at.isoformat(),
                    reading.model_dump_json(),
                ),
            )
            self._require_conn().commit()

    def get(self, reading_id: UUID) -> Reading | None:
        """Return the full reading by *reading_id*, or ``None`` if not found.

        Reconstructs the :class:`Reading` from the stored JSON payload.
        """
        with self._lock:
            row = (
                self._require_conn()
                .execute(
                    _SELECT_PAYLOAD_BY_ID,
                    (str(reading_id),),
                )
                .fetchone()
            )
        if row is None:
            return None
        return Reading.model_validate_json(row["payload"])

    def list_recent(self, limit: int = 50) -> list[ReadingListItem]:
        """Return recent readings (metadata only), newest first.

        Args:
            limit: Maximum number of rows to return. Defaults to 50.

        Returns:
            A list of :class:`ReadingListItem` objects ordered by
            ``created_at`` descending.
        """
        with self._lock:
            rows = (
                self._require_conn()
                .execute(
                    _SELECT_RECENT,
                    (limit,),
                )
                .fetchall()
            )
        return [_row_to_list_item(row) for row in rows]

    def delete(self, reading_id: UUID) -> bool:
        """Delete a reading by *reading_id*.

        Returns:
            ``True`` if a row was removed, ``False`` if the id was not found.
        """
        with self._lock:
            cursor = self._require_conn().execute(
                _DELETE_BY_ID,
                (str(reading_id),),
            )
            self._require_conn().commit()
            return cursor.rowcount > 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_schema(self) -> None:
        """Create tables and indexes if they do not exist, and seed
        ``schema_version``.
        """
        conn = self._require_conn()
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute(_CREATE_READINGS_TABLE)
        conn.execute(_CREATE_CREATED_AT_INDEX)
        conn.execute(_CREATE_SCHEMA_VERSION_TABLE)
        conn.execute(
            "INSERT OR IGNORE INTO schema_version (version) VALUES (?)",
            (self.SCHEMA_VERSION,),
        )
        conn.commit()

    def _require_conn(self) -> sqlite3.Connection:
        """Return the open connection or raise ``RuntimeError``."""
        if self._conn is None:
            msg = (
                "SQLiteStore is not open. Use it as a context manager "
                "(`with SQLiteStore(path) as store: ...`) or call .open() first."
            )
            raise RuntimeError(msg)
        return self._conn


def _row_to_list_item(row: sqlite3.Row) -> ReadingListItem:
    """Convert a database row into a :class:`ReadingListItem`."""
    summary = row["summary"]
    preview = (
        summary[:_SUMMARY_PREVIEW_LENGTH] if len(summary) > _SUMMARY_PREVIEW_LENGTH else summary
    )

    card_names = row["card_names"].split(",") if row["card_names"] else []

    created_at = datetime.fromisoformat(row["created_at"])
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=UTC)

    return ReadingListItem(
        id=UUID(row["id"]),
        deck_id=row["deck_id"],
        spread_id=row["spread_id"],
        card_names=card_names,
        summary_preview=preview,
        created_at=created_at,
    )

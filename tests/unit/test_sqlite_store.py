"""Unit tests for :mod:`fortune_teller.application.stores.sqlite`.

Covers the full public API of :class:`SQLiteStore`: lifecycle, round-trip
persistence, listing, deletion, and schema management. All tests use
``tmp_path`` for isolated database files.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest

from fortune_teller.application.models.domain import (
    CardInterpretation,
    DealtCard,
    Orientation,
    Reading,
    ReadingListItem,
)
from fortune_teller.application.stores.sqlite import SQLiteStore


def _make_reading(
    *,
    reading_id: uuid.UUID | None = None,
    deck_id: str = "test-deck",
    spread_id: str = "test-spread",
    summary: str = "Test summary.",
    created_at: datetime | None = None,
) -> Reading:
    fixed_id = reading_id or uuid.uuid4()
    fixed_time = created_at or datetime(2025, 6, 1, 12, 0, 0, tzinfo=UTC)
    return Reading(
        id=fixed_id,
        deck_id=deck_id,
        spread_id=spread_id,
        dealt=[
            DealtCard(card_id="the-fool", orientation=Orientation.UPRIGHT, position_index=0),
            DealtCard(card_id="the-magician", orientation=Orientation.UPRIGHT, position_index=1),
        ],
        per_card=[
            CardInterpretation(
                dealt=DealtCard(
                    card_id="the-fool", orientation=Orientation.UPRIGHT, position_index=0
                ),
                card_name="The Fool",
                position_name="Past",
                text="New beginnings.",
            ),
            CardInterpretation(
                dealt=DealtCard(
                    card_id="the-magician", orientation=Orientation.UPRIGHT, position_index=1
                ),
                card_name="The Magician",
                position_name="Present",
                text="Harness your tools.",
            ),
        ],
        summary=summary,
        created_at=fixed_time,
    )


@pytest.mark.unit
class TestSQLiteStoreLifecycle:
    def test_open_creates_parent_dirs(self, tmp_path: pytest.TempPathFactory) -> None:
        db_path = tmp_path / "sub" / "dir" / "test.db"
        store = SQLiteStore(db_path)
        store.open()
        assert db_path.parent.exists()
        store.close()

    def test_context_manager(self, tmp_path: pytest.TempPathFactory) -> None:
        db_path = tmp_path / "ctx.db"
        with SQLiteStore(db_path) as store:
            assert store._conn is not None
        assert store._conn is None

    def test_double_close_is_safe(self, tmp_path: pytest.TempPathFactory) -> None:
        db_path = tmp_path / "dbl.db"
        store = SQLiteStore(db_path)
        store.open()
        store.close()
        store.close()

    def test_require_conn_raises_when_not_open(self, tmp_path: pytest.TempPathFactory) -> None:
        store = SQLiteStore(tmp_path / "never.db")
        with pytest.raises(RuntimeError, match="not open"):
            store._require_conn()


@pytest.mark.unit
class TestSQLiteStoreSaveAndGet:
    def test_round_trip(self, tmp_path: pytest.TempPathFactory) -> None:
        reading = _make_reading()
        with SQLiteStore(tmp_path / "rt.db") as store:
            store.save(reading)
            loaded = store.get(reading.id)
        assert loaded is not None
        assert loaded == reading

    def test_save_is_idempotent(self, tmp_path: pytest.TempPathFactory) -> None:
        reading = _make_reading()
        with SQLiteStore(tmp_path / "idem.db") as store:
            store.save(reading)
            store.save(reading)
            assert len(store.list_recent()) == 1

    def test_get_missing_returns_none(self, tmp_path: pytest.TempPathFactory) -> None:
        with SQLiteStore(tmp_path / "miss.db") as store:
            assert store.get(uuid.uuid4()) is None

    def test_persistence_across_reopen(self, tmp_path: pytest.TempPathFactory) -> None:
        reading = _make_reading()
        db_path = tmp_path / "persist.db"
        with SQLiteStore(db_path) as store:
            store.save(reading)
        with SQLiteStore(db_path) as store2:
            loaded = store2.get(reading.id)
        assert loaded is not None
        assert loaded == reading

    def test_save_replaces_on_same_id(self, tmp_path: pytest.TempPathFactory) -> None:
        fixed_id = uuid.UUID("11111111-2222-3333-4444-555555555555")
        original = _make_reading(reading_id=fixed_id, summary="Original summary.")
        updated = _make_reading(reading_id=fixed_id, summary="Updated summary.")
        db_path = tmp_path / "replace.db"
        with SQLiteStore(db_path) as store:
            store.save(original)
            store.save(updated)
            loaded = store.get(fixed_id)
            assert loaded is not None
            assert loaded.summary == "Updated summary."
            assert len(store.list_recent()) == 1


@pytest.mark.unit
class TestSQLiteStoreListRecent:
    def test_orders_newest_first(self, tmp_path: pytest.TempPathFactory) -> None:
        t1 = datetime(2025, 1, 1, tzinfo=UTC)
        t2 = datetime(2025, 6, 1, tzinfo=UTC)
        t3 = datetime(2025, 3, 1, tzinfo=UTC)
        r1 = _make_reading(reading_id=uuid.uuid4(), summary="Old", created_at=t1)
        r2 = _make_reading(reading_id=uuid.uuid4(), summary="New", created_at=t2)
        r3 = _make_reading(reading_id=uuid.uuid4(), summary="Mid", created_at=t3)
        with SQLiteStore(tmp_path / "order.db") as store:
            store.save(r1)
            store.save(r2)
            store.save(r3)
            items = store.list_recent()
        assert len(items) == 3
        assert items[0].created_at >= items[1].created_at >= items[2].created_at

    def test_respects_limit(self, tmp_path: pytest.TempPathFactory) -> None:
        with SQLiteStore(tmp_path / "limit.db") as store:
            for i in range(10):
                store.save(
                    _make_reading(
                        reading_id=uuid.uuid4(),
                        summary=f"Reading {i}",
                        created_at=datetime(2025, 1, i + 1, tzinfo=UTC),
                    )
                )
            items = store.list_recent(limit=3)
        assert len(items) == 3

    def test_returns_reading_list_items(self, tmp_path: pytest.TempPathFactory) -> None:
        reading = _make_reading(summary="A " * 50)
        with SQLiteStore(tmp_path / "types.db") as store:
            store.save(reading)
            items = store.list_recent()
        assert len(items) == 1
        item = items[0]
        assert isinstance(item, ReadingListItem)
        assert item.deck_id == "test-deck"
        assert item.spread_id == "test-spread"
        assert len(item.card_names) == 2
        assert "The Fool" in item.card_names

    def test_summary_preview_truncation(self, tmp_path: pytest.TempPathFactory) -> None:
        long_summary = "x" * 300
        reading = _make_reading(summary=long_summary)
        with SQLiteStore(tmp_path / "trunc.db") as store:
            store.save(reading)
            items = store.list_recent()
        assert len(items) == 1
        assert len(items[0].summary_preview) <= 120


@pytest.mark.unit
class TestSQLiteStoreDelete:
    def test_delete_existing(self, tmp_path: pytest.TempPathFactory) -> None:
        reading = _make_reading()
        with SQLiteStore(tmp_path / "del.db") as store:
            store.save(reading)
            assert store.delete(reading.id) is True
            assert store.get(reading.id) is None

    def test_delete_missing_returns_false(self, tmp_path: pytest.TempPathFactory) -> None:
        with SQLiteStore(tmp_path / "delmiss.db") as store:
            assert store.delete(uuid.uuid4()) is False


@pytest.mark.unit
class TestSQLiteStoreSchema:
    def test_schema_version_seeded(self, tmp_path: pytest.TempPathFactory) -> None:
        with SQLiteStore(tmp_path / "schema.db") as store:
            conn = store._require_conn()
            row = conn.execute("SELECT version FROM schema_version").fetchone()
        assert row is not None
        assert row[0] == 1

    def test_wal_mode_enabled(self, tmp_path: pytest.TempPathFactory) -> None:
        with SQLiteStore(tmp_path / "wal.db") as store:
            conn = store._require_conn()
            row = conn.execute("PRAGMA journal_mode").fetchone()
        assert row[0] == "wal"


@pytest.mark.unit
class TestSQLiteStoreEmptyDb:
    def test_list_recent_on_empty_db(self, tmp_path: pytest.TempPathFactory) -> None:
        with SQLiteStore(tmp_path / "empty.db") as store:
            items = store.list_recent()
        assert items == []

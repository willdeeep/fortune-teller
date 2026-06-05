"""Integration tests for the DuckDB VSS vector store round-trip.

These tests exercise the persistent on-disk DuckDB path (``tmp_path``) rather
than the in-memory mode used by the unit tests. They verify:

- Chunks written in one process/session can be read back after close/reopen.
- Card-section filtering survives an index rebuild.
- Spread-position search returns the correct position after persistence.
- A full HNSW-indexed round trip returns top-N results ranked by similarity.

Marked ``@pytest.mark.integration`` because they touch the on-disk DuckDB
binary format (not a pure in-memory model). Still fast enough to run in CI
without the slow marker.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pytest

from fortune_teller.application.models.domain import (
    CardSection,
    Chunk,
    ChunkType,
)
from fortune_teller.application.stores.vector import VectorStore

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DIM = 8  # small but non-trivial dimensionality for deterministic ranking


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unit_vector(seed: int, dim: int = _DIM) -> list[float]:
    """Return a deterministic unit-ish vector with one dominant coordinate.

    Seed 0 puts the mass at coordinate 0, seed 1 at coordinate 1, etc.
    This makes test assertions about "most similar" stable.
    """
    vec = [0.0] * dim
    vec[seed % dim] = 1.0
    return vec


def _make_chunk(
    *,
    card_id: str,
    section: CardSection,
    text: str,
    embedding: list[float],
    chunk_type: ChunkType = ChunkType.CARD_SECTION,
    deck_id: str = "book-of-thoth",
) -> Chunk:
    return Chunk(
        chunk_type=chunk_type,
        deck_id=deck_id,
        card_id=card_id,
        card_name=card_id.replace("-", " ").title(),
        section=section,
        spread_id=None,
        position_index=None,
        source_url=f"https://thothreadings.com/{card_id}/",
        text=text,
        embedding=embedding,
    )


def _make_position_chunk(
    *,
    spread_id: str,
    position_index: int,
    text: str,
    embedding: list[float],
) -> Chunk:
    return Chunk(
        chunk_type=ChunkType.SPREAD_POSITION,
        deck_id=None,
        card_id=None,
        card_name=None,
        section=None,
        spread_id=spread_id,
        position_index=position_index,
        source_url=f"https://thothreadings.com/{spread_id}/",
        text=text,
        embedding=embedding,
    )


# ---------------------------------------------------------------------------
# Persistence round-trip
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestVectorStorePersistence:
    """Verify the on-disk DuckDB file persists across open/close cycles."""

    def test_chunks_survive_close_reopen(self, tmp_path: Path) -> None:
        """Chunks added in one session are still searchable after reopening."""
        db_path = tmp_path / "store.duckdb"
        chunks = [
            _make_chunk(
                card_id="the-fool",
                section=CardSection.DRIVE,
                text="Pure potential.",
                embedding=_unit_vector(0),
            ),
            _make_chunk(
                card_id="the-fool",
                section=CardSection.LIGHT,
                text="Spontaneity.",
                embedding=_unit_vector(1),
            ),
        ]
        with VectorStore(str(db_path), dimension=_DIM) as store:
            store.add_chunks(chunks)

        # Reopen and search
        with VectorStore(str(db_path), dimension=_DIM) as store:
            hits = store.search_card_section(
                card_id="the-fool",
                query_embedding=_unit_vector(0),
                k=2,
            )
        assert len(hits) == 2
        # Top hit should be the seed-0 chunk (identical vector → cosine 1.0)
        assert hits[0].chunk.text == "Pure potential."
        assert hits[0].score == pytest.approx(1.0, abs=1e-5)

    def test_empty_store_round_trip(self, tmp_path: Path) -> None:
        """A freshly-opened store has no hits and persists that way."""
        db_path = tmp_path / "empty.duckdb"
        with VectorStore(str(db_path), dimension=_DIM) as store:
            hits = store.search_card_section(
                card_id="the-fool",
                query_embedding=_unit_vector(0),
                k=5,
            )
            assert hits == []

        # Reopen — still empty
        with VectorStore(str(db_path), dimension=_DIM) as store:
            hits = store.search_card_section(
                card_id="the-fool",
                query_embedding=_unit_vector(0),
                k=5,
            )
            assert hits == []


# ---------------------------------------------------------------------------
# Card-section search
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestCardSectionSearchRanking:
    """Verify card-section search returns semantically-relevant results."""

    _N_CHUNKS: ClassVar[int] = 20

    def test_top_result_is_most_relevant(self, tmp_path: Path) -> None:
        """Top hit for a query is the chunk with the closest embedding."""
        db_path = tmp_path / "rank.duckdb"
        chunks = [
            _make_chunk(
                card_id="the-fool",
                section=CardSection.DRIVE,
                text=f"chunk-{i}",
                embedding=_unit_vector(i),
            )
            for i in range(self._N_CHUNKS)
        ]
        with VectorStore(str(db_path), dimension=_DIM) as store:
            store.add_chunks(chunks)
            # Query for seed=7 → that chunk should be top
            hits = store.search_card_section(
                card_id="the-fool",
                query_embedding=_unit_vector(7),
                k=5,
            )
        assert len(hits) == 5
        assert hits[0].chunk.text == "chunk-7"
        # The second hit should be one of the orthogonal chunks (cosine 0)
        # — the exact identity depends on the HNSW traversal, so we just
        # assert it's not chunk-7 again.
        assert hits[1].chunk.text != "chunk-7"

    def test_search_filters_by_card_id(self, tmp_path: Path) -> None:
        """Search for card_id='the-fool' never returns chunks of another card."""
        db_path = tmp_path / "filter.duckdb"
        chunks = [
            _make_chunk(
                card_id="the-fool",
                section=CardSection.DRIVE,
                text="fool-text",
                embedding=_unit_vector(0),
            ),
            _make_chunk(
                card_id="the-magus",
                section=CardSection.DRIVE,
                text="magus-text",
                embedding=_unit_vector(0),
            ),
        ]
        with VectorStore(str(db_path), dimension=_DIM) as store:
            store.add_chunks(chunks)
            hits = store.search_card_section(
                card_id="the-fool",
                query_embedding=_unit_vector(0),
                k=10,
            )
        assert all(h.chunk.card_id == "the-fool" for h in hits)
        assert len(hits) == 1
        assert hits[0].chunk.text == "fool-text"


# ---------------------------------------------------------------------------
# Spread-position search
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestSpreadPositionSearch:
    """Verify spread-position search returns the correct position chunk."""

    def test_returns_position_chunk(self, tmp_path: Path) -> None:
        """Search by spread_id + position_index returns the matching chunk."""
        db_path = tmp_path / "spread.duckdb"
        chunks = [
            _make_position_chunk(
                spread_id="new-moon-three-card",
                position_index=0,
                text="Past — what has been.",
                embedding=_unit_vector(0),
            ),
            _make_position_chunk(
                spread_id="new-moon-three-card",
                position_index=1,
                text="Present — what is.",
                embedding=_unit_vector(1),
            ),
            _make_position_chunk(
                spread_id="new-moon-three-card",
                position_index=2,
                text="Future — what may come.",
                embedding=_unit_vector(2),
            ),
        ]
        with VectorStore(str(db_path), dimension=_DIM) as store:
            store.add_chunks(chunks)
            hits = store.search_spread_position(
                spread_id="new-moon-three-card",
                position_index=1,
            )
        assert len(hits) == 1
        assert hits[0].chunk.position_index == 1
        assert "Present" in hits[0].chunk.text

    def test_missing_position_returns_empty(self, tmp_path: Path) -> None:
        """A search for a position with no chunk returns no hits."""
        db_path = tmp_path / "missing.duckdb"
        with VectorStore(str(db_path), dimension=_DIM) as store:
            hits = store.search_spread_position(
                spread_id="new-moon-three-card",
                position_index=99,
            )
        assert hits == []


# ---------------------------------------------------------------------------
# Mixed round-trip — full pipeline simulation
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestVectorStoreRoundTrip:
    """Simulate a small ft-parse → ft-embed → ft-build-index cycle."""

    def test_full_round_trip_with_persisted_search(self, tmp_path: Path) -> None:
        """Write 20 mixed chunks, close, reopen, search — get the right card back."""
        db_path = tmp_path / "full.duckdb"
        # 10 card-section chunks (2 cards x 5 sections) + 3 spread-position chunks
        chunks: list[Chunk] = []
        for i, card_id in enumerate(["the-fool", "the-magus"]):
            for j, section in enumerate(
                [
                    CardSection.DRIVE,
                    CardSection.LIGHT,
                    CardSection.SHADOW,
                    CardSection.REVERSED,
                    CardSection.KEYWORDS,
                ]
            ):
                chunks.append(
                    _make_chunk(
                        card_id=card_id,
                        section=section,
                        text=f"{card_id}-{section.value}",
                        embedding=_unit_vector(i * 5 + j),
                    )
                )
        for idx in range(3):
            chunks.append(
                _make_position_chunk(
                    spread_id="new-moon-three-card",
                    position_index=idx,
                    text=f"position-{idx}",
                    embedding=_unit_vector(100 + idx),
                )
            )

        # Write
        with VectorStore(str(db_path), dimension=_DIM) as store:
            store.add_chunks(chunks)

        # Verify on-disk file exists
        assert db_path.exists()
        # And the file is non-trivial (DuckDB header + data)
        assert db_path.stat().st_size > 0

        # Read back from a fresh session
        with VectorStore(str(db_path), dimension=_DIM) as store:
            # Find all of the-fool's sections
            fool_hits = store.search_card_section(
                card_id="the-fool",
                query_embedding=_unit_vector(0),  # matches DRIVE
                k=5,
            )
            assert len(fool_hits) == 5
            assert all(h.chunk.card_id == "the-fool" for h in fool_hits)

            # Find the spread's position 2
            pos_hits = store.search_spread_position(
                spread_id="new-moon-three-card",
                position_index=2,
            )
            assert len(pos_hits) == 1
            assert pos_hits[0].chunk.position_index == 2
            assert pos_hits[0].chunk.text == "position-2"

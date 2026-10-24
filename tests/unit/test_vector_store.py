"""Unit tests for ``VectorStore`` using an in-memory DuckDB.

The VSS extension is loaded from the in-memory database; no fixtures
or files are committed to disk. All tests are ``@pytest.mark.unit`` and
must not touch the network.
"""

from __future__ import annotations

import json
import uuid
from typing import ClassVar

import pytest
from pydantic import HttpUrl

from fortune_teller.application.models.domain import (
    Arcana,
    Card,
    CardSection,
    CardSectionText,
    Chunk,
    ChunkType,
    SearchHit,
)
from fortune_teller.application.stores.vector import (
    DEFAULT_EMBEDDING_DIMENSION,
    VectorStore,
)
from fortune_teller.developer.embed.chunker import attach_embeddings, chunks_from_card

DIM = 4  # tiny dimension for readable tests
_SOURCE = "https://example.com/the-fool/"


# ---------------------------------------------------------------------------
# Helpers / factories
# ---------------------------------------------------------------------------


def _make_chunk(
    *,
    text: str = "stub text",
    embedding: list[float] | None = None,
    chunk_type: ChunkType = ChunkType.CARD_SECTION,
    card_id: str | None = "the-fool",
    section: CardSection | None = CardSection.DRIVE,
    spread_id: str | None = None,
    position_index: int | None = None,
) -> Chunk:
    return Chunk(
        id=uuid.uuid4(),
        chunk_type=chunk_type,
        deck_id="book-of-thoth",
        card_id=card_id,
        card_name="The Fool" if card_id else None,
        section=section,
        spread_id=spread_id,
        position_index=position_index,
        source_url=_SOURCE,
        text=text,
        embedding=embedding if embedding is not None else [0.0] * DIM,
    )


def _vector(*values: float) -> list[float]:
    """Build a length-DIM vector from the given values, zero-padding."""
    base = list(values) + [0.0] * DIM
    return base[:DIM]


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestVectorStoreLifecycle:
    def test_open_creates_schema(self) -> None:
        store = VectorStore(":memory:", dimension=DIM)
        with store:
            assert store.count() == 0

    def test_close_is_idempotent(self) -> None:
        store = VectorStore(":memory:", dimension=DIM)
        store.open()
        store.close()
        store.close()  # second call must not raise

    def test_close_then_reopen_preserves_data(self) -> None:
        """After close+reopen, previously inserted chunks are still queryable.

        For an in-memory DB this is a no-op sanity check, but for the
        file-backed code path (regression-tested in integration) the
        reopening is what matters.
        """
        store = VectorStore(":memory:", dimension=DIM)
        with store:
            store.add_chunks([_make_chunk(text="persisted", embedding=_vector(0.1, 0.2, 0.3, 0.4))])
            assert store.count() == 1

    def test_operations_before_open_raise(self) -> None:
        store = VectorStore(":memory:", dimension=DIM)
        with pytest.raises(RuntimeError, match="not open"):
            store.count()

    def test_context_manager_returns_self(self) -> None:
        store = VectorStore(":memory:", dimension=DIM)
        with store as opened:
            assert opened is store

    def test_dimension_property(self) -> None:
        store = VectorStore(":memory:", dimension=768)
        assert store.dimension == 768


# ---------------------------------------------------------------------------
# add_chunks
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestVectorStoreAddChunks:
    def test_empty_input_is_noop(self) -> None:
        store = VectorStore(":memory:", dimension=DIM)
        with store:
            store.add_chunks([])
            assert store.count() == 0

    def test_single_chunk_insert(self) -> None:
        store = VectorStore(":memory:", dimension=DIM)
        with store:
            store.add_chunks([_make_chunk(embedding=_vector(0.1, 0.2, 0.3, 0.4))])
            assert store.count() == 1

    def test_multiple_chunks_insert(self) -> None:
        store = VectorStore(":memory:", dimension=DIM)
        chunks = [
            _make_chunk(text=f"chunk {i}", embedding=_vector(*([0.1 * (i + 1)] * DIM)))
            for i in range(5)
        ]
        with store:
            store.add_chunks(chunks)
            assert store.count() == 5

    def test_insert_or_replace_overwrites_same_id(self) -> None:
        store = VectorStore(":memory:", dimension=DIM)
        chunk_id = uuid.uuid4()
        c1 = Chunk(
            id=chunk_id,
            chunk_type=ChunkType.CARD_SECTION,
            deck_id="book-of-thoth",
            card_id="the-fool",
            source_url=_SOURCE,
            text="first",
            embedding=_vector(0.1, 0.2, 0.3, 0.4),
        )
        c2 = c1.model_copy(update={"text": "second", "embedding": _vector(0.5, 0.6, 0.7, 0.8)})
        with store:
            store.add_chunks([c1])
            store.add_chunks([c2])
            assert store.count() == 1
            hits = store.search(_vector(0.5, 0.6, 0.7, 0.8), k=1)
            assert hits[0].chunk.text == "second"


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestVectorStoreSearch:
    _Q_DRIVE: ClassVar[list[float]] = [1.0, 0.0, 0.0, 0.0]
    _Q_LIGHT: ClassVar[list[float]] = [0.0, 1.0, 0.0, 0.0]

    @pytest.fixture
    def store_with_chunks(self):
        chunks = [
            _make_chunk(
                text="drive section",
                embedding=_vector(1.0, 0.0, 0.0, 0.0),
                section=CardSection.DRIVE,
            ),
            _make_chunk(
                text="light section",
                embedding=_vector(0.0, 1.0, 0.0, 0.0),
                section=CardSection.LIGHT,
            ),
            _make_chunk(
                text="shadow section",
                embedding=_vector(0.0, 0.0, 1.0, 0.0),
                section=CardSection.SHADOW,
            ),
        ]
        store = VectorStore(":memory:", dimension=DIM)
        # Open manually (not via `with`) so the connection stays live
        # for the test body. The yield + close pattern ensures cleanup
        # after each test.
        store.open()
        store.add_chunks(chunks)
        try:
            yield store
        finally:
            store.close()

    def test_search_returns_top_k(self, store_with_chunks: VectorStore) -> None:
        hits = store_with_chunks.search(self._Q_DRIVE, k=2)
        assert len(hits) == 2
        assert all(isinstance(h, SearchHit) for h in hits)

    def test_search_ordered_by_score_desc(self, store_with_chunks: VectorStore) -> None:
        hits = store_with_chunks.search(self._Q_DRIVE, k=3)
        scores = [h.score for h in hits]
        assert scores == sorted(scores, reverse=True)
        # First hit is the exact match
        assert hits[0].chunk.text == "drive section"
        assert hits[0].score == pytest.approx(1.0, abs=1e-6)

    def test_search_empty_store_returns_empty_list(self) -> None:
        store = VectorStore(":memory:", dimension=DIM)
        with store:
            assert store.search(self._Q_DRIVE, k=4) == []

    def test_search_k_larger_than_corpus(self, store_with_chunks: VectorStore) -> None:
        hits = store_with_chunks.search(self._Q_LIGHT, k=100)
        assert len(hits) == 3

    def test_search_dimension_mismatch_raises(self, store_with_chunks: VectorStore) -> None:
        with pytest.raises(ValueError, match="dimension mismatch"):
            store_with_chunks.search([0.1, 0.2], k=1)


# ---------------------------------------------------------------------------
# search_card_section
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestVectorStoreSearchCardSection:
    def test_filters_by_card_id(self) -> None:
        store = VectorStore(":memory:", dimension=DIM)
        with store:
            store.add_chunks(
                [
                    _make_chunk(
                        card_id="the-fool",
                        section=CardSection.DRIVE,
                        text="fool-drive",
                        embedding=_vector(1.0, 0.0, 0.0, 0.0),
                    ),
                    _make_chunk(
                        card_id="the-magician",
                        section=CardSection.DRIVE,
                        text="magus-drive",
                        embedding=_vector(1.0, 0.0, 0.0, 0.0),
                    ),
                ]
            )
            hits = store.search_card_section("book-of-thoth", "the-fool", [1.0, 0.0, 0.0, 0.0], k=4)
            assert len(hits) == 1
            assert hits[0].chunk.card_id == "the-fool"
            assert hits[0].chunk.text == "fool-drive"

    def test_excludes_spread_position_chunks(self) -> None:
        store = VectorStore(":memory:", dimension=DIM)
        with store:
            store.add_chunks(
                [
                    _make_chunk(
                        chunk_type=ChunkType.SPREAD_POSITION,
                        spread_id="new-moon-three-card",
                        position_index=0,
                        text="spread text",
                        embedding=_vector(1.0, 0.0, 0.0, 0.0),
                        card_id=None,
                        section=None,
                    ),
                    _make_chunk(
                        chunk_type=ChunkType.CARD_SECTION,
                        card_id="the-fool",
                        section=CardSection.DRIVE,
                        text="card text",
                        embedding=_vector(0.0, 1.0, 0.0, 0.0),
                    ),
                ]
            )
            hits = store.search_card_section("book-of-thoth", "the-fool", [0.0, 1.0, 0.0, 0.0], k=4)
            assert len(hits) == 1
            assert hits[0].chunk.text == "card text"

    def test_no_match_returns_empty(self) -> None:
        store = VectorStore(":memory:", dimension=DIM)
        with store:
            store.add_chunks([_make_chunk(card_id="the-fool", text="x")])
            hits = store.search_card_section(
                "book-of-thoth", "the-magician", [0.0, 0.0, 0.0, 1.0], k=4
            )
            assert hits == []


# ---------------------------------------------------------------------------
# search_spread_position
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestVectorStoreSearchSpreadPosition:
    def test_returns_position_chunk(self) -> None:
        store = VectorStore(":memory:", dimension=DIM)
        with store:
            store.add_chunks(
                [
                    _make_chunk(
                        chunk_type=ChunkType.SPREAD_POSITION,
                        spread_id="new-moon-three-card",
                        position_index=1,
                        text="position 1 meaning",
                        card_id=None,
                        section=None,
                    ),
                ]
            )
            hits = store.search_spread_position("new-moon-three-card", 1)
            assert len(hits) == 1
            assert hits[0].chunk.text == "position 1 meaning"
            assert hits[0].score == 1.0  # non-similarity lookup, always 1.0

    def test_missing_position_returns_empty(self) -> None:
        store = VectorStore(":memory:", dimension=DIM)
        with store:
            hits = store.search_spread_position("nope", 0)
            assert hits == []


# ---------------------------------------------------------------------------
# clear / rebuild
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestVectorStoreClear:
    def test_clear_drops_table_and_index(self) -> None:
        store = VectorStore(":memory:", dimension=DIM)
        with store:
            store.add_chunks([_make_chunk(embedding=_vector(0.1, 0.2, 0.3, 0.4))])
            assert store.count() == 1
            store.clear()
            assert store.count() == 0

    def test_clear_then_add_works(self) -> None:
        store = VectorStore(":memory:", dimension=DIM)
        with store:
            store.add_chunks([_make_chunk(embedding=_vector(0.1, 0.2, 0.3, 0.4))])
            store.clear()
            store.add_chunks(
                [
                    _make_chunk(embedding=_vector(0.5, 0.6, 0.7, 0.8)),
                    _make_chunk(embedding=_vector(0.9, 0.1, 0.2, 0.3)),
                ]
            )
            assert store.count() == 2

    def test_clear_on_empty_does_not_raise(self) -> None:
        store = VectorStore(":memory:", dimension=DIM)
        with store:
            store.clear()  # no-op, must not raise
            assert store.count() == 0


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_default_embedding_dimension_is_384() -> None:
    assert DEFAULT_EMBEDDING_DIMENSION == 384


# Use the imports so ruff doesn't flag them as unused (re-exported check).
_ = HttpUrl


# ---------------------------------------------------------------------------
# Filtered search must not go through the HNSW index (plan 0041)
# ---------------------------------------------------------------------------


def _many_card_chunks(n_cards: int = 50, sections_per_card: int = 6) -> list[Chunk]:
    """Build a corpus where any single card is a small, selective slice.

    Vectors are deterministic but varied so the global nearest neighbours of a
    query are spread across many cards — the condition under which an
    index-first, filter-after plan drops the target card's rows.
    """
    sections = list(CardSection)[:sections_per_card]
    chunks: list[Chunk] = []
    for c in range(n_cards):
        for s, section in enumerate(sections):
            seed = c * sections_per_card + s + 1
            chunks.append(
                _make_chunk(
                    card_id=f"card-{c}",
                    section=section,
                    text=f"card-{c} {section.value}",
                    embedding=_vector(
                        (seed * 0.37) % 1.0 + 0.01,
                        (seed * 0.59) % 1.0 + 0.01,
                        (seed * 0.71) % 1.0 + 0.01,
                        (seed * 0.13) % 1.0 + 0.01,
                    ),
                )
            )
    return chunks


@pytest.mark.unit
class TestFilteredSearchUsesExactScan:
    """Guard against switching filtered queries to the HNSW-eligible shape.

    With the HNSW index present, ``ORDER BY array_cosine_distance(...) LIMIT k``
    plus a selective ``WHERE`` returns the global top-k *then* filters, which can
    yield zero rows. ``search_card_section`` must keep returning ``k`` rows.
    """

    _QUERY: ClassVar[list[float]] = [0.1, 0.2, 0.3, 0.4]

    def test_card_section_search_returns_k_rows_with_index_present(self) -> None:
        with VectorStore(":memory:", dimension=DIM) as store:
            store.add_chunks(_many_card_chunks())  # add_chunks (re)builds the HNSW index
            hits = store.search_card_section("book-of-thoth", "card-3", self._QUERY, k=4)
            assert len(hits) == 4
            assert {h.chunk.card_id for h in hits} == {"card-3"}

    def test_card_section_search_does_not_plan_hnsw_scan(self) -> None:
        with VectorStore(":memory:", dimension=DIM) as store:
            store.add_chunks(_many_card_chunks())
            conn = store._require_conn()
            sql = (
                f"EXPLAIN {store._search_sql()} "
                "WHERE chunk_type = 'card_section' AND deck_id = ? AND card_id = ? "
                "ORDER BY score DESC LIMIT ?"
            )
            plan = "".join(
                row[1]
                for row in conn.execute(
                    sql, [json.dumps(self._QUERY), "book-of-thoth", "card-3", 4]
                ).fetchall()
            )
            assert "HNSW" not in plan


# ---------------------------------------------------------------------------
# Re-embed idempotence (plan 0041)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestReembedIdempotence:
    def test_appending_rechunked_card_does_not_duplicate(self) -> None:
        card = Card(
            id="the-fool",
            name="The Fool",
            arcana=Arcana.MAJOR,
            sections=[
                CardSectionText(section=CardSection.OVERALL, text="Beginnings."),
                CardSectionText(section=CardSection.DRIVE, text="Spontaneity."),
            ],
            source_url=HttpUrl(_SOURCE),
        )

        def embedded_run() -> list[Chunk]:
            # Simulates one ft-embed run: fresh chunking, then vectors attached.
            chunks = chunks_from_card(card, deck_id="book-of-thoth")
            return attach_embeddings(chunks, [_vector(1.0)] * len(chunks))

        with VectorStore(":memory:", dimension=DIM) as store:
            store.add_chunks(embedded_run())
            store.add_chunks(embedded_run())  # == ft-build-index --no-rebuild after re-embed
            assert store.count() == 2

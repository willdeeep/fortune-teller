"""Cross-deck isolation regression tests (plan 0018).

Proves that search_card_section with deck_id filters correctly,
preventing cross-deck contamination when two decks share card IDs.
"""

from __future__ import annotations

import uuid

import pytest

from fortune_teller.application.models.domain import CardSection, Chunk, ChunkType
from fortune_teller.application.stores.vector import VectorStore

_DIM = 4
_SOURCE = "https://example.com/card/"


def _make_chunk(
    *,
    deck_id: str,
    card_id: str,
    text: str,
    embedding: list[float] | None = None,
) -> Chunk:
    return Chunk(
        id=uuid.uuid4(),
        chunk_type=ChunkType.CARD_SECTION,
        deck_id=deck_id,
        card_id=card_id,
        card_name=card_id.replace("-", " ").title(),
        section=CardSection.DRIVE,
        source_url=f"{_SOURCE}{card_id}/",
        text=text,
        embedding=embedding if embedding is not None else [0.0] * _DIM,
    )


@pytest.mark.unit
class TestCrossDeckIsolation:
    """Regression: two decks sharing a card_id must never cross-contaminate."""

    def test_search_card_section_filters_by_deck_id(self) -> None:
        """search_card_section with deck_id only returns chunks from that deck."""
        store = VectorStore(":memory:", dimension=_DIM)
        with store:
            # Insert two chunks with the same card_id but different deck_ids
            store.add_chunks(
                [
                    _make_chunk(
                        deck_id="book-of-thoth",
                        card_id="the-fool",
                        text="Thoth: pure potential",
                        embedding=[1.0, 0.0, 0.0, 0.0],
                    ),
                    _make_chunk(
                        deck_id="rider-waite",
                        card_id="the-fool",
                        text="RW: new beginnings",
                        embedding=[1.0, 0.0, 0.0, 0.0],
                    ),
                ]
            )

            # Searching Thoth deck should only return the Thoth chunk
            thoth_hits = store.search_card_section(
                deck_id="book-of-thoth",
                card_id="the-fool",
                query_embedding=[1.0, 0.0, 0.0, 0.0],
                k=4,
            )
            assert len(thoth_hits) == 1
            assert thoth_hits[0].chunk.deck_id == "book-of-thoth"
            assert thoth_hits[0].chunk.text == "Thoth: pure potential"

            # Searching RW deck should only return the RW chunk
            rw_hits = store.search_card_section(
                deck_id="rider-waite",
                card_id="the-fool",
                query_embedding=[1.0, 0.0, 0.0, 0.0],
                k=4,
            )
            assert len(rw_hits) == 1
            assert rw_hits[0].chunk.deck_id == "rider-waite"
            assert rw_hits[0].chunk.text == "RW: new beginnings"

    def test_search_card_section_no_cross_deck_results(self) -> None:
        """A deck_id with no matching chunks returns empty results."""
        store = VectorStore(":memory:", dimension=_DIM)
        with store:
            store.add_chunks(
                [
                    _make_chunk(
                        deck_id="book-of-thoth",
                        card_id="the-fool",
                        text="Thoth: pure potential",
                        embedding=[1.0, 0.0, 0.0, 0.0],
                    ),
                ]
            )

            # Searching a different deck returns nothing
            hits = store.search_card_section(
                deck_id="rider-waite",
                card_id="the-fool",
                query_embedding=[1.0, 0.0, 0.0, 0.0],
                k=4,
            )
            assert hits == []

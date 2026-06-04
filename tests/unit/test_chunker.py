"""Unit tests for the chunker: pure logic, no I/O."""

from __future__ import annotations

import pytest
from pydantic import HttpUrl

from fortune_teller.application.models.domain import (
    Arcana,
    Card,
    CardSection,
    CardSectionText,
    ChunkType,
    Spread,
    SpreadPosition,
)
from fortune_teller.developer.embed.chunker import (
    attach_embeddings,
    chunks_from_card,
    chunks_from_spread,
)

_SOURCE = "https://thothreadings.com/blog/0-the-fool/"
_SPREAD_URL = "https://thothreadings.com/spread-new-moon/"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def card() -> Card:
    return Card(
        id="0-the-fool",
        name="The Fool",
        arcana=Arcana.MAJOR,
        number=0,
        sections=[
            CardSectionText(
                section=CardSection.OVERALL, text="The Fool represents new beginnings."
            ),
            CardSectionText(section=CardSection.DRIVE, text="Spontaneity, innocence."),
            CardSectionText(section=CardSection.LIGHT, text="Idealism, optimism."),
        ],
        source_url=HttpUrl(_SOURCE),
    )


@pytest.fixture
def spread() -> Spread:
    return Spread(
        id="new-moon-three-card",
        name="New Moon Three-Card Spread",
        positions=[
            SpreadPosition(
                index=0,
                name="Past",
                meaning="What has been.",
                source_url=HttpUrl(_SPREAD_URL),
            ),
            SpreadPosition(
                index=1,
                name="Present",
                meaning="What is.",
                source_url=HttpUrl(_SPREAD_URL),
            ),
            SpreadPosition(
                index=2,
                name="Future",
                meaning="What will be.",
                source_url=HttpUrl(_SPREAD_URL),
            ),
        ],
    )


# ---------------------------------------------------------------------------
# chunks_from_card
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestChunksFromCard:
    def test_one_chunk_per_section(self, card: Card) -> None:
        chunks = chunks_from_card(card, deck_id="book-of-thoth")
        assert len(chunks) == len(card.sections) == 3

    def test_chunk_type_is_card_section(self, card: Card) -> None:
        chunks = chunks_from_card(card, deck_id="book-of-thoth")
        assert all(c.chunk_type == ChunkType.CARD_SECTION for c in chunks)

    def test_section_text_preserved(self, card: Card) -> None:
        chunks = chunks_from_card(card, deck_id="book-of-thoth")
        assert chunks[0].text == "The Fool represents new beginnings."
        assert chunks[0].section == CardSection.OVERALL

    def test_section_preserved(self, card: Card) -> None:
        chunks = chunks_from_card(card, deck_id="book-of-thoth")
        sections = [c.section for c in chunks]
        assert sections == [CardSection.OVERALL, CardSection.DRIVE, CardSection.LIGHT]

    def test_card_id_and_name_set(self, card: Card) -> None:
        chunks = chunks_from_card(card, deck_id="book-of-thoth")
        for c in chunks:
            assert c.card_id == "0-the-fool"
            assert c.card_name == "The Fool"

    def test_deck_id_attached(self, card: Card) -> None:
        chunks = chunks_from_card(card, deck_id="book-of-thoth")
        for c in chunks:
            assert c.deck_id == "book-of-thoth"

    def test_spread_fields_are_none(self, card: Card) -> None:
        chunks = chunks_from_card(card, deck_id="book-of-thoth")
        for c in chunks:
            assert c.spread_id is None
            assert c.position_index is None

    def test_source_url_is_card_url(self, card: Card) -> None:
        chunks = chunks_from_card(card, deck_id="book-of-thoth")
        for c in chunks:
            assert str(c.source_url) == _SOURCE

    def test_embedding_is_none(self, card: Card) -> None:
        chunks = chunks_from_card(card, deck_id="book-of-thoth")
        for c in chunks:
            assert c.embedding is None

    def test_card_with_no_sections_returns_empty(self) -> None:
        card = Card(
            id="empty",
            name="Empty",
            arcana=Arcana.MAJOR,
            source_url=HttpUrl(_SOURCE),
        )
        assert chunks_from_card(card, deck_id="d") == []


# ---------------------------------------------------------------------------
# chunks_from_spread
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestChunksFromSpread:
    def test_one_chunk_per_position(self, spread: Spread) -> None:
        chunks = chunks_from_spread(spread)
        assert len(chunks) == 3

    def test_chunk_type_is_spread_position(self, spread: Spread) -> None:
        chunks = chunks_from_spread(spread)
        assert all(c.chunk_type == ChunkType.SPREAD_POSITION for c in chunks)

    def test_spread_id_preserved(self, spread: Spread) -> None:
        chunks = chunks_from_spread(spread)
        for c in chunks:
            assert c.spread_id == "new-moon-three-card"

    def test_position_indices_preserved(self, spread: Spread) -> None:
        chunks = chunks_from_spread(spread)
        indices = [c.position_index for c in chunks]
        assert indices == [0, 1, 2]

    def test_text_includes_name_and_meaning(self, spread: Spread) -> None:
        chunks = chunks_from_spread(spread)
        assert "Past" in chunks[0].text
        assert "What has been" in chunks[0].text

    def test_card_fields_are_none(self, spread: Spread) -> None:
        chunks = chunks_from_spread(spread)
        for c in chunks:
            assert c.card_id is None
            assert c.card_name is None
            assert c.section is None
            assert c.deck_id is None

    def test_embedding_is_none(self, spread: Spread) -> None:
        chunks = chunks_from_spread(spread)
        for c in chunks:
            assert c.embedding is None

    def test_empty_spread_returns_empty_list(self) -> None:
        spread = Spread(id="empty", name="Empty Spread")
        assert chunks_from_spread(spread) == []


# ---------------------------------------------------------------------------
# attach_embeddings
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAttachEmbeddings:
    def test_attaches_vectors_in_order(self, card: Card) -> None:
        chunks = chunks_from_card(card, deck_id="book-of-thoth")
        vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
        result = attach_embeddings(chunks, vectors)
        assert result[0].embedding == [0.1, 0.2, 0.3]
        assert result[1].embedding == [0.4, 0.5, 0.6]
        assert result[2].embedding == [0.7, 0.8, 0.9]

    def test_returns_new_chunks_not_mutations(self, card: Card) -> None:
        """attach_embeddings must return new frozen chunks, not mutate inputs."""
        chunks = chunks_from_card(card, deck_id="book-of-thoth")
        original_id = chunks[0].id
        result = attach_embeddings(chunks, [[0.1] * 3 for _ in chunks])
        # Originals must still have embedding=None (frozen + not mutated).
        assert all(c.embedding is None for c in chunks)
        # Result copies are new objects.
        assert result[0].id == original_id
        assert result[0] is not chunks[0]

    def test_preserves_text_and_metadata(self, card: Card) -> None:
        chunks = chunks_from_card(card, deck_id="book-of-thoth")
        result = attach_embeddings(chunks, [[0.0, 0.0, 0.0] for _ in chunks])
        assert result[0].text == chunks[0].text
        assert result[0].section == chunks[0].section
        assert result[0].card_id == chunks[0].card_id

    def test_length_mismatch_raises(self, card: Card) -> None:
        chunks = chunks_from_card(card, deck_id="book-of-thoth")
        with pytest.raises(ValueError, match="length mismatch"):
            attach_embeddings(chunks, [[0.0, 0.0, 0.0]])

    def test_empty_inputs(self) -> None:
        assert attach_embeddings([], []) == []

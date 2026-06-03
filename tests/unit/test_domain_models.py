"""Unit tests for fortune_teller.application.models.domain.

All tests are pure logic — no I/O, no network, no filesystem writes.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import ClassVar

import pytest
from pydantic import HttpUrl, ValidationError

from fortune_teller.application.models.domain import (
    Arcana,
    Card,
    CardInterpretation,
    CardSection,
    CardSectionText,
    Chunk,
    ChunkType,
    DealtCard,
    Deck,
    Orientation,
    Reading,
    Spread,
    SpreadPosition,
    Suit,
)

# ---------------------------------------------------------------------------
# Helpers / shared factories
# ---------------------------------------------------------------------------

_SOURCE_URL = "https://thothreadings.com/the-fool/"
_SPREAD_URL = "https://thothreadings.com/spread-new-moon/"


def _make_card(
    *,
    card_id: str = "the-fool",
    name: str = "The Fool",
    arcana: Arcana = Arcana.MAJOR,
    suit: Suit | None = None,
    sections: list[CardSectionText] | None = None,
) -> Card:
    return Card(
        id=card_id,
        name=name,
        arcana=arcana,
        suit=suit,
        sections=sections or [],
        source_url=HttpUrl(_SOURCE_URL),
    )


def _make_minor_card(
    *,
    card_id: str = "ace-of-wands",
    name: str = "Ace of Wands",
    suit: Suit = Suit.WANDS,
) -> Card:
    return _make_card(card_id=card_id, name=name, arcana=Arcana.MINOR, suit=suit)


def _make_spread_position(index: int, name: str = "Past") -> SpreadPosition:
    return SpreadPosition(
        index=index,
        name=name,
        meaning=f"What was in the {name.lower()}.",
        source_url=HttpUrl(_SPREAD_URL),
    )


def _make_spread(*, position_count: int = 3) -> Spread:
    names = ["Past", "Present", "Future", "Advice", "Outcome"]
    positions = [_make_spread_position(i, names[i]) for i in range(position_count)]
    return Spread(
        id="new-moon-three-card",
        name="New Moon Three-Card Spread",
        positions=positions,
    )


def _make_dealt(card_id: str = "the-fool", position_index: int = 0) -> DealtCard:
    return DealtCard(
        card_id=card_id,
        orientation=Orientation.UPRIGHT,
        position_index=position_index,
    )


# ---------------------------------------------------------------------------
# Orientation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestOrientation:
    def test_values(self) -> None:
        assert Orientation.UPRIGHT == "upright"
        assert Orientation.REVERSED == "reversed"

    def test_round_trips_json(self) -> None:
        for member in Orientation:
            assert Orientation(member.value) == member


# ---------------------------------------------------------------------------
# CardSection — all 11 members present
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCardSection:
    _EXPECTED: ClassVar[set[str]] = {
        "overall",
        "drive",
        "light",
        "shadow",
        "reversed",
        "keywords",
        "advice",
        "question",
        "proposal",
        "confirmation",
        "affirmation",
    }

    def test_has_exactly_11_members(self) -> None:
        assert len(CardSection) == 11

    def test_expected_values_present(self) -> None:
        assert {m.value for m in CardSection} == self._EXPECTED

    def test_round_trips_json(self) -> None:
        for member in CardSection:
            assert CardSection(member.value) == member


# ---------------------------------------------------------------------------
# CardSectionText
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCardSectionText:
    def test_creates_successfully(self) -> None:
        cst = CardSectionText(section=CardSection.DRIVE, text="Forward motion.")
        assert cst.section == CardSection.DRIVE
        assert cst.text == "Forward motion."

    def test_empty_text_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CardSectionText(section=CardSection.DRIVE, text="")

    def test_is_immutable(self) -> None:
        cst = CardSectionText(section=CardSection.DRIVE, text="Forward motion.")
        with pytest.raises(ValidationError):
            cst.text = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Card
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCard:
    def test_major_arcana_no_suit(self) -> None:
        card = _make_card()
        assert card.arcana == Arcana.MAJOR
        assert card.suit is None

    def test_minor_arcana_with_suit(self) -> None:
        card = _make_minor_card()
        assert card.arcana == Arcana.MINOR
        assert card.suit == Suit.WANDS

    def test_minor_arcana_without_suit_raises(self) -> None:
        with pytest.raises(ValidationError, match="must have a suit"):
            Card(
                id="ace-of-wands",
                name="Ace of Wands",
                arcana=Arcana.MINOR,
                suit=None,
                source_url=HttpUrl(_SOURCE_URL),
            )

    def test_major_arcana_with_suit_raises(self) -> None:
        with pytest.raises(ValidationError, match="must not have a suit"):
            Card(
                id="the-fool",
                name="The Fool",
                arcana=Arcana.MAJOR,
                suit=Suit.WANDS,
                source_url=HttpUrl(_SOURCE_URL),
            )

    def test_empty_id_raises(self) -> None:
        with pytest.raises(ValidationError):
            Card(
                id="",
                name="The Fool",
                arcana=Arcana.MAJOR,
                source_url=HttpUrl(_SOURCE_URL),
            )

    def test_section_text_returns_text(self) -> None:
        sections = [
            CardSectionText(section=CardSection.DRIVE, text="Forward motion."),
            CardSectionText(section=CardSection.LIGHT, text="Bright side."),
        ]
        card = _make_card(sections=sections)
        assert card.section_text(CardSection.DRIVE) == "Forward motion."
        assert card.section_text(CardSection.LIGHT) == "Bright side."

    def test_section_text_returns_none_for_missing(self) -> None:
        card = _make_card(sections=[])
        assert card.section_text(CardSection.AFFIRMATION) is None

    def test_is_immutable(self) -> None:
        card = _make_card()
        with pytest.raises(ValidationError):
            card.name = "changed"  # type: ignore[misc]

    def test_json_round_trip(self) -> None:
        card = _make_card()
        reloaded = Card.model_validate_json(card.model_dump_json())
        assert reloaded.id == card.id
        assert reloaded.arcana == card.arcana


# ---------------------------------------------------------------------------
# Deck
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDeck:
    def _make_deck(self, count: int = 3) -> Deck:
        cards = [_make_card(card_id=f"card-{i}", name=f"Card {i}") for i in range(count)]
        return Deck(id="test-deck", name="Test Deck", cards=cards)

    def test_card_by_id_returns_correct_card(self) -> None:
        deck = self._make_deck()
        card = deck.card_by_id("card-1")
        assert card.name == "Card 1"

    def test_card_by_id_raises_for_unknown(self) -> None:
        deck = self._make_deck()
        with pytest.raises(KeyError, match="nope"):
            deck.card_by_id("nope")

    def test_card_ids_returns_all_ids(self) -> None:
        deck = self._make_deck(3)
        assert deck.card_ids() == ["card-0", "card-1", "card-2"]

    def test_empty_deck_allowed(self) -> None:
        deck = Deck(id="empty", name="Empty Deck")
        assert deck.card_ids() == []

    def test_is_immutable(self) -> None:
        deck = self._make_deck()
        with pytest.raises(ValidationError):
            deck.name = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# SpreadPosition
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSpreadPosition:
    def test_creates_successfully(self) -> None:
        pos = _make_spread_position(0, "Past")
        assert pos.index == 0
        assert pos.name == "Past"

    def test_negative_index_raises(self) -> None:
        with pytest.raises(ValidationError):
            SpreadPosition(
                index=-1,
                name="Bad",
                meaning="Bad meaning.",
                source_url=HttpUrl(_SPREAD_URL),
            )


# ---------------------------------------------------------------------------
# Spread
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSpread:
    def test_three_position_spread_valid(self) -> None:
        spread = _make_spread(position_count=3)
        assert len(spread.positions) == 3

    def test_non_contiguous_indices_raises(self) -> None:
        positions = [
            _make_spread_position(0, "Past"),
            _make_spread_position(2, "Future"),  # index 1 is missing
        ]
        with pytest.raises(ValidationError, match="contiguous"):
            Spread(id="bad", name="Bad Spread", positions=positions)

    def test_duplicate_indices_raises(self) -> None:
        positions = [
            _make_spread_position(0, "Past"),
            _make_spread_position(0, "Also Past"),  # duplicate 0
        ]
        with pytest.raises(ValidationError, match="contiguous"):
            Spread(id="bad", name="Bad Spread", positions=positions)

    def test_position_by_index_returns_correct(self) -> None:
        spread = _make_spread()
        pos = spread.position_by_index(1)
        assert pos.name == "Present"

    def test_position_by_index_raises_for_missing(self) -> None:
        spread = _make_spread()
        with pytest.raises(KeyError):
            spread.position_by_index(99)

    def test_empty_spread_valid(self) -> None:
        spread = Spread(id="empty", name="Empty Spread")
        assert spread.positions == []

    def test_json_round_trip(self) -> None:
        spread = _make_spread()
        reloaded = Spread.model_validate_json(spread.model_dump_json())
        assert reloaded.id == spread.id
        assert len(reloaded.positions) == len(spread.positions)


# ---------------------------------------------------------------------------
# DealtCard
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDealtCard:
    def test_creates_upright(self) -> None:
        dc = _make_dealt()
        assert dc.orientation == Orientation.UPRIGHT

    def test_creates_reversed(self) -> None:
        dc = DealtCard(card_id="the-fool", orientation=Orientation.REVERSED, position_index=0)
        assert dc.orientation == Orientation.REVERSED

    def test_negative_position_index_raises(self) -> None:
        with pytest.raises(ValidationError):
            DealtCard(card_id="the-fool", orientation=Orientation.UPRIGHT, position_index=-1)

    def test_empty_card_id_raises(self) -> None:
        with pytest.raises(ValidationError):
            DealtCard(card_id="", orientation=Orientation.UPRIGHT, position_index=0)


# ---------------------------------------------------------------------------
# CardInterpretation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCardInterpretation:
    def test_creates_successfully(self) -> None:
        interp = CardInterpretation(
            dealt=_make_dealt(),
            card_name="The Fool",
            position_name="Past",
            text="This card represents new beginnings.",
        )
        assert interp.card_name == "The Fool"

    def test_empty_text_raises(self) -> None:
        with pytest.raises(ValidationError):
            CardInterpretation(
                dealt=_make_dealt(),
                card_name="The Fool",
                position_name="Past",
                text="",
            )


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestReading:
    def test_creates_with_defaults(self) -> None:
        reading = Reading(deck_id="book-of-thoth", spread_id="new-moon-three-card")
        assert isinstance(reading.id, uuid.UUID)
        assert reading.dealt == []
        assert reading.summary == ""
        assert reading.created_at.tzinfo is not None

    def test_duplicate_card_ids_raise(self) -> None:
        dealt = [
            _make_dealt("the-fool", 0),
            _make_dealt("the-fool", 1),  # duplicate
        ]
        with pytest.raises(ValidationError, match="duplicate"):
            Reading(
                deck_id="book-of-thoth",
                spread_id="new-moon-three-card",
                dealt=dealt,
            )

    def test_distinct_card_ids_valid(self) -> None:
        dealt = [
            _make_dealt("the-fool", 0),
            _make_dealt("the-magus", 1),
            _make_dealt("the-empress", 2),
        ]
        reading = Reading(
            deck_id="book-of-thoth",
            spread_id="new-moon-three-card",
            dealt=dealt,
        )
        assert len(reading.dealt) == 3

    def test_created_at_is_utc(self) -> None:
        reading = Reading(deck_id="book-of-thoth", spread_id="new-moon-three-card")
        assert reading.created_at.tzinfo == UTC

    def test_custom_id_preserved(self) -> None:
        fixed_id = uuid.UUID("12345678-1234-5678-1234-567812345678")
        reading = Reading(
            id=fixed_id,
            deck_id="book-of-thoth",
            spread_id="new-moon-three-card",
        )
        assert reading.id == fixed_id

    def test_json_round_trip(self) -> None:
        dealt = [_make_dealt("the-fool", 0)]
        reading = Reading(
            deck_id="book-of-thoth",
            spread_id="new-moon-three-card",
            dealt=dealt,
            summary="A brief summary.",
        )
        reloaded = Reading.model_validate_json(reading.model_dump_json())
        assert reloaded.id == reading.id
        assert reloaded.dealt[0].card_id == "the-fool"
        assert reloaded.summary == "A brief summary."

    def test_empty_deck_id_raises(self) -> None:
        with pytest.raises(ValidationError):
            Reading(deck_id="", spread_id="new-moon-three-card")


# ---------------------------------------------------------------------------
# Chunk
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestChunk:
    def test_card_section_chunk(self) -> None:
        chunk = Chunk(
            chunk_type=ChunkType.CARD_SECTION,
            deck_id="book-of-thoth",
            card_id="the-fool",
            card_name="The Fool",
            section=CardSection.DRIVE,
            source_url="https://thothreadings.com/the-fool/",
            text="The drive of The Fool is pure potential.",
        )
        assert chunk.chunk_type == ChunkType.CARD_SECTION
        assert chunk.embedding is None
        assert isinstance(chunk.id, uuid.UUID)

    def test_spread_position_chunk(self) -> None:
        chunk = Chunk(
            chunk_type=ChunkType.SPREAD_POSITION,
            spread_id="new-moon-three-card",
            position_index=0,
            source_url="https://thothreadings.com/spread-new-moon/",
            text="Past: What has been.",
        )
        assert chunk.spread_id == "new-moon-three-card"
        assert chunk.card_id is None

    def test_with_embedding(self) -> None:
        embedding = [0.1] * 384
        chunk = Chunk(
            chunk_type=ChunkType.CARD_SECTION,
            source_url="https://example.com/",
            text="Some text.",
            embedding=embedding,
        )
        assert chunk.embedding is not None
        assert len(chunk.embedding) == 384

    def test_empty_text_raises(self) -> None:
        with pytest.raises(ValidationError):
            Chunk(
                chunk_type=ChunkType.CARD_SECTION,
                source_url="https://example.com/",
                text="",
            )

    def test_is_immutable(self) -> None:
        chunk = Chunk(
            chunk_type=ChunkType.CARD_SECTION,
            source_url="https://example.com/",
            text="Some text.",
        )
        with pytest.raises(ValidationError):
            chunk.text = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Cross-model: Suit enum completeness
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSuitEnum:
    def test_four_suits(self) -> None:
        assert len(Suit) == 4

    def test_expected_values(self) -> None:
        assert {s.value for s in Suit} == {"wands", "cups", "swords", "disks"}

    def test_round_trips(self) -> None:
        for suit in Suit:
            assert Suit(suit.value) == suit


# ---------------------------------------------------------------------------
# created_at default: timezone-aware
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_reading_created_at_is_timezone_aware() -> None:
    reading = Reading(deck_id="d", spread_id="s")
    assert isinstance(reading.created_at, datetime)
    assert reading.created_at.tzinfo is not None

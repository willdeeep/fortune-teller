"""Pydantic v2 domain models for Fortune Teller.

All models are immutable by default (``model_config = ConfigDict(frozen=True)``).
Use ``model.model_copy(update={...})`` when you need a modified copy.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Orientation(StrEnum):
    """Card orientation when dealt."""

    UPRIGHT = "upright"
    REVERSED = "reversed"


class Arcana(StrEnum):
    """Major or minor arcana."""

    MAJOR = "major"
    MINOR = "minor"


class Suit(StrEnum):
    """The four suits of the minor arcana. Thoth uses Disks; Rider-Waite uses Pentacles."""

    WANDS = "wands"
    CUPS = "cups"
    SWORDS = "swords"
    DISKS = "disks"
    PENTACLES = "pentacles"


class CardSection(StrEnum):
    """Structured section types scraped from a card definition page."""

    OVERALL = "overall"
    DRIVE = "drive"
    LIGHT = "light"
    SHADOW = "shadow"
    REVERSED = "reversed"
    KEYWORDS = "keywords"
    ADVICE = "advice"
    QUESTION = "question"
    PROPOSAL = "proposal"
    CONFIRMATION = "confirmation"
    AFFIRMATION = "affirmation"
    REINFORCING = "reinforcing"
    OPPOSING = "opposing"


class ChunkType(StrEnum):
    """Discriminator for vector store chunks."""

    CARD_SECTION = "card_section"
    SPREAD_POSITION = "spread_position"


# ---------------------------------------------------------------------------
# Card
# ---------------------------------------------------------------------------


class CardSectionText(BaseModel):
    """A single labelled text section belonging to a card definition."""

    model_config = ConfigDict(frozen=True)

    section: CardSection
    text: Annotated[str, Field(min_length=1)]


class Card(BaseModel):
    """A single Tarot card with all its structured definition text."""

    model_config = ConfigDict(frozen=True)

    id: Annotated[str, Field(min_length=1)]  # slug, e.g. "the-fool"
    name: Annotated[str, Field(min_length=1)]  # "The Fool"
    arcana: Arcana
    suit: Suit | None = None  # None for major arcana
    number: int | None = None  # 0-21 major, 1-14 minor
    sections: list[CardSectionText] = Field(default_factory=list)
    reinforcing_ids: list[str] = Field(default_factory=list)  # card IDs that amplify this card
    opposing_ids: list[str] = Field(default_factory=list)  # card IDs that challenge this card
    source_url: HttpUrl
    image_url: str | None = None  # full-res artwork URL, parsed from page

    @model_validator(mode="after")
    def _validate_suit_matches_arcana(self) -> Card:
        if self.arcana == Arcana.MINOR and self.suit is None:
            raise ValueError("Minor arcana card must have a suit.")
        if self.arcana == Arcana.MAJOR and self.suit is not None:
            raise ValueError("Major arcana card must not have a suit.")
        return self

    def section_text(self, section: CardSection) -> str | None:
        """Return the text for *section*, or ``None`` if not present."""
        for s in self.sections:
            if s.section == section:
                return s.text
        return None


# ---------------------------------------------------------------------------
# Deck
# ---------------------------------------------------------------------------


class Deck(BaseModel):
    """A named collection of Tarot cards."""

    model_config = ConfigDict(frozen=True)

    id: Annotated[str, Field(min_length=1)]  # "book-of-thoth"
    name: Annotated[str, Field(min_length=1)]  # "Book of Thoth"
    cards: list[Card] = Field(default_factory=list)
    source_url: str | None = None
    attribution: str | None = None
    description: str | None = None

    def card_by_id(self, card_id: str) -> Card:
        """Return the card with *card_id*.

        Raises:
            KeyError: if no card with that id exists in the deck.
        """
        for card in self.cards:
            if card.id == card_id:
                return card
        raise KeyError(f"Card not found in deck '{self.id}': {card_id!r}")

    def card_ids(self) -> list[str]:
        """Return an ordered list of all card IDs in the deck."""
        return [c.id for c in self.cards]


# ---------------------------------------------------------------------------
# Spread
# ---------------------------------------------------------------------------


class SpreadPosition(BaseModel):
    """One named position within a spread (e.g. 'Past', 'Present', 'Future').

    Optional layout fields (``row``, ``col``, ``rotation``) enable 2D spread
    rendering (e.g. Celtic Cross).  When absent, the UI falls back to a simple
    linear row layout.
    """

    model_config = ConfigDict(frozen=True)

    index: Annotated[int, Field(ge=0)]  # 0-based
    name: Annotated[str, Field(min_length=1)]
    meaning: Annotated[str, Field(min_length=1)]
    source_url: HttpUrl
    row: int | None = None  # grid placement (0-based)
    col: int | None = None  # grid placement (0-based)
    rotation: int = 0  # degrees; 90 for the crossing card


class Spread(BaseModel):
    """A named Tarot spread with an ordered list of positions."""

    model_config = ConfigDict(frozen=True)

    id: Annotated[str, Field(min_length=1)]  # "new-moon-three-card"
    name: Annotated[str, Field(min_length=1)]  # "New Moon Three-Card Spread"
    positions: list[SpreadPosition] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_contiguous_indices(self) -> Spread:
        indices = sorted(p.index for p in self.positions)
        expected = list(range(len(self.positions)))
        if indices != expected:
            raise ValueError(f"SpreadPosition indices must be contiguous from 0, got {indices!r}.")
        return self

    def position_by_index(self, index: int) -> SpreadPosition:
        """Return the position at *index*.

        Raises:
            KeyError: if no position with that index exists.
        """
        for pos in self.positions:
            if pos.index == index:
                return pos
        raise KeyError(f"No position with index {index} in spread '{self.id}'.")


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


class DealtCard(BaseModel):
    """A card as dealt into a specific reading position."""

    model_config = ConfigDict(frozen=True)

    card_id: Annotated[str, Field(min_length=1)]
    orientation: Orientation
    position_index: Annotated[int, Field(ge=0)]


class CardInterpretation(BaseModel):
    """LLM-generated interpretation for a single dealt card."""

    model_config = ConfigDict(frozen=True)

    dealt: DealtCard
    card_name: Annotated[str, Field(min_length=1)]
    position_name: Annotated[str, Field(min_length=1)]
    text: Annotated[str, Field(min_length=1)]


class Reading(BaseModel):
    """A complete Tarot reading session."""

    model_config = ConfigDict(frozen=True)

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    deck_id: Annotated[str, Field(min_length=1)]
    spread_id: Annotated[str, Field(min_length=1)]
    dealt: list[DealtCard] = Field(default_factory=list)
    per_card: list[CardInterpretation] = Field(default_factory=list)
    summary: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))

    @model_validator(mode="after")
    def _validate_no_duplicate_card_ids(self) -> Reading:
        card_ids = [d.card_id for d in self.dealt]
        if len(card_ids) != len(set(card_ids)):
            duplicates = {cid for cid in card_ids if card_ids.count(cid) > 1}
            raise ValueError(
                f"A reading cannot contain duplicate cards. Duplicates: {duplicates!r}"
            )
        return self


# ---------------------------------------------------------------------------
# Reading history list item (lightweight row for UI)
# ---------------------------------------------------------------------------


class ReadingListItem(BaseModel):
    """Lightweight row for the reading-history list UI.

    Does *not* contain the full payload — just enough to render a list
    row. Retrieve the full :class:`Reading` via
    :meth:`SQLiteStore.get` when the user selects a row.
    """

    model_config = ConfigDict(frozen=True)

    id: uuid.UUID
    deck_id: str
    spread_id: str
    card_names: list[str]
    summary_preview: str
    created_at: datetime


# ---------------------------------------------------------------------------
# Vector store chunk
# ---------------------------------------------------------------------------


class Chunk(BaseModel):
    """A text chunk ready for embedding and storage in DuckDB.

    ``embedding`` is ``None`` before the embed step and a list of floats after.
    """

    model_config = ConfigDict(frozen=True)

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    chunk_type: ChunkType
    deck_id: str | None = None
    card_id: str | None = None
    card_name: str | None = None
    section: CardSection | None = None
    spread_id: str | None = None
    position_index: int | None = None
    source_url: Annotated[str, Field(min_length=1)]
    text: Annotated[str, Field(min_length=1)]
    embedding: list[float] | None = None  # populated after ft-embed


# ---------------------------------------------------------------------------
# Vector search result
# ---------------------------------------------------------------------------


class SearchHit(BaseModel):
    """A single hit from a vector-store search.

    Pairs a :class:`Chunk` (with all its metadata) with its cosine similarity
    score against the query embedding. Score is in ``[-1.0, 1.0]``; higher is
    more similar.
    """

    model_config = ConfigDict(frozen=True)

    chunk: Chunk
    score: float


# ---------------------------------------------------------------------------
# Convenience type aliases
# ---------------------------------------------------------------------------

#: Literal type accepted wherever only major arcana is valid.
MajorArcana = Literal["major"]
#: Literal type accepted wherever only minor arcana is valid.
MinorArcana = Literal["minor"]

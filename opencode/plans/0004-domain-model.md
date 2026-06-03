# 0004 — Domain Model (pydantic v2)

Module: `fortune_teller.application.models.domain`

## Full Model Definitions

```python
from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Orientation(StrEnum):
    UPRIGHT = "upright"
    REVERSED = "reversed"


class Arcana(StrEnum):
    MAJOR = "major"
    MINOR = "minor"


class Suit(StrEnum):
    WANDS = "wands"
    CUPS = "cups"
    SWORDS = "swords"
    DISKS = "disks"


class CardSection(StrEnum):
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


# ---------------------------------------------------------------------------
# Card
# ---------------------------------------------------------------------------

class CardSectionText(BaseModel):
    """A single structured section of text for a card."""
    section: CardSection
    text: str


class Card(BaseModel):
    """A single Tarot card with all its structured definition text."""
    id: str                           # slug, e.g. "the-fool"
    name: str                         # "The Fool"
    arcana: Arcana
    suit: Suit | None = None          # None for major arcana
    number: int | None = None         # 0–21 for major, 1–14 for minor
    sections: list[CardSectionText]
    source_url: HttpUrl

    @model_validator(mode="after")
    def validate_suit_for_minor(self) -> "Card":
        if self.arcana == Arcana.MINOR and self.suit is None:
            raise ValueError("Minor arcana card must have a suit")
        if self.arcana == Arcana.MAJOR and self.suit is not None:
            raise ValueError("Major arcana card must not have a suit")
        return self


# ---------------------------------------------------------------------------
# Deck
# ---------------------------------------------------------------------------

class Deck(BaseModel):
    """A named collection of 78 Tarot cards."""
    id: str                    # "book-of-thoth"
    name: str                  # "Book of Thoth"
    cards: list[Card]

    @model_validator(mode="after")
    def validate_card_count(self) -> "Deck":
        # Allow partial decks during development (e.g. fixtures)
        # but warn if not exactly 78 in production
        return self

    def card_by_id(self, card_id: str) -> Card:
        for card in self.cards:
            if card.id == card_id:
                return card
        raise KeyError(f"Card not found: {card_id}")


# ---------------------------------------------------------------------------
# Spread
# ---------------------------------------------------------------------------

class SpreadPosition(BaseModel):
    """One position within a spread (e.g. 'Past', 'Present', 'Future')."""
    index: int           # 0-based
    name: str
    meaning: str
    source_url: HttpUrl


class Spread(BaseModel):
    """A named Tarot spread with ordered positions."""
    id: str              # "new-moon-three-card"
    name: str            # "New Moon Three-Card Spread"
    positions: list[SpreadPosition]

    @model_validator(mode="after")
    def validate_position_indices(self) -> "Spread":
        indices = [p.index for p in self.positions]
        expected = list(range(len(self.positions)))
        if sorted(indices) != expected:
            raise ValueError(
                f"SpreadPosition indices must be contiguous from 0, got {indices}"
            )
        return self


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

class DealtCard(BaseModel):
    """A card as it was dealt in a specific reading position."""
    card_id: str
    orientation: Orientation
    position_index: int


class CardInterpretation(BaseModel):
    """LLM-generated interpretation for a single dealt card."""
    dealt: DealtCard
    card_name: str
    position_name: str
    text: str


class Reading(BaseModel):
    """A complete Tarot reading session."""
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    deck_id: str
    spread_id: str
    dealt: list[DealtCard] = Field(default_factory=list)
    per_card: list[CardInterpretation] = Field(default_factory=list)
    summary: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @model_validator(mode="after")
    def validate_no_duplicate_cards(self) -> "Reading":
        card_ids = [d.card_id for d in self.dealt]
        if len(card_ids) != len(set(card_ids)):
            raise ValueError("A reading cannot contain duplicate cards")
        return self


# ---------------------------------------------------------------------------
# Vector store chunk (internal, not stored via pydantic but validated here)
# ---------------------------------------------------------------------------

class ChunkType(StrEnum):
    CARD_SECTION = "card_section"
    SPREAD_POSITION = "spread_position"


class Chunk(BaseModel):
    """A text chunk ready for embedding and storage in DuckDB."""
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    chunk_type: ChunkType
    deck_id: str | None = None
    card_id: str | None = None
    card_name: str | None = None
    section: CardSection | None = None
    spread_id: str | None = None
    position_index: int | None = None
    source_url: str
    text: str
    embedding: list[float] | None = None  # populated after embed step
```

## Unit Test Targets

- `Card` with `arcana=minor` and no `suit` raises `ValidationError`.
- `Card` with `arcana=major` and a `suit` raises `ValidationError`.
- `Spread` with non-contiguous indices raises `ValidationError`.
- `Reading` with duplicate `card_id`s raises `ValidationError`.
- All `StrEnum` values round-trip through JSON serialisation.
- `Deck.card_by_id` returns the correct card; raises `KeyError` for unknown id.

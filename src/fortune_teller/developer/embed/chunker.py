"""Convert parsed ``Card`` and ``Spread`` JSON into :class:`Chunk` objects.

Each :class:`~fortune_teller.application.models.domain.CardSectionText`
becomes one :class:`Chunk` of type ``CARD_SECTION``; each
:class:`~fortune_teller.application.models.domain.SpreadPosition` becomes
one :class:`Chunk` of type ``SPREAD_POSITION``. Embeddings are left as
``None`` — the ``ft-embed`` CLI populates them after the chunker runs.

One chunk per (small) section is the chosen granularity (see
``docs/architecture.md``): it enables precise retrieval of, e.g., only
the "reversed" section when a card is dealt inverted.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from fortune_teller.application.models.domain import (
    Card,
    CardSection,
    Chunk,
    ChunkType,
    Spread,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

#: Namespace for deterministic chunk IDs. Never change it: doing so re-keys
#: every chunk (harmless with the default ``ft-build-index --rebuild``, but it
#: breaks idempotent ``--no-rebuild`` appends against an existing index).
CHUNK_ID_NAMESPACE = uuid.uuid5(
    uuid.NAMESPACE_URL, "https://github.com/willdeeep/fortune-teller#chunk"
)


def _card_chunk_id(deck_id: str, card_id: str, section: CardSection) -> uuid.UUID:
    """Return the stable ID for one card section within one deck."""
    return uuid.uuid5(CHUNK_ID_NAMESPACE, f"card_section:{deck_id}:{card_id}:{section.value}")


def _spread_chunk_id(spread_id: str, position_index: int) -> uuid.UUID:
    """Return the stable ID for one spread position."""
    return uuid.uuid5(CHUNK_ID_NAMESPACE, f"spread_position:{spread_id}:{position_index}")


def chunks_from_card(card: Card, deck_id: str) -> list[Chunk]:
    """Convert a single :class:`Card` into one chunk per section.

    Chunk IDs are deterministic (``uuid5`` of deck, card and section), so
    re-embedding the same card yields the same IDs and the store's
    ``INSERT OR REPLACE`` upsert stays idempotent.

    Args:
        card:   The parsed card to chunk.
        deck_id: Deck identifier, attached to every emitted chunk.

    Returns:
        A list of :class:`Chunk` objects, one per section, in the order
        the sections appear on the card. The list is empty if the card
        has no sections.

    Raises:
        ValueError: If the card has two sections of the same kind (their IDs
            would collide and one would silently overwrite the other).
    """
    seen: set[CardSection] = set()
    for section in card.sections:
        if section.section in seen:
            raise ValueError(
                f"Card {card.id!r} has duplicate section {section.section.value!r}; "
                "chunk IDs must be unique per (deck, card, section)."
            )
        seen.add(section.section)

    return [
        Chunk(
            id=_card_chunk_id(deck_id, card.id, section.section),
            chunk_type=ChunkType.CARD_SECTION,
            deck_id=deck_id,
            card_id=card.id,
            card_name=card.name,
            section=section.section,
            source_url=str(card.source_url),
            text=section.text,
        )
        for section in card.sections
    ]


def chunks_from_spread(spread: Spread) -> list[Chunk]:
    """Convert a :class:`Spread` into one chunk per position.

    Chunk IDs are deterministic (``uuid5`` of spread and position index).

    Returns:
        A list of :class:`Chunk` objects, one per spread position, in
        index order. The list is empty if the spread has no positions.
    """
    return [
        Chunk(
            id=_spread_chunk_id(spread.id, position.index),
            chunk_type=ChunkType.SPREAD_POSITION,
            spread_id=spread.id,
            position_index=position.index,
            source_url=str(position.source_url),
            text=f"{position.name}: {position.meaning}",
        )
        for position in spread.positions
    ]


def attach_embeddings(
    chunks: Sequence[Chunk],
    embeddings: Sequence[Sequence[float]],
) -> list[Chunk]:
    """Return new :class:`Chunk` objects with *embeddings* attached.

    Pairs *chunks* and *embeddings* positionally. Returns frozen copies
    (one per input chunk) with the corresponding vector filled in. Used
    by ``ft-embed`` after the :class:`Embedder` has vectorised the
    chunk text.

    Args:
        chunks:     Source chunks (frozen).
        embeddings: One embedding vector per chunk, in the same order.

    Returns:
        A new list of :class:`Chunk` objects, each with ``embedding`` set.

    Raises:
        ValueError: If the two sequences have different lengths.
    """
    if len(chunks) != len(embeddings):
        raise ValueError(f"chunks/embeddings length mismatch: {len(chunks)} vs {len(embeddings)}")
    return [
        chunk.model_copy(update={"embedding": list(emb)})
        for chunk, emb in zip(chunks, embeddings, strict=True)
    ]

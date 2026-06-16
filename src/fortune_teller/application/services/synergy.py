"""Orientation-aware synergy computation for Tarot readings.

The XOR rule: store the *base* relationship (Upright↔Upright). At reading
time, each reversal flips the relationship:

- Both upright or both reversed → base relationship unchanged.
- Exactly one reversed → relationship flips (reinforce↔oppose).

This module provides the pure function for the XOR rule and a synergy
computation that finds all reinforce/oppose pairs among dealt cards.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from fortune_teller.application.models.domain import DealtCard, Deck, Orientation


@dataclass(frozen=True)
class SynergyHit:
    """A reinforce/oppose relationship between two dealt cards.

    Attributes:
        card_id_a: First card ID (lexicographically smaller of the pair).
        card_id_b: Second card ID.
        orientation_a: Orientation of card A when dealt.
        orientation_b: Orientation of card B when dealt.
        base: The stored relationship (reinforce or oppose).
        effective: The relationship after applying the orientation XOR rule.
    """

    card_id_a: str
    card_id_b: str
    orientation_a: Orientation
    orientation_b: Orientation
    base: Literal["reinforce", "oppose"]
    effective: Literal["reinforce", "oppose"]


def effective_relationship(
    base: Literal["reinforce", "oppose"],
    orientation_a: Orientation,
    orientation_b: Orientation,
) -> Literal["reinforce", "oppose"]:
    """Apply the orientation XOR rule to a base relationship.

    If exactly one card is reversed, flip the relationship.
    If both or neither are reversed, keep the base relationship.

    Args:
        base: The stored relationship (Upright↔Upright baseline).
        orientation_a: Orientation of card A.
        orientation_b: Orientation of card B.

    Returns:
        The effective relationship after XOR.
    """
    if (orientation_a == Orientation.REVERSED) != (orientation_b == Orientation.REVERSED):
        return "oppose" if base == "reinforce" else "reinforce"
    return base


def compute_synergies(
    dealt: list[DealtCard],
    deck: Deck,
) -> list[SynergyHit]:
    """Find all reinforce/oppose pairs among dealt cards.

    For each dealt card, check its ``reinforcing_ids`` and ``opposing_ids``
    against the other dealt cards. Apply the orientation XOR rule.
    Return one :class:`SynergyHit` per pair, avoiding duplicates by only
    checking pairs where ``card_id_a < card_id_b`` lexicographically.

    Args:
        dealt: The dealt cards in the reading.
        deck: The deck (used to look up reinforce/oppose IDs).

    Returns:
        List of :class:`SynergyHit` for all matching pairs.
    """
    dealt_by_id: dict[str, DealtCard] = {d.card_id: d for d in dealt}
    seen: set[tuple[str, str, str]] = set()  # (a_id, b_id, base)
    hits: list[SynergyHit] = []

    for d in dealt:
        card = deck.card_by_id(d.card_id)

        # Check reinforcing IDs
        for other_id in card.reinforcing_ids:
            if other_id in dealt_by_id and other_id != d.card_id:
                pair = tuple(sorted([d.card_id, other_id]))
                key = (pair[0], pair[1], "reinforce")
                if key not in seen:
                    seen.add(key)
                    other = dealt_by_id[other_id]
                    a_id, b_id = pair
                    a_orient = d.orientation if d.card_id == a_id else other.orientation
                    b_orient = other.orientation if other.card_id == b_id else d.orientation
                    hits.append(
                        SynergyHit(
                            card_id_a=a_id,
                            card_id_b=b_id,
                            orientation_a=a_orient,
                            orientation_b=b_orient,
                            base="reinforce",
                            effective=effective_relationship("reinforce", a_orient, b_orient),
                        )
                    )

        # Check opposing IDs
        for other_id in card.opposing_ids:
            if other_id in dealt_by_id and other_id != d.card_id:
                pair = tuple(sorted([d.card_id, other_id]))
                key = (pair[0], pair[1], "oppose")
                if key not in seen:
                    seen.add(key)
                    other = dealt_by_id[other_id]
                    a_id, b_id = pair
                    a_orient = d.orientation if d.card_id == a_id else other.orientation
                    b_orient = other.orientation if other.card_id == b_id else d.orientation
                    hits.append(
                        SynergyHit(
                            card_id_a=a_id,
                            card_id_b=b_id,
                            orientation_a=a_orient,
                            orientation_b=b_orient,
                            base="oppose",
                            effective=effective_relationship("oppose", a_orient, b_orient),
                        )
                    )

    return hits

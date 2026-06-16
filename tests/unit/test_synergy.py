"""Unit tests for the synergy computation module.

Tests cover the orientation XOR rule (``effective_relationship``) and the
pair-finding logic (``compute_synergies``).
"""

from __future__ import annotations

import pytest
from pydantic import HttpUrl

from fortune_teller.application.models.domain import (
    Arcana,
    Card,
    DealtCard,
    Deck,
    Orientation,
)
from fortune_teller.application.services.synergy import (
    SynergyHit,
    compute_synergies,
    effective_relationship,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SOURCE_URL = "https://thothreadings.com/the-fool/"


def _make_card(
    card_id: str, name: str, reinforcing: list[str] | None = None, opposing: list[str] | None = None
) -> Card:
    """Build a Card with the given identity and relationship IDs."""
    return Card(
        id=card_id,
        name=name,
        arcana=Arcana.MAJOR,
        sections=[],
        reinforcing_ids=reinforcing or [],
        opposing_ids=opposing or [],
        source_url=HttpUrl(_SOURCE_URL),
    )


def _make_deck(*cards: Card) -> Deck:
    """Build a Deck containing the given cards."""
    return Deck(id="test-deck", name="Test Deck", cards=list(cards))


# ---------------------------------------------------------------------------
# effective_relationship — orientation XOR rule
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEffectiveRelationship:
    """The XOR rule: exactly one reversed flips the relationship."""

    @pytest.mark.parametrize(
        ("base", "orient_a", "orient_b", "expected"),
        [
            # reinforce + both upright → reinforce
            ("reinforce", Orientation.UPRIGHT, Orientation.UPRIGHT, "reinforce"),
            # reinforce + both reversed → reinforce
            ("reinforce", Orientation.REVERSED, Orientation.REVERSED, "reinforce"),
            # reinforce + A reversed → oppose
            ("reinforce", Orientation.REVERSED, Orientation.UPRIGHT, "oppose"),
            # reinforce + B reversed → oppose
            ("reinforce", Orientation.UPRIGHT, Orientation.REVERSED, "oppose"),
            # oppose + both upright → oppose
            ("oppose", Orientation.UPRIGHT, Orientation.UPRIGHT, "oppose"),
            # oppose + both reversed → oppose
            ("oppose", Orientation.REVERSED, Orientation.REVERSED, "oppose"),
            # oppose + A reversed → reinforce
            ("oppose", Orientation.REVERSED, Orientation.UPRIGHT, "reinforce"),
            # oppose + B reversed → reinforce
            ("oppose", Orientation.UPRIGHT, Orientation.REVERSED, "reinforce"),
        ],
    )
    def test_all_orientation_combos(
        self,
        base: str,
        orient_a: Orientation,
        orient_b: Orientation,
        expected: str,
    ) -> None:
        assert effective_relationship(base, orient_a, orient_b) == expected  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# compute_synergies
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestComputeSynergies:
    """Pair-finding among dealt cards."""

    def test_empty_dealt_returns_empty_list(self) -> None:
        deck = _make_deck()
        assert compute_synergies([], deck) == []

    def test_one_reinforce_pair(self) -> None:
        """Two dealt cards where one reinforces the other."""
        card_a = _make_card("card-a", "Card A", reinforcing=["card-b"])
        card_b = _make_card("card-b", "Card B")
        deck = _make_deck(card_a, card_b)
        dealt = [
            DealtCard(card_id="card-a", orientation=Orientation.UPRIGHT, position_index=0),
            DealtCard(card_id="card-b", orientation=Orientation.UPRIGHT, position_index=1),
        ]
        hits = compute_synergies(dealt, deck)
        assert len(hits) == 1
        hit = hits[0]
        assert hit.base == "reinforce"
        assert hit.effective == "reinforce"
        assert {hit.card_id_a, hit.card_id_b} == {"card-a", "card-b"}

    def test_one_oppose_pair(self) -> None:
        """Two dealt cards where one opposes the other."""
        card_a = _make_card("card-a", "Card A", opposing=["card-b"])
        card_b = _make_card("card-b", "Card B")
        deck = _make_deck(card_a, card_b)
        dealt = [
            DealtCard(card_id="card-a", orientation=Orientation.UPRIGHT, position_index=0),
            DealtCard(card_id="card-b", orientation=Orientation.UPRIGHT, position_index=1),
        ]
        hits = compute_synergies(dealt, deck)
        assert len(hits) == 1
        assert hits[0].base == "oppose"
        assert hits[0].effective == "oppose"

    def test_xor_flip_reinforce_becomes_oppose(self) -> None:
        """Reinforce pair with exactly one reversed → effective oppose."""
        card_a = _make_card("card-a", "Card A", reinforcing=["card-b"])
        card_b = _make_card("card-b", "Card B")
        deck = _make_deck(card_a, card_b)
        dealt = [
            DealtCard(card_id="card-a", orientation=Orientation.UPRIGHT, position_index=0),
            DealtCard(card_id="card-b", orientation=Orientation.REVERSED, position_index=1),
        ]
        hits = compute_synergies(dealt, deck)
        assert len(hits) == 1
        assert hits[0].base == "reinforce"
        assert hits[0].effective == "oppose"

    def test_multiple_pairs(self) -> None:
        """Three cards producing multiple reinforce/oppose pairs."""
        card_a = _make_card("card-a", "Card A", reinforcing=["card-b", "card-c"])
        card_b = _make_card("card-b", "Card B", opposing=["card-c"])
        card_c = _make_card("card-c", "Card C")
        deck = _make_deck(card_a, card_b, card_c)
        dealt = [
            DealtCard(card_id="card-a", orientation=Orientation.UPRIGHT, position_index=0),
            DealtCard(card_id="card-b", orientation=Orientation.UPRIGHT, position_index=1),
            DealtCard(card_id="card-c", orientation=Orientation.UPRIGHT, position_index=2),
        ]
        hits = compute_synergies(dealt, deck)
        # card-a reinforces card-b, card-a reinforces card-c, card-b opposes card-c
        assert len(hits) == 3
        bases = {hit.base for hit in hits}
        assert bases == {"reinforce", "oppose"}

    def test_self_referential_ids_ignored(self) -> None:
        """A card that lists itself in reinforcing_ids should not produce a pair."""
        card_a = _make_card("card-a", "Card A", reinforcing=["card-a"])
        deck = _make_deck(card_a)
        dealt = [
            DealtCard(card_id="card-a", orientation=Orientation.UPRIGHT, position_index=0),
        ]
        hits = compute_synergies(dealt, deck)
        assert hits == []

    def test_duplicate_pairs_deduplicated(self) -> None:
        """Card A reinforces B and B opposes A → two separate hits, not four."""
        card_a = _make_card("card-a", "Card A", reinforcing=["card-b"], opposing=["card-b"])
        card_b = _make_card("card-b", "Card B", reinforcing=["card-a"], opposing=["card-a"])
        deck = _make_deck(card_a, card_b)
        dealt = [
            DealtCard(card_id="card-a", orientation=Orientation.UPRIGHT, position_index=0),
            DealtCard(card_id="card-b", orientation=Orientation.UPRIGHT, position_index=1),
        ]
        hits = compute_synergies(dealt, deck)
        # Two relationships: reinforce (from a→b and b→a deduplicated) and oppose
        # a.reinforcing_ids=["card-b"] → reinforce
        # a.opposing_ids=["card-b"] → oppose
        # b.reinforcing_ids=["card-a"] → dedup of reinforce
        # b.opposing_ids=["card-a"] → dedup of oppose
        # So we should have 2 hits: one reinforce, one oppose
        assert len(hits) == 2
        bases = {hit.base for hit in hits}
        assert bases == {"reinforce", "oppose"}


# ---------------------------------------------------------------------------
# SynergyHit dataclass
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSynergyHit:
    def test_creates_successfully(self) -> None:
        hit = SynergyHit(
            card_id_a="the-fool",
            card_id_b="the-magician",
            orientation_a=Orientation.UPRIGHT,
            orientation_b=Orientation.REVERSED,
            base="reinforce",
            effective="oppose",
        )
        assert hit.card_id_a == "the-fool"
        assert hit.card_id_b == "the-magician"
        assert hit.orientation_a == Orientation.UPRIGHT
        assert hit.orientation_b == Orientation.REVERSED
        assert hit.base == "reinforce"
        assert hit.effective == "oppose"

    def test_is_frozen(self) -> None:
        hit = SynergyHit(
            card_id_a="a",
            card_id_b="b",
            orientation_a=Orientation.UPRIGHT,
            orientation_b=Orientation.UPRIGHT,
            base="reinforce",
            effective="reinforce",
        )
        with pytest.raises((TypeError, AttributeError, ValueError)):
            hit.card_id_a = "changed"  # type: ignore[misc]

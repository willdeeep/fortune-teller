"""Unit and property-based tests for DeckSession and ReadingService.

All tests are pure logic — no I/O, no network, no filesystem writes.

Property tests (Hypothesis) verify the central no-replace invariant:
  For any seed and any N ≤ deck-size deals, all dealt card IDs are
  pairwise distinct.
"""

from __future__ import annotations

import random
from typing import ClassVar

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import HttpUrl

from fortune_teller.application.models.domain import (
    Arcana,
    Card,
    CardInterpretation,
    DealtCard,
    Deck,
    Orientation,
    Spread,
    SpreadPosition,
)
from fortune_teller.application.services.deck import DeckExhaustedError, DeckSession
from fortune_teller.application.services.reading import ReadingHandle, ReadingService

# ---------------------------------------------------------------------------
# Fixtures / factories
# ---------------------------------------------------------------------------

_SOURCE_URL = "https://thothreadings.com/the-fool/"
_SPREAD_URL = "https://thothreadings.com/spread-new-moon/"


def _make_card(card_id: str) -> Card:
    return Card(
        id=card_id,
        name=card_id.replace("-", " ").title(),
        arcana=Arcana.MAJOR,
        source_url=HttpUrl(_SOURCE_URL),
    )


def _make_deck(size: int = 10) -> Deck:
    """Return a deck with *size* uniquely-IDed cards."""
    cards = [_make_card(f"card-{i:03d}") for i in range(size)]
    return Deck(id="test-deck", name="Test Deck", cards=cards)


def _make_spread(position_count: int = 3) -> Spread:
    names = ["Past", "Present", "Future", "Advice", "Outcome"]
    positions = [
        SpreadPosition(
            index=i,
            name=names[i],
            meaning=f"Meaning of {names[i]}.",
            source_url=HttpUrl(_SPREAD_URL),
        )
        for i in range(position_count)
    ]
    return Spread(id="test-spread", name="Test Spread", positions=positions)


# Stub chain: returns a canned string without invoking any LLM
class _StubChain:
    def invoke(self, inputs: dict[str, str]) -> str:
        return f"Stub text for {inputs.get('card_name', 'unknown')}."


# ---------------------------------------------------------------------------
# DeckSession — basic behaviour
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDeckSessionBasic:
    def test_remaining_count_starts_at_deck_size(self) -> None:
        deck = _make_deck(10)
        session = DeckSession(deck)
        assert session.remaining_count == 10

    def test_total_count_equals_deck_size(self) -> None:
        deck = _make_deck(7)
        session = DeckSession(deck)
        assert session.total_count == 7

    def test_is_not_exhausted_initially(self) -> None:
        session = DeckSession(_make_deck(5))
        assert not session.is_exhausted

    def test_deal_one_decrements_remaining(self) -> None:
        deck = _make_deck(5)
        session = DeckSession(deck)
        session.deal_one(0)
        assert session.remaining_count == 4

    def test_deal_one_returns_dealt_card(self) -> None:
        deck = _make_deck(5)
        session = DeckSession(deck)
        dealt = session.deal_one(0)
        assert isinstance(dealt, DealtCard)
        assert dealt.position_index == 0
        assert dealt.orientation in (Orientation.UPRIGHT, Orientation.REVERSED)

    def test_deal_one_card_id_is_in_deck(self) -> None:
        deck = _make_deck(5)
        session = DeckSession(deck)
        dealt = session.deal_one(0)
        assert dealt.card_id in deck.card_ids()

    def test_is_exhausted_after_full_deal(self) -> None:
        deck = _make_deck(3)
        session = DeckSession(deck)
        for i in range(3):
            session.deal_one(i)
        assert session.is_exhausted

    def test_deal_on_empty_raises_deck_exhausted(self) -> None:
        deck = _make_deck(2)
        session = DeckSession(deck)
        session.deal_one(0)
        session.deal_one(1)
        with pytest.raises(DeckExhaustedError):
            session.deal_one(2)

    def test_deck_exhausted_message_contains_deck_id(self) -> None:
        deck = _make_deck(1)
        session = DeckSession(deck)
        session.deal_one(0)
        with pytest.raises(DeckExhaustedError, match="test-deck"):
            session.deal_one(1)

    def test_position_index_stored_on_dealt_card(self) -> None:
        deck = _make_deck(5)
        session = DeckSession(deck)
        dealt = session.deal_one(position_index=2)
        assert dealt.position_index == 2


# ---------------------------------------------------------------------------
# DeckSession — no-replace invariant (deterministic)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDeckSessionNoReplace:
    def test_all_dealt_ids_are_unique(self) -> None:
        deck = _make_deck(10)
        session = DeckSession(deck, rng=random.Random(42))
        dealt_ids = [session.deal_one(i).card_id for i in range(10)]
        assert len(dealt_ids) == len(set(dealt_ids))

    def test_dealt_ids_are_subset_of_deck(self) -> None:
        deck = _make_deck(10)
        session = DeckSession(deck, rng=random.Random(99))
        dealt_ids = {session.deal_one(i).card_id for i in range(7)}
        assert dealt_ids.issubset(set(deck.card_ids()))

    def test_dealt_ids_method_tracks_dealt_cards(self) -> None:
        deck = _make_deck(5)
        session = DeckSession(deck, rng=random.Random(0))
        d1 = session.deal_one(0)
        d2 = session.deal_one(1)
        assert session.dealt_ids() == {d1.card_id, d2.card_id}

    def test_remaining_plus_dealt_equals_total(self) -> None:
        deck = _make_deck(8)
        session = DeckSession(deck, rng=random.Random(7))
        for i in range(5):
            session.deal_one(i)
        assert session.remaining_count + len(session.dealt_ids()) == session.total_count


# ---------------------------------------------------------------------------
# DeckSession — reset
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDeckSessionReset:
    def test_reset_restores_full_remaining_count(self) -> None:
        deck = _make_deck(5)
        session = DeckSession(deck)
        for i in range(5):
            session.deal_one(i)
        session.reset()
        assert session.remaining_count == 5

    def test_reset_clears_dealt_ids(self) -> None:
        deck = _make_deck(5)
        session = DeckSession(deck, rng=random.Random(1))
        for i in range(3):
            session.deal_one(i)
        session.reset()
        assert session.dealt_ids() == frozenset()

    def test_reset_allows_dealing_again(self) -> None:
        deck = _make_deck(3)
        session = DeckSession(deck)
        for i in range(3):
            session.deal_one(i)
        session.reset()
        dealt = session.deal_one(0)
        assert isinstance(dealt, DealtCard)

    def test_reset_reshuffles_order(self) -> None:
        """Two resets with different RNG state very likely produce different order."""
        deck = _make_deck(20)
        rng = random.Random(42)
        session = DeckSession(deck, rng=rng)
        order_1 = [session.deal_one(i).card_id for i in range(20)]
        session.reset()
        order_2 = [session.deal_one(i).card_id for i in range(20)]
        # Not strictly guaranteed, but with 20 cards the probability of
        # identical order is 1/20! ≈ 0 — safe to assert inequality.
        assert order_1 != order_2


# ---------------------------------------------------------------------------
# DeckSession — deal_spread
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDeckSessionDealSpread:
    def test_deal_spread_returns_correct_count(self) -> None:
        deck = _make_deck(10)
        session = DeckSession(deck)
        dealt = session.deal_spread(3)
        assert len(dealt) == 3

    def test_deal_spread_positions_are_sequential(self) -> None:
        deck = _make_deck(10)
        session = DeckSession(deck)
        dealt = session.deal_spread(3)
        assert [d.position_index for d in dealt] == [0, 1, 2]

    def test_deal_spread_cards_are_unique(self) -> None:
        deck = _make_deck(10)
        session = DeckSession(deck)
        dealt = session.deal_spread(5)
        ids = [d.card_id for d in dealt]
        assert len(ids) == len(set(ids))

    def test_deal_spread_zero_returns_empty(self) -> None:
        deck = _make_deck(5)
        session = DeckSession(deck)
        assert session.deal_spread(0) == []

    def test_deal_spread_negative_raises(self) -> None:
        deck = _make_deck(5)
        session = DeckSession(deck)
        with pytest.raises(ValueError, match=">="):
            session.deal_spread(-1)

    def test_deal_spread_too_many_raises_exhausted(self) -> None:
        deck = _make_deck(2)
        session = DeckSession(deck)
        with pytest.raises(DeckExhaustedError):
            session.deal_spread(3)


# ---------------------------------------------------------------------------
# DeckSession — orientation distribution
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDeckSessionOrientation:
    """Orientation should be roughly 50/50 over many draws."""

    _TOLERANCE: ClassVar[float] = 0.05  # allow 45%-55% range over 2000 draws
    # Chi-square critical values for 1 degree of freedom:
    #   p=0.05 → 3.841
    #   p=0.01 → 6.635
    #   p=0.001 → 10.828
    _CHI_SQUARE_P05: ClassVar[float] = 3.841

    def test_orientation_is_approximately_uniform(self) -> None:
        deck = _make_deck(1)
        n_trials = 2000
        reversed_count = 0
        for trial in range(n_trials):
            session = DeckSession(deck, rng=random.Random(trial))
            if session.deal_one(0).orientation == Orientation.REVERSED:
                reversed_count += 1
        ratio = reversed_count / n_trials
        assert abs(ratio - 0.5) < self._TOLERANCE, (
            f"Expected ~50% reversed, got {ratio:.1%} over {n_trials} trials"
        )

    def test_orientation_passes_chi_square_at_p005(self) -> None:
        """A chi-square goodness-of-fit test against a 50/50 distribution.

        The orientation flip is a Bernoulli(0.5) trial, so the count of
        REVERSED outcomes over many deals should not significantly
        deviate from n/2. The chi-square statistic with 1 degree of
        freedom at p=0.05 is 3.841 — we expect this test to pass
        comfortably given the deterministic seed sequence.
        """
        deck = _make_deck(1)
        n_trials = 2000
        reversed_count = 0
        for trial in range(n_trials):
            session = DeckSession(deck, rng=random.Random(trial))
            if session.deal_one(0).orientation == Orientation.REVERSED:
                reversed_count += 1

        expected = n_trials / 2
        observed_upright = n_trials - reversed_count
        # Chi-square = Σ (O - E)² / E for each category
        chi_square = (reversed_count - expected) ** 2 / expected + (
            observed_upright - expected
        ) ** 2 / expected
        assert chi_square < self._CHI_SQUARE_P05, (
            f"Chi-square={chi_square:.3f} exceeds p=0.05 critical value "
            f"({self._CHI_SQUARE_P05:.3f}) for {n_trials} trials — "
            f"orientation distribution is not 50/50 (reversed={reversed_count})"
        )


# ---------------------------------------------------------------------------
# DeckSession — seed reproducibility
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDeckSessionSeeds:
    def test_same_seed_produces_same_order(self) -> None:
        deck = _make_deck(10)
        s1 = DeckSession(deck, rng=random.Random(123))
        s2 = DeckSession(deck, rng=random.Random(123))
        ids1 = [s1.deal_one(i).card_id for i in range(10)]
        ids2 = [s2.deal_one(i).card_id for i in range(10)]
        assert ids1 == ids2

    def test_different_seeds_produce_different_order(self) -> None:
        deck = _make_deck(20)
        s1 = DeckSession(deck, rng=random.Random(1))
        s2 = DeckSession(deck, rng=random.Random(2))
        ids1 = [s1.deal_one(i).card_id for i in range(20)]
        ids2 = [s2.deal_one(i).card_id for i in range(20)]
        assert ids1 != ids2


# ---------------------------------------------------------------------------
# Hypothesis property tests — no-replace invariant
# ---------------------------------------------------------------------------


def _full_deck() -> Deck:
    """Return a 78-card deck (reused across Hypothesis examples)."""
    cards = [
        Card(
            id=f"card-{i:03d}",
            name=f"Card {i}",
            arcana=Arcana.MAJOR,
            source_url=HttpUrl(_SOURCE_URL),
        )
        for i in range(78)
    ]
    return Deck(id="full-deck", name="Full Deck", cards=cards)


_FULL_DECK = _full_deck()  # build once


@pytest.mark.unit
@given(
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    n=st.integers(min_value=1, max_value=78),
)
@settings(max_examples=300)
def test_property_dealt_ids_are_unique(seed: int, n: int) -> None:
    """For any seed and any N ≤ 78, all dealt card IDs are pairwise distinct."""
    session = DeckSession(_FULL_DECK, rng=random.Random(seed))
    dealt_ids = [session.deal_one(i).card_id for i in range(n)]
    assert len(dealt_ids) == len(set(dealt_ids)), (
        f"Duplicate card IDs after {n} deals with seed={seed}: {dealt_ids}"
    )


@pytest.mark.unit
@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(max_examples=100)
def test_property_deal_79th_raises(seed: int) -> None:
    """Dealing card #79 always raises DeckExhaustedError regardless of seed."""
    session = DeckSession(_FULL_DECK, rng=random.Random(seed))
    for i in range(78):
        session.deal_one(i)
    with pytest.raises(DeckExhaustedError):
        session.deal_one(78)


@pytest.mark.unit
@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(max_examples=100)
def test_property_reset_restores_full_deck(seed: int) -> None:
    """After reset, remaining_count always equals total_count."""
    session = DeckSession(_FULL_DECK, rng=random.Random(seed))
    # deal a random partial amount then reset
    n = random.Random(seed).randint(0, 78)
    for i in range(n):
        session.deal_one(i)
    session.reset()
    assert session.remaining_count == session.total_count == 78


# ---------------------------------------------------------------------------
# ReadingService
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestReadingService:
    def _make_service(self) -> ReadingService:
        deck = _make_deck(10)
        spread = _make_spread(3)
        return ReadingService(
            deck=deck,
            spread=spread,
            per_card_chain=_StubChain(),
            summary_chain=_StubChain(),
        )

    def test_start_returns_handle(self) -> None:
        svc = self._make_service()
        handle = svc.start(seed=42)
        assert isinstance(handle, ReadingHandle)
        assert handle.deck_session.remaining_count == 10

    def test_start_with_same_seed_gives_same_first_card(self) -> None:
        svc = self._make_service()
        h1 = svc.start(seed=7)
        h2 = svc.start(seed=7)
        assert svc.deal_next(h1).dealt.card_id == svc.deal_next(h2).dealt.card_id

    def test_deal_next_returns_interpretation(self) -> None:
        svc = self._make_service()
        handle = svc.start(seed=1)
        interp = svc.deal_next(handle)
        assert isinstance(interp, CardInterpretation)
        assert interp.text.startswith("Stub text for")

    def test_deal_next_appends_to_handle(self) -> None:
        svc = self._make_service()
        handle = svc.start(seed=1)
        svc.deal_next(handle)
        svc.deal_next(handle)
        assert len(handle.dealt) == 2
        assert len(handle.interpretations) == 2

    def test_deal_next_cards_are_unique(self) -> None:
        svc = self._make_service()
        handle = svc.start(seed=5)
        interps = [svc.deal_next(handle) for _ in range(3)]
        ids = [i.dealt.card_id for i in interps]
        assert len(ids) == len(set(ids))

    def test_deal_next_fills_positions_in_order(self) -> None:
        svc = self._make_service()
        handle = svc.start(seed=3)
        for expected_pos in range(3):
            interp = svc.deal_next(handle)
            assert interp.dealt.position_index == expected_pos

    def test_deal_next_position_name_matches_spread(self) -> None:
        svc = self._make_service()
        handle = svc.start(seed=2)
        interp = svc.deal_next(handle)
        assert interp.position_name == "Past"

    def test_deal_next_beyond_spread_raises(self) -> None:
        svc = self._make_service()
        handle = svc.start(seed=0)
        for _ in range(3):
            svc.deal_next(handle)
        with pytest.raises(RuntimeError, match="already filled"):
            svc.deal_next(handle)

    def test_deal_next_without_chain_raises(self) -> None:
        deck = _make_deck(5)
        spread = _make_spread(3)
        svc = ReadingService(deck=deck, spread=spread)  # no chains
        handle = svc.start()
        with pytest.raises(RuntimeError, match="per_card_chain"):
            svc.deal_next(handle)

    def test_finalize_returns_reading(self) -> None:
        svc = self._make_service()
        handle = svc.start(seed=10)
        for _ in range(3):
            svc.deal_next(handle)
        reading = svc.finalize(handle)
        assert len(reading.dealt) == 3
        assert len(reading.per_card) == 3

    def test_finalize_reading_has_no_duplicate_card_ids(self) -> None:
        svc = self._make_service()
        handle = svc.start(seed=11)
        for _ in range(3):
            svc.deal_next(handle)
        reading = svc.finalize(handle)
        ids = [d.card_id for d in reading.dealt]
        assert len(ids) == len(set(ids))

    def test_finalize_summary_from_stub_chain(self) -> None:
        svc = self._make_service()
        handle = svc.start(seed=12)
        for _ in range(3):
            svc.deal_next(handle)
        reading = svc.finalize(handle)
        assert reading.summary != ""

    def test_finalize_without_summary_chain_gives_empty_summary(self) -> None:
        deck = _make_deck(5)
        spread = _make_spread(3)
        svc = ReadingService(deck=deck, spread=spread, per_card_chain=_StubChain())
        handle = svc.start()
        for _ in range(3):
            svc.deal_next(handle)
        reading = svc.finalize(handle)
        assert reading.summary == ""

    def test_two_readings_from_same_service_are_independent(self) -> None:
        """Starting a second reading via start() gives a fresh deck."""
        svc = self._make_service()
        h1 = svc.start(seed=1)
        for _ in range(3):
            svc.deal_next(h1)

        h2 = svc.start(seed=2)
        # h2 should have a full deck, not affected by h1
        assert h2.deck_session.remaining_count == 10

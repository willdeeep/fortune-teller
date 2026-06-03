# 0005 — Deck & Reading Engine

Modules:
- `fortune_teller.application.services.deck`
- `fortune_teller.application.services.reading`

## Core Invariant

Any card dealt in a reading MUST NOT be available for re-deal within the
same reading. A new reading resets the full deck. This is enforced at the
`DeckSession` level — not in UI or chains.

---

## `DeckSession`

```python
import random
from fortune_teller.application.models.domain import (
    Card, Deck, DealtCard, Orientation,
)


class DeckExhausted(Exception):
    """Raised when deal_one() is called on an empty deck."""


class DeckSession:
    """
    A single-reading view of a Deck.

    Maintains a shuffled list of remaining card IDs. Once a card is dealt
    it cannot be re-dealt in the same session. Call reset() or create a new
    DeckSession to start over.
    """

    def __init__(self, deck: Deck, rng: random.Random | None = None) -> None:
        self._deck = deck
        self._rng = rng or random.Random()
        self._remaining: list[str] = [c.id for c in deck.cards]
        self._rng.shuffle(self._remaining)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def remaining_count(self) -> int:
        return len(self._remaining)

    @property
    def total_count(self) -> int:
        return len(self._deck.cards)

    def deal_one(self, position_index: int) -> DealtCard:
        """
        Deal one card into the given spread position.

        Raises DeckExhausted if the deck is empty.
        """
        if not self._remaining:
            raise DeckExhausted("No cards left in the deck for this reading.")
        card_id = self._remaining.pop()
        orientation = (
            Orientation.REVERSED
            if self._rng.random() < 0.5
            else Orientation.UPRIGHT
        )
        return DealtCard(
            card_id=card_id,
            orientation=orientation,
            position_index=position_index,
        )

    def deal_spread(self, position_count: int) -> list[DealtCard]:
        """Deal one card per position, returning a list in position order."""
        return [self.deal_one(i) for i in range(position_count)]

    def reset(self) -> None:
        """Restore all cards and re-shuffle. Equivalent to starting a new reading."""
        self._remaining = [c.id for c in self._deck.cards]
        self._rng.shuffle(self._remaining)
```

---

## `ReadingService`

```python
from fortune_teller.application.models.domain import Deck, Reading, Spread
from fortune_teller.application.services.deck import DeckSession


class ReadingHandle:
    """Mutable state object for a reading in progress."""
    deck_session: DeckSession
    reading: Reading   # partially populated


class ReadingService:
    """
    Orchestrates a reading: deck management, RAG calls, result assembly.

    In the spike, the chains and vector store are injected as dependencies
    so they can be stubbed in tests.
    """

    def __init__(
        self,
        deck: Deck,
        spread: Spread,
        per_card_chain: ...,   # PerCardChain
        summary_chain: ...,    # SummaryChain
        vector_store: ...,     # VectorStore
    ) -> None: ...

    def start(self, seed: int | None = None) -> ReadingHandle: ...
    def deal_next(self, handle: ReadingHandle) -> CardInterpretation: ...
    def finalize(self, handle: ReadingHandle) -> Reading: ...
```

---

## Hypothesis Property Tests

```python
from hypothesis import given, settings
from hypothesis import strategies as st
from fortune_teller.application.services.deck import DeckSession, DeckExhausted

@given(
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    n=st.integers(min_value=1, max_value=78),
)
@settings(max_examples=200)
def test_dealt_cards_are_unique(seed: int, n: int, small_deck) -> None:
    """For any seed, all dealt card IDs within a single reading are distinct."""
    session = DeckSession(small_deck, rng=random.Random(seed))
    dealt = [session.deal_one(i) for i in range(n)]
    ids = [d.card_id for d in dealt]
    assert len(ids) == len(set(ids))


def test_deal_beyond_deck_raises(small_deck) -> None:
    session = DeckSession(small_deck)
    for i in range(len(small_deck.cards)):
        session.deal_one(i)
    with pytest.raises(DeckExhausted):
        session.deal_one(0)


def test_reset_restores_full_deck(small_deck) -> None:
    session = DeckSession(small_deck)
    for i in range(len(small_deck.cards)):
        session.deal_one(i)
    session.reset()
    assert session.remaining_count == session.total_count
```

## Additional Unit Tests

- Inversion is roughly 50/50 over 10 000 deals (chi-square, p > 0.001).
- `deal_spread(n)` returns exactly `n` cards in positions 0..n-1.
- Two `DeckSession`s with the same seed produce the same deal order.
- Two `DeckSession`s with different seeds very likely produce different order.
- `remaining_count` decrements by 1 per `deal_one` call.

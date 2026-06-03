"""Deck session service — manages no-replace dealing within a single reading.

The central invariant: once a card is dealt it cannot be re-dealt within the
same :class:`DeckSession`.  Call :meth:`DeckSession.reset` (or create a new
instance) to start a fresh reading with all 78 cards available again.
"""

from __future__ import annotations

import random

from fortune_teller.application.models.domain import DealtCard, Deck, Orientation

_REVERSED_PROBABILITY = 0.5


class DeckExhaustedError(Exception):
    """Raised when :meth:`DeckSession.deal_one` is called on an empty deck."""


class DeckSession:
    """A single-reading view of a :class:`Deck`.

    Maintains a shuffled list of remaining card IDs.  Once a card is dealt it
    is removed from the pool and cannot be re-dealt until :meth:`reset` is
    called.

    Args:
        deck: The :class:`Deck` to deal from.
        rng:  Optional :class:`random.Random` instance.  Supply a seeded RNG
              for reproducible tests; omit (or pass ``None``) for true
              randomness.
    """

    def __init__(self, deck: Deck, rng: random.Random | None = None) -> None:
        self._deck = deck
        self._rng: random.Random = rng if rng is not None else random.Random()
        self._remaining: list[str] = [c.id for c in deck.cards]
        self._rng.shuffle(self._remaining)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def remaining_count(self) -> int:
        """Number of cards still available to be dealt."""
        return len(self._remaining)

    @property
    def total_count(self) -> int:
        """Total number of cards in the deck."""
        return len(self._deck.cards)

    @property
    def is_exhausted(self) -> bool:
        """``True`` when no cards remain."""
        return len(self._remaining) == 0

    # ------------------------------------------------------------------
    # Dealing
    # ------------------------------------------------------------------

    def deal_one(self, position_index: int) -> DealtCard:
        """Deal one card into *position_index*.

        The card is removed from the remaining pool immediately.  Orientation
        is chosen uniformly at random (50 % upright, 50 % reversed).

        Args:
            position_index: The 0-based spread position this card fills.

        Returns:
            A :class:`~fortune_teller.application.models.domain.DealtCard`.

        Raises:
            DeckExhaustedError: If no cards remain in the session.
        """
        if not self._remaining:
            raise DeckExhaustedError(f"No cards left in deck '{self._deck.id}' for this reading.")
        card_id = self._remaining.pop()
        orientation = (
            Orientation.REVERSED
            if self._rng.random() < _REVERSED_PROBABILITY
            else Orientation.UPRIGHT
        )
        return DealtCard(
            card_id=card_id,
            orientation=orientation,
            position_index=position_index,
        )

    def deal_spread(self, position_count: int) -> list[DealtCard]:
        """Deal *position_count* cards, one per spread position (0-based).

        Args:
            position_count: Number of cards to deal.

        Returns:
            List of :class:`~fortune_teller.application.models.domain.DealtCard`
            objects in position order (0, 1, …, position_count - 1).

        Raises:
            DeckExhaustedError: If the deck runs out before all positions are filled.
            ValueError: If *position_count* is negative.
        """
        if position_count < 0:
            raise ValueError(f"position_count must be >= 0, got {position_count}.")
        return [self.deal_one(i) for i in range(position_count)]

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Restore all cards and re-shuffle.

        Equivalent to starting a brand-new reading against the same deck.
        """
        self._remaining = [c.id for c in self._deck.cards]
        self._rng.shuffle(self._remaining)

    # ------------------------------------------------------------------
    # Introspection (useful for tests / UI)
    # ------------------------------------------------------------------

    def dealt_ids(self) -> frozenset[str]:
        """Return the set of card IDs that have already been dealt.

        Derived from the difference between the full deck and remaining pool.
        """
        return frozenset(self._deck.card_ids()) - frozenset(self._remaining)

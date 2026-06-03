"""Reading service — orchestrates a Tarot reading end-to-end.

In the spike this service is a thin data-carrying shell: the LangChain chains
and vector store are injected as callables so they can be swapped for stubs
in tests without touching any I/O.

Full chain/store wiring happens in plan steps 0007 and 0008.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from fortune_teller.application.models.domain import (
    CardInterpretation,
    DealtCard,
    Deck,
    Reading,
    Spread,
)
from fortune_teller.application.services.deck import DeckSession

# ---------------------------------------------------------------------------
# Protocol interfaces for chain/store dependencies
# (allows injecting stubs in tests without importing langchain)
# ---------------------------------------------------------------------------


@runtime_checkable
class InterpretationChain(Protocol):
    """Callable that produces a per-card interpretation string."""

    def invoke(self, inputs: dict[str, str]) -> str:
        """Run the chain synchronously and return the interpretation text."""
        ...


@runtime_checkable
class SummaryChain(Protocol):
    """Callable that produces a reading summary string."""

    def invoke(self, inputs: dict[str, str]) -> str:
        """Run the chain synchronously and return the summary text."""
        ...


# ---------------------------------------------------------------------------
# ReadingHandle — mutable in-progress state
# ---------------------------------------------------------------------------


@dataclass
class ReadingHandle:
    """Mutable state for a reading that is currently in progress.

    Passed between :meth:`ReadingService.start`,
    :meth:`ReadingService.deal_next`, and :meth:`ReadingService.finalize`.
    """

    deck_session: DeckSession
    deck_id: str
    spread: Spread
    dealt: list[DealtCard] = field(default_factory=list)
    interpretations: list[CardInterpretation] = field(default_factory=list)


# ---------------------------------------------------------------------------
# ReadingService
# ---------------------------------------------------------------------------


class ReadingService:
    """Orchestrates a Tarot reading: deck management, RAG, result assembly.

    All I/O-heavy dependencies (LangChain chains, vector store) are injected
    so the service remains unit-testable without network or disk access.

    Args:
        deck:              The :class:`~fortune_teller.application.models.domain.Deck`
                           to deal from.
        spread:            The :class:`~fortune_teller.application.models.domain.Spread`
                           to use for this reading.
        per_card_chain:    Chain that returns an interpretation string given a
                           dict of context keys.  Pass ``None`` to defer wiring
                           (raises :exc:`RuntimeError` if called).
        summary_chain:     Chain that returns a summary string.  Pass ``None``
                           to defer wiring.
    """

    def __init__(
        self,
        deck: Deck,
        spread: Spread,
        per_card_chain: InterpretationChain | None = None,
        summary_chain: SummaryChain | None = None,
    ) -> None:
        self._deck = deck
        self._spread = spread
        self._per_card_chain = per_card_chain
        self._summary_chain = summary_chain

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self, seed: int | None = None) -> ReadingHandle:
        """Create a new :class:`ReadingHandle` with a freshly shuffled deck.

        Args:
            seed: Optional RNG seed for reproducible test readings.

        Returns:
            A :class:`ReadingHandle` ready for dealing.
        """
        rng = random.Random(seed)
        session = DeckSession(self._deck, rng=rng)
        return ReadingHandle(
            deck_session=session,
            deck_id=self._deck.id,
            spread=self._spread,
        )

    def deal_next(self, handle: ReadingHandle) -> CardInterpretation:
        """Deal the next card and produce its interpretation.

        The card fills the spread position at index ``len(handle.dealt)``.

        Args:
            handle: An active :class:`ReadingHandle` from :meth:`start`.

        Returns:
            A :class:`~fortune_teller.application.models.domain.CardInterpretation`
            which is also appended to ``handle.interpretations``.

        Raises:
            RuntimeError:  If called after all spread positions are filled, or
                           if no ``per_card_chain`` was provided.
            DeckExhaustedError: (propagated) if the deck runs out unexpectedly.
        """
        position_index = len(handle.dealt)
        if position_index >= len(handle.spread.positions):
            raise RuntimeError(f"All {len(handle.spread.positions)} positions already filled.")

        position = handle.spread.position_by_index(position_index)
        dealt = handle.deck_session.deal_one(position_index)
        handle.dealt.append(dealt)

        card = self._deck.card_by_id(dealt.card_id)

        if self._per_card_chain is None:
            raise RuntimeError(
                "per_card_chain is not configured. Inject a chain or use a stub for testing."
            )

        text = self._per_card_chain.invoke(
            {
                "card_name": card.name,
                "orientation": dealt.orientation.value,
                "position_name": position.name,
                "position_meaning": position.meaning,
            }
        )

        interp = CardInterpretation(
            dealt=dealt,
            card_name=card.name,
            position_name=position.name,
            text=text,
        )
        handle.interpretations.append(interp)
        return interp

    def finalize(self, handle: ReadingHandle) -> Reading:
        """Produce the final :class:`Reading` with a summary.

        Calls the ``summary_chain`` if configured, otherwise leaves
        ``summary`` as an empty string (acceptable in tests).

        Args:
            handle: A :class:`ReadingHandle` where all positions have been
                    dealt (but this is not enforced — partial readings are
                    allowed for future flexibility).

        Returns:
            A complete, immutable :class:`Reading`.
        """
        summary = ""
        if self._summary_chain is not None:
            card_summaries = "\n\n".join(
                f"Position {i.dealt.position_index} — "
                f"{i.position_name} ({i.card_name}, {i.dealt.orientation}):\n"
                f"{i.text}"
                for i in handle.interpretations
            )
            summary = self._summary_chain.invoke(
                {
                    "spread_name": handle.spread.name,
                    "card_summaries": card_summaries,
                }
            )

        return Reading(
            deck_id=handle.deck_id,
            spread_id=handle.spread.id,
            dealt=list(handle.dealt),
            per_card=list(handle.interpretations),
            summary=summary,
        )

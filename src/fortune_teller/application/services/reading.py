"""Reading service — orchestrates a Tarot reading end-to-end.

The service glues together a :class:`Deck`, a :class:`Spread`, and the
two LangChain interpretation chains. All I/O-heavy dependencies
(LangChain chains, vector store, embedder) are injected so the service
remains unit-testable without network or disk access.

Full chain/store wiring happens in :func:`build_reading_service`
(plan 0009).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Protocol, cast, runtime_checkable
from uuid import UUID

from fortune_teller.application.chains.per_card import build_per_card_context
from fortune_teller.application.chains.summary import build_summary_context
from fortune_teller.application.models.domain import (
    CardInterpretation,
    DealtCard,
    Deck,
    Reading,
    ReadingListItem,
    Spread,
)
from fortune_teller.application.services.deck import DeckSession
from fortune_teller.application.services.synergy import compute_synergies

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


@runtime_checkable
class HistoryStore(Protocol):
    """Persists completed readings and provides history queries.

    Injected so the service and UI have no hard dependency on the SQLite
    store; tests pass ``None`` or a stub.
    """

    def save(self, reading: Reading) -> None:
        """Persist a finalised reading."""
        ...

    def get(self, reading_id: UUID) -> Reading | None:
        """Return the full reading by *reading_id*, or ``None``."""
        ...

    def list_recent(self, limit: int = 50) -> list[ReadingListItem]:
        """Return recent readings (metadata only), newest first."""
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

    All I/O-heavy dependencies (LangChain chains, vector store, embedder)
    are injected so the service remains unit-testable without network or
    disk access.

    Args:
        deck:              The :class:`~fortune_teller.application.models.domain.Deck`
                           to deal from.
        spread:            The :class:`~fortune_teller.application.models.domain.Spread`
                           to use for this reading.
        per_card_chain:    Chain that returns an interpretation string given a
                           dict of context keys.  Pass ``None`` to defer wiring
                           (raises :exc:`RuntimeError` if called).
        summary_chain:     Chain that returns a summary string.  Pass ``None``
                           to skip summary generation (summary will be empty).
        vector_store:      Optional :class:`~fortune_teller.application.stores.vector.VectorStore`
                           used by :func:`build_per_card_context` to retrieve
                           card-section chunks for RAG.  When ``None`` the
                           service uses a minimal 4-key context (no retrieval).
        embedder:          Optional :class:`~fortune_teller.application.stores.embeddings.Embedder`
                           used to embed the card query for retrieval.  Must
                           be provided together with *vector_store*.
        history_store:     Optional :class:`HistoryStore` for persisting
                           completed readings.  When ``None`` (the default)
                           finalize is a pure function with no I/O side effects.
    """

    @property
    def deck_id(self) -> str:
        """The deck identifier (e.g. ``"book-of-thoth"``)."""
        return self._deck.id

    def __init__(
        self,
        deck: Deck,
        spread: Spread,
        per_card_chain: InterpretationChain | None = None,
        summary_chain: SummaryChain | None = None,
        vector_store: object | None = None,
        embedder: object | None = None,
        history_store: HistoryStore | None = None,
    ) -> None:
        self._deck = deck
        self._spread = spread
        self._per_card_chain = per_card_chain
        self._summary_chain = summary_chain
        # vector_store/embedder are typed as ``object`` (not the concrete
        # classes) to avoid a runtime import of the stores package in
        # tests that don't need RAG. mypy strict is fine because the
        # consumer (``build_per_card_context``) accepts Protocol-like
        # duck types.
        self._vector_store = vector_store
        self._embedder = embedder
        self._history_store = history_store
        if (vector_store is None) != (embedder is None):
            raise ValueError(
                "vector_store and embedder must be provided together (both or neither)."
            )

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

        # RAG context when both vector_store and embedder are wired;
        # otherwise a minimal 4-key context (used in unit tests).
        if self._vector_store is not None and self._embedder is not None:
            inputs = build_per_card_context(
                dealt=dealt,
                card=card,
                spread=handle.spread,
                position=position,
                vector_store=self._vector_store,  # type: ignore[arg-type]
                embedder=self._embedder,  # type: ignore[arg-type]
                deck_id=self._deck.id,
            )
        else:
            inputs = {
                "card_name": card.name,
                "orientation": dealt.orientation.value,
                "position_name": position.name,
                "position_meaning": position.meaning,
            }
        text = self._per_card_chain.invoke(inputs)

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

        Calls the ``summary_chain`` (with the structured RAG context
        produced by :func:`build_summary_context`) if configured.
        When ``summary_chain`` is ``None`` the summary is left as the
        empty string.

        Args:
            handle: A :class:`ReadingHandle` where all positions have been
                    dealt (but this is not enforced — partial readings are
                    allowed for future flexibility).

        Returns:
            A complete, immutable :class:`Reading`.
        """
        summary = ""
        if self._summary_chain is not None:
            synergies = compute_synergies(handle.dealt, self._deck)
            context = build_summary_context(
                handle.interpretations, handle.spread, synergies=synergies
            )
            summary = self._summary_chain.invoke(context)

        reading = Reading(
            deck_id=handle.deck_id,
            spread_id=handle.spread.id,
            dealt=list(handle.dealt),
            per_card=list(handle.interpretations),
            summary=summary,
        )

        if self._history_store is not None:
            self._history_store.save(reading)

        return reading


# ---------------------------------------------------------------------------
# Wiring factory
# ---------------------------------------------------------------------------


def build_reading_service(
    settings: object,
    *,
    deck_id: str = "book-of-thoth",
    spread_id: str | None = None,
    history_store: HistoryStore | None = None,
) -> ReadingService:
    """Construct a fully-wired :class:`ReadingService` from app settings.

    Wires together:

    - :class:`~fortune_teller.application.stores.embeddings.Embedder` (lazy)
    - :class:`~fortune_teller.application.stores.vector.VectorStore` (opened)
    - :class:`~fortune_teller.application.chains.per_card.build_chat_model`
    - :func:`build_per_card_chain` and :func:`build_summary_chain`
    - The :class:`Deck` and :class:`Spread` loaded from
      ``settings.ft_data_dir / "parsed"``.

    The returned service is ready to be passed to
    :func:`~fortune_teller.application.ui.nicegui_app.build_app`. The vector
    store is left open for the lifetime of the process.

    Args:
        settings:  A :class:`~fortune_teller.application.config.Settings`
                   instance (typed as ``object`` to avoid a circular
                   import for the UI module).
        deck_id:   Deck slug to load (default ``"book-of-thoth"``).
        spread_id: Spread slug to load, or ``None`` to pick the first
                   spread found under ``data/parsed/spreads/``.

    Raises:
        FileNotFoundError: If the parsed data directory is missing.
    """
    # Lazy imports so test patches on the source modules (``stores.embeddings``
    # and ``stores.vector``) are picked up at call time.
    from fortune_teller.application.chains.per_card import (  # noqa: PLC0415
        build_chat_model,
        build_per_card_chain,
    )
    from fortune_teller.application.chains.summary import (  # noqa: PLC0415
        build_summary_chain,
    )
    from fortune_teller.application.config import summary_timeout  # noqa: PLC0415
    from fortune_teller.application.services.loading import (  # noqa: PLC0415
        load_deck,
        load_first_spread,
        load_spread,
    )
    from fortune_teller.application.stores.embeddings import Embedder  # noqa: PLC0415
    from fortune_teller.application.stores.vector import VectorStore  # noqa: PLC0415

    parsed_dir = settings.ft_data_dir / "parsed"  # type: ignore[attr-defined]
    deck = load_deck(parsed_dir, deck_id)
    spread = load_spread(parsed_dir, spread_id) if spread_id else load_first_spread(parsed_dir)

    embedder = Embedder()
    db_path = settings.ft_data_dir / "duckdb" / "fortune.duckdb"  # type: ignore[attr-defined]
    vector_store = VectorStore(str(db_path), dimension=embedder.dimension)
    vector_store.open()

    # Per-card prompts are small and finish quickly; the summary prompt grows
    # linearly with position count, so its timeout must scale accordingly.
    per_card_llm = build_chat_model()
    summary_llm = build_chat_model(timeout=summary_timeout(len(spread.positions)))

    # ``build_per_card_chain`` / ``build_summary_chain`` return a generic
    # LangChain ``Runnable`` whose ``invoke`` signature is wider than the
    # ``InterpretationChain``/``SummaryChain`` Protocols expect. The
    # Protocols are satisfied at runtime; we cast here to keep mypy happy.
    per_card_chain = cast(InterpretationChain, build_per_card_chain(per_card_llm))
    summary_chain = cast(SummaryChain, build_summary_chain(summary_llm))

    return ReadingService(
        deck=deck,
        spread=spread,
        per_card_chain=per_card_chain,
        summary_chain=summary_chain,
        vector_store=vector_store,
        embedder=embedder,
        history_store=history_store,
    )

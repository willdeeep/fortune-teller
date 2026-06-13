"""Unit tests for :class:`HistoryStore` injection into :class:`ReadingService`.

Verifies that ``finalize()`` calls ``history_store.save()`` when a store is
provided, and is a no-op when ``None``.
"""

from __future__ import annotations

import pytest
from pydantic import HttpUrl

from fortune_teller.application.models.domain import (
    Card,
    Deck,
    Reading,
    Spread,
    SpreadPosition,
)
from fortune_teller.application.services.reading import ReadingService

_SPREAD_URL = "https://example.test/spread"
_CARD_URL = "https://example.test/card"


def _make_deck(count: int = 3) -> Deck:
    cards = [
        Card(
            id=f"card-{i}",
            name=f"Card {i}",
            arcana="major",  # type: ignore[arg-type]
            source_url=HttpUrl(_CARD_URL),
        )
        for i in range(count)
    ]
    return Deck(id="test-deck", name="Test Deck", cards=cards)


def _make_spread(position_count: int = 3) -> Spread:
    return Spread(
        id="test-spread",
        name="Test Spread",
        positions=[
            SpreadPosition(
                index=i,
                name=f"Position {i}",
                meaning=f"Meaning of position {i}.",
                source_url=HttpUrl(_SPREAD_URL),
            )
            for i in range(position_count)
        ],
    )


class _StubPerCardChain:
    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    def invoke(self, inputs: dict[str, str]) -> str:
        self.calls.append(dict(inputs))
        return f"Interpretation for {inputs.get('card_name', '?')}"


class _StubSummaryChain:
    def invoke(self, inputs: dict[str, str]) -> str:  # noqa: ARG002
        return "A summary of the reading."


class _StubHistoryStore:
    def __init__(self) -> None:
        self.saved: list[Reading] = []

    def save(self, reading: Reading) -> None:
        self.saved.append(reading)


@pytest.mark.unit
class TestFinalizeWithHistoryStore:
    def test_finalize_calls_save(self) -> None:
        deck = _make_deck(5)
        spread = _make_spread(3)
        history = _StubHistoryStore()
        service = ReadingService(
            deck=deck,
            spread=spread,
            per_card_chain=_StubPerCardChain(),
            summary_chain=_StubSummaryChain(),
            history_store=history,
        )
        handle = service.start(seed=42)
        service.deal_next(handle)
        service.deal_next(handle)
        service.deal_next(handle)
        reading = service.finalize(handle)

        assert len(history.saved) == 1
        assert history.saved[0].id == reading.id

    def test_finalize_calls_save_with_correct_reading(self) -> None:
        deck = _make_deck(5)
        spread = _make_spread(2)
        history = _StubHistoryStore()
        service = ReadingService(
            deck=deck,
            spread=spread,
            per_card_chain=_StubPerCardChain(),
            summary_chain=_StubSummaryChain(),
            history_store=history,
        )
        handle = service.start(seed=0)
        service.deal_next(handle)
        service.finalize(handle)

        assert history.saved[0].deck_id == "test-deck"
        assert history.saved[0].spread_id == "test-spread"
        assert len(history.saved[0].dealt) == 1
        assert history.saved[0].summary == "A summary of the reading."


@pytest.mark.unit
class TestFinalizeWithoutHistoryStore:
    def test_finalize_without_store_is_noop(self) -> None:
        deck = _make_deck(5)
        spread = _make_spread(3)
        service = ReadingService(
            deck=deck,
            spread=spread,
            per_card_chain=_StubPerCardChain(),
            summary_chain=_StubSummaryChain(),
            history_store=None,
        )
        handle = service.start(seed=42)
        service.deal_next(handle)
        service.deal_next(handle)
        service.deal_next(handle)
        service.finalize(handle)

    def test_finalize_without_store_still_returns_reading(self) -> None:
        deck = _make_deck(5)
        spread = _make_spread(2)
        service = ReadingService(
            deck=deck,
            spread=spread,
            per_card_chain=_StubPerCardChain(),
            summary_chain=_StubSummaryChain(),
            history_store=None,
        )
        handle = service.start(seed=0)
        service.deal_next(handle)
        reading = service.finalize(handle)

        assert isinstance(reading, Reading)
        assert len(reading.dealt) == 1

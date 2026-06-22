"""NiceGUI "main file" for the grid-layout + spread-selector interaction tests.

Like ``tests/nicegui_main.py`` but builds the app with two spreads and a service
factory so the ``User`` simulation can exercise plan 0030's CSS-grid renderer,
the spread ``ui.select``, and spread switching. The default spread is a 2-D grid
(with a 90°-rotated crossing card sharing a cell); the alternate is a plain row
spread, so switching toggles between the grid and row layouts.
"""

from __future__ import annotations

from nicegui import ui
from pydantic import HttpUrl

from fortune_teller.application.models.domain import Spread, SpreadPosition
from fortune_teller.application.services.reading import ReadingService
from fortune_teller.application.ui.nicegui_app import build_app
from tests.unit.test_nicegui_app import (
    _make_deck,
    _make_spread,
    _StubHistoryStore,
    _StubReadingService,
)

_URL = "https://example.test/spread"


def _grid_spread() -> Spread:
    return Spread(
        id="grid-spread",
        name="Grid Spread",
        positions=[
            SpreadPosition(
                index=0,
                name="Center",
                meaning="The centre.",
                source_url=HttpUrl(_URL),
                row=0,
                col=0,
                rotation=0,
            ),
            SpreadPosition(
                index=1,
                name="Crossing",
                meaning="The crossing card.",
                source_url=HttpUrl(_URL),
                row=0,
                col=0,
                rotation=90,
            ),
            SpreadPosition(
                index=2,
                name="Right",
                meaning="To the right.",
                source_url=HttpUrl(_URL),
                row=0,
                col=1,
                rotation=0,
            ),
            SpreadPosition(
                index=3,
                name="Below",
                meaning="Below the centre.",
                source_url=HttpUrl(_URL),
                row=1,
                col=0,
                rotation=0,
            ),
        ],
    )


_deck = _make_deck(12)
_grid_service = _StubReadingService(deck=_deck, spread=_grid_spread())
_row_service = _StubReadingService(deck=_deck, spread=_make_spread(3))
_history = _StubHistoryStore()

_services = {
    ("test-deck", "grid-spread"): _grid_service,
    ("test-deck", "test-spread"): _row_service,
}


def _factory(deck_id: str, spread_id: str) -> ReadingService:
    return _services[(deck_id, spread_id)]  # type: ignore[return-value]


build_app(
    _grid_service,
    history_store=_history,
    spread_options=[("grid-spread", "Grid Spread"), ("test-spread", "Test Spread")],
    deck_options=[("test-deck", "Test Deck")],
    service_factory=_factory,
)

ui.run()

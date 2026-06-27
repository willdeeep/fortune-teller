"""NiceGUI test harness for reading-resilience tests.

Like ``nicegui_main.py`` but the stub service's :meth:`finalize` raises an
exception so the ``try/except`` in ``_run_reading`` is exercised.
"""

from __future__ import annotations

from nicegui import ui

from fortune_teller.application.ui.nicegui_app import build_app
from tests.unit.test_nicegui_app import (
    _make_deck,
    _make_spread,
    _StubHistoryStore,
    _StubReadingService,
)


class _FailingReadingService(_StubReadingService):
    """Stub whose ``finalize`` always raises."""

    def finalize(self, handle: object) -> object:  # noqa: ARG002
        raise RuntimeError("LLM summary timed out")


_service = _FailingReadingService(deck=_make_deck(5), spread=_make_spread(3))
_history = _StubHistoryStore()

build_app(_service, history_store=_history)

ui.run()

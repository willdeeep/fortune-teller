"""NiceGUI "main file" used by the ``nicegui.testing`` ``User`` fixture.

The ``User`` simulation runs this module via ``runpy`` (as ``__main__``) inside a
reset NiceGUI app, so the interaction tests in ``tests/unit/test_nicegui_app.py``
exercise the real :func:`build_app` wiring and page body against deterministic
in-process stubs — no config, SQLite, or LLM involved.

A reading is pre-saved so the History section renders a selectable row.
``build_app`` registers the ``/`` page route each time it runs (see its body),
which is what makes re-running this harness per test work.
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

_service = _StubReadingService(deck=_make_deck(5), spread=_make_spread(3))
_history = _StubHistoryStore()

# Seed one finalized reading so the History table has a row to select.
_handle = _service.start()
for _ in _service._spread.positions:
    _service.deal_next(_handle)
_history.save(_service.finalize(_handle))

build_app(_service, history_store=_history)

ui.run()

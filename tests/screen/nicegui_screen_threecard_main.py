"""NiceGUI ``Screen`` main file: 3-card spread with realistic long text.

Driven by the Selenium ``screen`` fixture (plan 0036). Uses a long
multi-paragraph stub interpretation so the screenshot reveals any text overrun
the short ``User``-test stub would hide.
"""

from __future__ import annotations

from nicegui import ui

from fortune_teller.application.ui.nicegui_app import build_app
from tests.unit.test_nicegui_app import _make_deck, _make_spread, _StubChain, _StubReadingService

_LONG_TEXT = (
    "This card speaks to a threshold moment, where the momentum gathered over "
    "many quiet weeks finally asks to be acted upon. The energy is forward-leaning "
    "but unproven.\n\n"
    "Look to what you have been rehearsing in private. The reversed currents here "
    "suggest hesitation born of past disappointment rather than any real obstacle "
    "in the present.\n\n"
    "Practical guidance: name the single next step, make it small enough to finish "
    "today, and let the larger pattern reveal itself through motion rather than "
    "deliberation."
)
_LONG_SUMMARY = (
    "Taken together the three positions trace a movement from buried potential, "
    "through present hesitation, toward a tentative but genuine opening — the "
    "reading favours deliberate small action over waiting for certainty."
)

_service = _StubReadingService(deck=_make_deck(5), spread=_make_spread(3))
_service._per_card_chain = _StubChain(reply=_LONG_TEXT)
_service._summary_chain = _StubChain(reply=_LONG_SUMMARY)

build_app(_service)

ui.run()

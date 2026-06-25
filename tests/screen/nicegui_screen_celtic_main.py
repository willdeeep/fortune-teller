"""NiceGUI ``Screen`` main file: real Celtic Cross spread with long stub text.

Loads the committed ``data/parsed/spreads/celtic-cross.json`` so the screenshot
exercises the true 10-position layout (including the 90° crossing card sharing a
cell), paired with a 12-card stub deck and long multi-paragraph interpretation.
"""

from __future__ import annotations

from pathlib import Path

from nicegui import ui

from fortune_teller.application.models.domain import Spread
from fortune_teller.application.ui.nicegui_app import build_app
from tests.unit.test_nicegui_app import _make_deck, _StubChain, _StubReadingService

_CELTIC_PATH = Path(__file__).parents[2] / "data" / "parsed" / "spreads" / "celtic-cross.json"
_spread = Spread.model_validate_json(_CELTIC_PATH.read_text())

_LONG_TEXT = (
    "This position carries a dense, layered meaning. The card's upright current "
    "draws on themes of initiative and exposure, while its placement colours that "
    "energy with the question you brought to the table.\n\n"
    "Notice how it converses with the cards around it — the cross and the staff are "
    "not read in isolation but as a single sentence whose grammar is spatial.\n\n"
    "Guidance: hold this interpretation lightly; it is one voice in a chord, and the "
    "summary below will resolve the tensions the individual cards only hint at."
)
_LONG_SUMMARY = (
    "Across all ten positions the reading describes a situation in transition: a "
    "settled past giving way under present pressure, with hopes and fears amplifying "
    "each other, and an outcome that hinges on how honestly the querent reads their "
    "own environment rather than on any fixed fate."
)

_service = _StubReadingService(deck=_make_deck(12), spread=_spread)
_service._per_card_chain = _StubChain(reply=_LONG_TEXT)
_service._summary_chain = _StubChain(reply=_LONG_SUMMARY)

build_app(_service)

ui.run()

"""Opt-in, non-gating screenshot tests (plan 0036).

Run explicitly: ``uv run pytest -m screen --no-cov``. Requires a local Chrome
install (Selenium Manager resolves the matching chromedriver automatically).
PNGs are written to ``tests/screen/_artifacts/`` (gitignored). These tests are
a local verification tool and are excluded from the default suite and CI via the
``-m 'not screen'`` default selector.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from nicegui.testing import Screen

_ARTIFACTS = Path(__file__).parent / "_artifacts"
_THREECARD_MAIN = "tests/screen/nicegui_screen_threecard_main.py"
_CELTIC_MAIN = "tests/screen/nicegui_screen_celtic_main.py"


def _capture(screen: Screen, name: str) -> None:
    """Point the Screen at our gitignored artefacts dir and shoot."""
    screen.SCREENSHOT_DIR = _ARTIFACTS
    screen.shot(name, failed=False)


@pytest.mark.screen
@pytest.mark.nicegui_main_file(_THREECARD_MAIN)
class TestThreeCardScreenshots:
    def test_backs_then_drawn(self, screen: Screen) -> None:
        screen.open("/")
        screen.should_contain("Position 0")
        # Spread selected → three card backs visible before any deal.
        assert len(screen.find_all_by_class("ft-card-box")) == 3
        _capture(screen, "three-card-backs")

        screen.click("New Reading")
        screen.should_contain("Summary")
        screen.should_contain("UPRIGHT")  # comes from a populated list item
        _capture(screen, "three-card-drawn")


@pytest.mark.screen
@pytest.mark.nicegui_main_file(_CELTIC_MAIN)
class TestCelticCrossScreenshots:
    def test_backs_then_drawn(self, screen: Screen) -> None:
        screen.open("/")
        screen.should_contain("Present")
        # All ten positions render as backs, including the crossing card.
        assert len(screen.find_all_by_class("ft-card-box")) == 10
        _capture(screen, "celtic-cross-backs")

        screen.click("New Reading")
        screen.should_contain("Summary")
        # Ten numbered list items below the grid, each carrying long text.
        assert len(screen.find_all_by_class("ft-list-item")) == 10
        _capture(screen, "celtic-cross-drawn")

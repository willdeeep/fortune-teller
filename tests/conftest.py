"""Shared pytest fixtures for Fortune Teller tests.

Fixtures are organised by scope:
- session-scoped: expensive setup done once per test session
- function-scoped (default): fresh state per test

See plan 0010 for full fixture specification.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "fixtures"
HTML_DIR = FIXTURES_DIR / "html" / "thothreadings"
PARSED_DIR = FIXTURES_DIR / "parsed"


# ---------------------------------------------------------------------------
# HTML fixtures (loaded from committed fixture files)
# ---------------------------------------------------------------------------


@pytest.fixture
def the_fool_html() -> str:
    """Raw HTML for The Fool from thothreadings.com (committed fixture)."""
    path = HTML_DIR / "the-fool.html"
    if not path.exists():
        pytest.skip("Fixture not yet committed: tests/fixtures/html/thothreadings/the-fool.html")
    return path.read_text(encoding="utf-8")


@pytest.fixture
def new_moon_spread_html() -> str:
    """Raw HTML for the New Moon spread page (committed fixture)."""
    path = HTML_DIR / "spread-new-moon.html"
    if not path.exists():
        pytest.skip(
            "Fixture not yet committed: tests/fixtures/html/thothreadings/spread-new-moon.html"
        )
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Stub LLM (no network calls in tests)
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_llm() -> RunnableLambda:  # type: ignore[type-arg]
    """A RunnableLambda that returns a canned AIMessage without network."""
    return RunnableLambda(lambda _: AIMessage(content="Stub interpretation for testing."))

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
from pydantic import HttpUrl

from fortune_teller.application.models.domain import (
    Arcana,
    Card,
    CardInterpretation,
    CardSection,
    CardSectionText,
    DealtCard,
    Orientation,
    Spread,
    SpreadPosition,
)
from fortune_teller.application.stores.embeddings import Embedder

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


# ---------------------------------------------------------------------------
# Stub Embedder (no network calls in tests)
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_embedder():
    """An :class:`Embedder` instance with a deterministic stub backend.

    The stub returns one ``[0.0, 0.0, ...]`` vector per text, so callers
    can exercise chunking/insert/search code paths without loading the
    real ``BAAI/bge-small-en-v1.5`` model.
    """
    embedder = Embedder(model_name="stub-model")

    class _StubBackend:
        def __init__(self, dim: int) -> None:
            self.dim = dim

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [[0.0] * self.dim for _ in texts]

        def embed_query(self, text: str) -> list[float]:  # noqa: ARG002
            return [0.0] * self.dim

    embedder.set_backend(_StubBackend(embedder.dimension))
    return embedder


@pytest.fixture
def stub_embedder_factory():
    """Factory for a stub embedder with a customisable dimensionality and backend.

    Returns a callable that takes a ``dim`` and returns a configured
    :class:`Embedder` whose backend returns zero vectors of that
    dimensionality. Useful for tests that need to exercise a specific
    DuckDB array size.
    """

    def _make(dim: int) -> Embedder:
        embedder = Embedder(model_name="stub-model", dimension=dim)

        class _ZeroBackend:
            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return [[0.0] * dim for _ in texts]

            def embed_query(self, text: str) -> list[float]:  # noqa: ARG002
                return [0.0] * dim

        embedder.set_backend(_ZeroBackend())
        return embedder

    return _make


# ---------------------------------------------------------------------------
# Sample domain models (used by chain tests, services tests, etc.)
# ---------------------------------------------------------------------------

_CARD_URL = "https://thothreadings.com/the-fool/"
_SPREAD_URL = "https://thothreadings.com/spread-new-moon/"


@pytest.fixture
def sample_card() -> Card:
    """A :class:`Card` (The Fool) with one section per :class:`CardSection` member."""
    return Card(
        id="the-fool",
        name="The Fool",
        arcana=Arcana.MAJOR,
        sections=[
            CardSectionText(section=CardSection.DRIVE, text="Pure potential, unformed."),
            CardSectionText(
                section=CardSection.LIGHT, text="Spontaneity, curiosity, leap of faith."
            ),
            CardSectionText(
                section=CardSection.SHADOW, text="Recklessness, naivety, fear of commitment."
            ),
            CardSectionText(
                section=CardSection.REVERSED, text="Holding back, fear of the unknown."
            ),
            CardSectionText(section=CardSection.KEYWORDS, text="beginnings, freedom, innocence"),
            CardSectionText(section=CardSection.ADVICE, text="Trust the journey."),
        ],
        source_url=HttpUrl(_CARD_URL),
    )


@pytest.fixture
def sample_spread() -> Spread:
    """A 3-position :class:`Spread` (New Moon Three-Card)."""
    return Spread(
        id="new-moon-three-card",
        name="New Moon Three-Card Spread",
        positions=[
            SpreadPosition(
                index=0,
                name="Past",
                meaning="What was set in motion before now.",
                source_url=HttpUrl(_SPREAD_URL),
            ),
            SpreadPosition(
                index=1,
                name="Present",
                meaning="The current energy at the new moon.",
                source_url=HttpUrl(_SPREAD_URL),
            ),
            SpreadPosition(
                index=2,
                name="Future",
                meaning="What the new moon is birthing.",
                source_url=HttpUrl(_SPREAD_URL),
            ),
        ],
    )


@pytest.fixture
def sample_position(sample_spread: Spread) -> SpreadPosition:
    """First position of :func:`sample_spread` (index 0, 'Past')."""
    return sample_spread.position_by_index(0)


@pytest.fixture
def sample_dealt_card() -> DealtCard:
    """The Fool dealt upright into position 0."""
    return DealtCard(card_id="the-fool", orientation=Orientation.UPRIGHT, position_index=0)


@pytest.fixture
def sample_dealt_reversed() -> DealtCard:
    """The Fool dealt reversed into position 0."""
    return DealtCard(card_id="the-fool", orientation=Orientation.REVERSED, position_index=0)


@pytest.fixture
def sample_interpretation() -> CardInterpretation:
    """A single :class:`CardInterpretation` for The Fool in 'Past' (upright)."""
    return CardInterpretation(
        dealt=DealtCard(card_id="the-fool", orientation=Orientation.UPRIGHT, position_index=0),
        card_name="The Fool",
        position_name="Past",
        text="A new beginning seeded before you were aware of it.",
    )

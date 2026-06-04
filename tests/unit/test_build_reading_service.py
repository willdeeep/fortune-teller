"""Unit tests for the ``build_reading_service`` wiring factory.

The factory glues together disk-backed loaders, the DuckDB vector
store, the embedder, and the LangChain chains. We exercise it with
stubbed-out ``Embedder`` and ``VectorStore`` (their internals are
already covered by their own test files) and a stubbed-out
``build_chat_model`` so no network or heavy model loading happens.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from pydantic import HttpUrl

from fortune_teller.application.models.domain import Card, Spread
from fortune_teller.application.services.reading import build_reading_service

# ---------------------------------------------------------------------------
# Helpers — write a populated parsed/ tree under tmp_path
# ---------------------------------------------------------------------------


def _write_card_json(path: Path, *, card_id: str) -> None:
    card = Card(
        id=card_id,
        name=card_id.replace("-", " ").title(),
        arcana="major",  # type: ignore[arg-type]
        source_url=HttpUrl("https://example.test/card"),
    )
    path.write_text(card.model_dump_json(indent=2), encoding="utf-8")


def _write_spread_json(path: Path, *, spread_id: str, position_count: int) -> None:
    spread = Spread(
        id=spread_id,
        name=f"Test {spread_id}",
        positions=[
            {
                "index": i,
                "name": f"Pos {i}",
                "meaning": f"Meaning {i}.",
                "source_url": HttpUrl("https://example.test/spread"),
            }
            for i in range(position_count)
        ],
    )
    path.write_text(spread.model_dump_json(indent=2), encoding="utf-8")


def _populated_parsed_dir(tmp_path: Path, *, positions: int = 3) -> Path:
    parsed = tmp_path / "parsed"
    deck_dir = parsed / "book-of-thoth"
    deck_dir.mkdir(parents=True)
    for cid in ("0-the-fool", "i-the-magician"):
        _write_card_json(deck_dir / f"{cid}.json", card_id=cid)
    spreads_dir = parsed / "spreads"
    spreads_dir.mkdir()
    _write_spread_json(
        spreads_dir / "new-moon-three-card.json",
        spread_id="new-moon-three-card",
        position_count=positions,
    )
    return parsed


# ---------------------------------------------------------------------------
# Stubs for Embedder / VectorStore / build_chat_model
# ---------------------------------------------------------------------------


class _StubEmbedder:
    """Embedder stub — never embeds anything; records the dimension."""

    instances: ClassVar[list[_StubEmbedder] | None] = None  # populated per test

    def __init__(self, *args, **kwargs) -> None:  # noqa: ARG002
        self.dimension = 4
        _StubEmbedder.instances = (_StubEmbedder.instances or []) + [self]

    def embed_query(self, text: str) -> list[float]:  # noqa: ARG002
        return [0.0] * self.dimension


class _StubVectorStore:
    """VectorStore stub — tracks calls to ``open()``."""

    instances: ClassVar[list[_StubVectorStore] | None] = None

    def __init__(self, path, dimension: int = 384) -> None:
        self.path = str(path)
        self.dimension = dimension
        self.opened = False
        _StubVectorStore.instances = (_StubVectorStore.instances or []) + [self]

    def open(self) -> None:
        self.opened = True


def _make_stub_llm() -> RunnableLambda:
    """Return a :class:`RunnableLambda` that pretends to be a chat model.

    LangChain's chain factory checks ``isinstance(llm, Runnable)``, and
    :class:`RunnableLambda` is a Runnable subclass, so this satisfies
    the ``prompt | llm | parser`` operator. Each ``invoke`` returns an
    ``AIMessage`` so the subsequent ``StrOutputParser`` in the chain
    works as expected.
    """

    def _respond(_messages: object) -> AIMessage:
        return AIMessage(content="Stub LLM response.")

    return RunnableLambda(_respond)


# ---------------------------------------------------------------------------
# Fixture: patch the heavy dependencies and the chat-model builder
# ---------------------------------------------------------------------------


@pytest.fixture
def patched_dependencies(monkeypatch: pytest.MonkeyPatch):
    """Replace Embedder, VectorStore, and ``build_chat_model`` with stubs."""
    _StubEmbedder.instances = []
    _StubVectorStore.instances = []

    monkeypatch.setattr(
        "fortune_teller.application.stores.embeddings.Embedder",
        _StubEmbedder,
    )
    monkeypatch.setattr(
        "fortune_teller.application.stores.vector.VectorStore",
        _StubVectorStore,
    )
    monkeypatch.setattr(
        "fortune_teller.application.chains.per_card.build_chat_model",
        _make_stub_llm,
    )
    return SimpleNamespace(Embedder=_StubEmbedder, VectorStore=_StubVectorStore)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildReadingService:
    def test_returns_reading_service(self, tmp_path: Path, patched_dependencies) -> None:  # noqa: ARG002
        _populated_parsed_dir(tmp_path)
        settings = SimpleNamespace(
            ft_data_dir=tmp_path,
            openai_base_url="http://example.test/v1",
            openai_api_key="sk-test",
            chat_model="stub-model",
        )
        service = build_reading_service(settings)
        assert service is not None
        # Both injected deps must be present
        assert service._vector_store is not None
        assert service._embedder is not None
        assert service._per_card_chain is not None
        assert service._summary_chain is not None

    def test_uses_settings_data_dir_for_parsed(
        self,
        tmp_path: Path,
        patched_dependencies,  # noqa: ARG002
    ) -> None:
        parsed = _populated_parsed_dir(tmp_path)
        settings = SimpleNamespace(
            ft_data_dir=tmp_path,
            openai_base_url="http://example.test/v1",
            openai_api_key="sk-test",
            chat_model="stub-model",
        )
        service = build_reading_service(settings)
        # Deck and spread were loaded from the populated dir.
        assert service._deck.id == "book-of-thoth"
        assert {c.id for c in service._deck.cards} == {
            "0-the-fool",
            "i-the-magician",
        }
        assert service._spread.id == "new-moon-three-card"
        # And parsed_dir attribute points to the populated location.
        assert parsed.exists()

    def test_opens_vector_store(self, tmp_path: Path, patched_dependencies) -> None:  # noqa: ARG002
        _populated_parsed_dir(tmp_path)
        settings = SimpleNamespace(
            ft_data_dir=tmp_path,
            openai_base_url="http://example.test/v1",
            openai_api_key="sk-test",
            chat_model="stub-model",
        )
        build_reading_service(settings)
        assert _StubVectorStore.instances is not None
        store = _StubVectorStore.instances[0]
        assert store.opened is True
        assert store.path.endswith("duckdb/fortune.duckdb")

    def test_embedder_dimension_passed_to_vector_store(
        self,
        tmp_path: Path,
        patched_dependencies,  # noqa: ARG002
    ) -> None:
        _populated_parsed_dir(tmp_path)
        settings = SimpleNamespace(
            ft_data_dir=tmp_path,
            openai_base_url="http://example.test/v1",
            openai_api_key="sk-test",
            chat_model="stub-model",
        )
        build_reading_service(settings)
        embedder = _StubEmbedder.instances[0]
        store = _StubVectorStore.instances[0]
        assert store.dimension == embedder.dimension == 4

    def test_specific_deck_and_spread(
        self,
        tmp_path: Path,
        patched_dependencies,  # noqa: ARG002
    ) -> None:
        # Add a second deck + a spread whose name sorts after
        # ``new-moon-three-card`` so the default-pick tests stay valid.
        parsed = _populated_parsed_dir(tmp_path)
        other_deck_dir = parsed / "other-deck"
        other_deck_dir.mkdir()
        _write_card_json(other_deck_dir / "queen-of-swords.json", card_id="queen-of-swords")
        spreads_dir = parsed / "spreads"
        _write_spread_json(
            spreads_dir / "zodiac-cross.json",
            spread_id="zodiac-cross",
            position_count=10,
        )

        settings = SimpleNamespace(
            ft_data_dir=tmp_path,
            openai_base_url="http://example.test/v1",
            openai_api_key="sk-test",
            chat_model="stub-model",
        )
        # Default invocation picks first deck (alphabetical) / first spread.
        service_default = build_reading_service(settings)
        assert service_default._deck.id == "book-of-thoth"
        assert service_default._spread.id == "new-moon-three-card"
        # And we can also pass explicit IDs.
        service_explicit = build_reading_service(
            settings, deck_id="other-deck", spread_id="zodiac-cross"
        )
        assert service_explicit._deck.id == "other-deck"
        assert {c.id for c in service_explicit._deck.cards} == {"queen-of-swords"}
        assert service_explicit._spread.id == "zodiac-cross"
        assert len(service_explicit._spread.positions) == 10

    def test_missing_parsed_dir_raises(
        self,
        tmp_path: Path,
        patched_dependencies,  # noqa: ARG002
    ) -> None:
        # tmp_path/parsed is NOT created — the factory should raise.
        settings = SimpleNamespace(
            ft_data_dir=tmp_path,
            openai_base_url="http://example.test/v1",
            openai_api_key="sk-test",
            chat_model="stub-model",
        )
        with pytest.raises(FileNotFoundError):
            build_reading_service(settings)

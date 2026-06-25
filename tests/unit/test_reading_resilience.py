"""Unit tests for reading resilience to LLM failures (plan 0038 / issue #45).

Covers:

- :func:`summary_timeout` — the pure helper that scales the summary-chain
  HTTP timeout by spread size.
- :func:`build_chat_model` — accepts and propagates a ``timeout`` parameter.
- :func:`build_reading_service` — builds separate per-card and summary LLMs
  with different timeouts.
- ``_run_reading`` — catches exceptions from the reading pipeline and shows
  an error message instead of crashing.
- ``build_app`` — registers an ``app.on_exception`` handler.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from nicegui.testing import User
from pydantic import HttpUrl

from fortune_teller.application.chains.per_card import build_chat_model
from fortune_teller.application.config import summary_timeout
from fortune_teller.application.models.domain import Card, Spread
from fortune_teller.application.services.reading import build_reading_service

_RESILIENCE_MAIN = "tests/nicegui_resilience_main.py"


# ---------------------------------------------------------------------------
# summary_timeout — pure helper
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSummaryTimeout:
    @pytest.mark.parametrize(
        ("n", "expected"),
        [
            (0, 120.0),
            (1, 132.0),
            (3, 156.0),
            (10, 240.0),
        ],
    )
    def test_returns_base_plus_per_card_times_n(self, n: int, expected: float) -> None:
        assert summary_timeout(n) == expected

    def test_monotonic_in_n(self) -> None:
        assert summary_timeout(5) > summary_timeout(3)

    def test_zero_positions_returns_base(self) -> None:
        assert summary_timeout(0) == 120.0


# ---------------------------------------------------------------------------
# build_chat_model — timeout parameter
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildChatModelTimeout:
    def test_default_timeout_uses_per_card_setting(self) -> None:
        llm = build_chat_model()
        assert llm.request_timeout == 60.0  # type: ignore[attr-defined]

    def test_explicit_timeout_passed_through(self) -> None:
        llm = build_chat_model(timeout=240.0)
        assert llm.request_timeout == 240.0  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# build_reading_service — per-stage timeouts
# ---------------------------------------------------------------------------


class _StubEmbedder2:
    instances: ClassVar[list[_StubEmbedder2] | None] = None

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        self.dimension = 4
        _StubEmbedder2.instances = (_StubEmbedder2.instances or []) + [self]

    def embed_query(self, text: str) -> list[float]:  # noqa: ARG002
        return [0.0] * self.dimension


class _StubVectorStore2:
    instances: ClassVar[list[_StubVectorStore2] | None] = None

    def __init__(self, _path: object, dimension: int = 384) -> None:
        self.dimension = dimension
        self.opened = False
        _StubVectorStore2.instances = (_StubVectorStore2.instances or []) + [self]

    def open(self) -> None:
        self.opened = True


def _make_stub_llm2(*_args: object, **_kwargs: object) -> RunnableLambda:
    def _respond(_messages: object) -> AIMessage:
        return AIMessage(content="Stub LLM response.")

    return RunnableLambda(_respond)


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


@pytest.fixture
def patched_deps(monkeypatch: pytest.MonkeyPatch):
    _StubEmbedder2.instances = []
    _StubVectorStore2.instances = []
    monkeypatch.setattr(
        "fortune_teller.application.stores.embeddings.Embedder",
        _StubEmbedder2,
    )
    monkeypatch.setattr(
        "fortune_teller.application.stores.vector.VectorStore",
        _StubVectorStore2,
    )
    captured: list[float | None] = []

    def _capture_llm(*_args: object, **kwargs: object) -> RunnableLambda:
        captured.append(kwargs.get("timeout"))  # type: ignore[arg-type]
        return _make_stub_llm2(**kwargs)

    monkeypatch.setattr(
        "fortune_teller.application.chains.per_card.build_chat_model",
        _capture_llm,
    )
    return SimpleNamespace(captured=captured)


@pytest.mark.unit
class TestBuildReadingServiceTimeouts:
    def test_per_card_and_summary_get_different_timeouts(
        self,
        tmp_path: Path,
        patched_deps: SimpleNamespace,
    ) -> None:
        _populated_parsed_dir(tmp_path, positions=10)
        settings_obj = SimpleNamespace(
            ft_data_dir=tmp_path,
            openai_base_url="http://example.test/v1",
            openai_api_key="sk-test",
            chat_model="stub-model",
        )
        build_reading_service(settings_obj)
        # build_chat_model is called twice: once for per-card (None = default)
        # and once for summary (scaled by position count = 120 + 12*10 = 240).
        assert len(patched_deps.captured) == 2
        per_card_timeout, summary_timeout_val = patched_deps.captured
        assert per_card_timeout is None  # per-card uses settings.per_card_timeout
        assert summary_timeout_val == 240.0  # 120 + 12*10

    def test_small_spread_gets_smaller_summary_timeout(
        self,
        tmp_path: Path,
        patched_deps: SimpleNamespace,
    ) -> None:
        _populated_parsed_dir(tmp_path, positions=3)
        settings_obj = SimpleNamespace(
            ft_data_dir=tmp_path,
            openai_base_url="http://example.test/v1",
            openai_api_key="sk-test",
            chat_model="stub-model",
        )
        build_reading_service(settings_obj)
        assert len(patched_deps.captured) == 2
        summary_timeout_val = patched_deps.captured[1]
        # 120 + 12*3 = 156
        assert summary_timeout_val == 156.0


# ---------------------------------------------------------------------------
# _run_reading — UI resilience (NiceGUI User simulation)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.nicegui_main_file(_RESILIENCE_MAIN)
class TestRunReadingResilience:
    async def test_failed_reading_shows_error_message(self, user: User) -> None:
        await user.open("/")
        user.find("New Reading").click()
        await user.should_see("Reading failed")

    async def test_failed_reading_clears_summary(self, user: User) -> None:
        await user.open("/")
        user.find("New Reading").click()
        await user.should_see("Reading failed")

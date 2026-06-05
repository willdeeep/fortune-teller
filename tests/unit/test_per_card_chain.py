"""Unit tests for :mod:`fortune_teller.application.chains.per_card`."""

from __future__ import annotations

import pytest
from langchain_core.runnables import RunnableLambda, RunnableSequence
from pydantic import HttpUrl

from fortune_teller.application.chains.per_card import (
    build_chat_model,
    build_per_card_chain,
    build_per_card_context,
    per_card_prompt,
)
from fortune_teller.application.config import Settings
from fortune_teller.application.models.domain import (
    Card,
    CardSection,
    CardSectionText,
    Chunk,
    ChunkType,
    DealtCard,
    Spread,
    SpreadPosition,
)
from fortune_teller.application.stores.vector import VectorStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_card(name: str = "The Fool", card_id: str = "the-fool") -> Card:
    return Card(
        id=card_id,
        name=name,
        sections=[
            CardSectionText(section=CardSection.DRIVE, text="Pure potential."),
            CardSectionText(section=CardSection.LIGHT, text="Spontaneity."),
            CardSectionText(section=CardSection.REVERSED, text="Holding back."),
        ],
        source_url=HttpUrl("https://example.com/the-fool/"),
    )


def _make_spread(spread_id: str = "new-moon-three-card") -> Spread:
    return Spread(
        id=spread_id,
        name="New Moon Three-Card Spread",
        positions=[
            SpreadPosition(
                index=0,
                name="Past",
                meaning="What has been.",
                source_url=HttpUrl("https://example.com/spread-new-moon/"),
            ),
            SpreadPosition(
                index=1,
                name="Present",
                meaning="What is.",
                source_url=HttpUrl("https://example.com/spread-new-moon/"),
            ),
        ],
    )


def _make_chunk(
    text: str,
    *,
    card_id: str | None = None,
    section: CardSection | None = None,
    spread_id: str | None = None,
    position_index: int | None = None,
) -> Chunk:
    return Chunk(
        chunk_type=(ChunkType.CARD_SECTION if card_id is not None else ChunkType.SPREAD_POSITION),
        deck_id="book-of-thoth" if card_id is not None else None,
        card_id=card_id,
        card_name="The Fool" if card_id is not None else None,
        section=section,
        spread_id=spread_id,
        position_index=position_index,
        source_url="https://example.com/x/",
        text=text,
        embedding=[0.0] * 4,
    )


# ---------------------------------------------------------------------------
# per_card_prompt rendering
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPerCardPromptRenders:
    def test_renders_two_messages(self) -> None:
        ctx = {
            "card_name": "The Fool",
            "orientation": "upright",
            "position_name": "Past",
            "position_meaning": "What has been.",
            "retrieved_card_sections": "- Drive: forward motion.",
            "retrieved_position_text": "- Past: what was.",
        }
        msgs = per_card_prompt.format_messages(**ctx)
        assert len(msgs) == 2
        assert msgs[0].type == "system"
        assert msgs[1].type == "human"

    def test_renders_card_name_and_orientation(self) -> None:
        msgs = per_card_prompt.format_messages(
            card_name="The Fool",
            orientation="upright",
            position_name="Past",
            position_meaning="What has been.",
            retrieved_card_sections="- Drive: forward motion.",
            retrieved_position_text="- Past: what was.",
        )
        assert "The Fool" in msgs[1].content
        assert "upright" in msgs[1].content

    def test_renders_reversed_orientation_lowercase(self) -> None:
        msgs = per_card_prompt.format_messages(
            card_name="The Fool",
            orientation="reversed",
            position_name="Past",
            position_meaning="What has been.",
            retrieved_card_sections="- Reversed: holding back.",
            retrieved_position_text="- Past: what was.",
        )
        assert "reversed" in msgs[1].content.lower()

    def test_renders_position_meaning(self) -> None:
        msgs = per_card_prompt.format_messages(
            card_name="The Fool",
            orientation="upright",
            position_name="Past",
            position_meaning="What was set in motion before now.",
            retrieved_card_sections="- Drive: forward motion.",
            retrieved_position_text="- Past: what was.",
        )
        assert "What was set in motion before now." in msgs[1].content

    def test_renders_retrieved_sections(self) -> None:
        msgs = per_card_prompt.format_messages(
            card_name="The Fool",
            orientation="upright",
            position_name="Past",
            position_meaning="What has been.",
            retrieved_card_sections="- Drive: forward motion.\n- Light: bright side.",
            retrieved_position_text="- Past: what was.",
        )
        assert "Drive: forward motion." in msgs[1].content
        assert "Light: bright side." in msgs[1].content


# ---------------------------------------------------------------------------
# build_chat_model
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildChatModel:
    def test_uses_default_settings(self) -> None:
        llm = build_chat_model()
        assert llm.model_name == "local-model"
        assert "127.0.0.1:8080" in str(llm.openai_api_base)
        assert llm.temperature == 0.0  # type: ignore[attr-defined]

    def test_uses_overridden_settings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from fortune_teller.application import (  # noqa: PLC0415
            config,
        )  # test needs late import to patch settings

        monkeypatch.setattr(
            config,
            "settings",
            Settings(
                openai_base_url="http://example.com:9999/v1",
                openai_api_key="test-key-123",
                chat_model="gpt-test",
            ),
        )
        # Reload the chain module to pick up the patched settings.
        import importlib  # noqa: PLC0415

        from fortune_teller.application.chains import (  # noqa: PLC0415
            per_card,
        )

        importlib.reload(per_card)
        try:
            llm = per_card.build_chat_model()
            assert llm.model_name == "gpt-test"
            assert "example.com:9999" in str(llm.openai_api_base)
            # ChatOpenAI stores the key as a SecretStr.
            assert llm.openai_api_key is not None  # type: ignore[attr-defined]
            assert llm.openai_api_key.get_secret_value() == "test-key-123"  # type: ignore[attr-defined]
        finally:
            # Reload again to restore the real settings for other tests.
            monkeypatch.undo()
            importlib.reload(per_card)


# ---------------------------------------------------------------------------
# build_per_card_chain
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildPerCardChain:
    def test_returns_runnable_sequence(self, stub_llm: RunnableLambda) -> None:  # type: ignore[type-arg]
        chain = build_per_card_chain(stub_llm)
        assert isinstance(chain, RunnableSequence)

    def test_invokes_to_string(self, stub_llm: RunnableLambda) -> None:  # type: ignore[type-arg]
        chain = build_per_card_chain(stub_llm)
        ctx = {
            "card_name": "The Fool",
            "orientation": "upright",
            "position_name": "Past",
            "position_meaning": "What has been.",
            "retrieved_card_sections": "- Drive: forward motion.",
            "retrieved_position_text": "- Past: what was.",
        }
        result = chain.invoke(ctx)
        assert isinstance(result, str)
        assert "Stub interpretation" in result


# ---------------------------------------------------------------------------
# build_per_card_context
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildPerCardContext:
    def test_returns_dict_with_all_keys(
        self,
        sample_card: Card,
        sample_spread: Spread,
        sample_position: SpreadPosition,
        sample_dealt_card: DealtCard,
        stub_embedder_factory: object,
    ) -> None:
        embedder = stub_embedder_factory(4)  # type: ignore[operator]
        with VectorStore(":memory:", dimension=4) as store:
            ctx = build_per_card_context(
                dealt=sample_dealt_card,
                card=sample_card,
                spread=sample_spread,
                position=sample_position,
                vector_store=store,
                embedder=embedder,
            )
        assert set(ctx.keys()) == {
            "card_name",
            "orientation",
            "position_name",
            "position_meaning",
            "retrieved_card_sections",
            "retrieved_position_text",
        }
        assert ctx["card_name"] == "The Fool"
        assert ctx["orientation"] == "upright"
        assert ctx["position_name"] == "Past"

    def test_orientation_value_is_string(
        self,
        sample_card: Card,
        sample_spread: Spread,
        sample_position: SpreadPosition,
        sample_dealt_reversed: DealtCard,
        stub_embedder_factory: object,
    ) -> None:
        embedder = stub_embedder_factory(4)  # type: ignore[operator]
        with VectorStore(":memory:", dimension=4) as store:
            ctx = build_per_card_context(
                dealt=sample_dealt_reversed,
                card=sample_card,
                spread=sample_spread,
                position=sample_position,
                vector_store=store,
                embedder=embedder,
            )
        assert ctx["orientation"] == "reversed"

    def test_empty_store_returns_empty_marker(
        self,
        sample_card: Card,
        sample_spread: Spread,
        sample_position: SpreadPosition,
        sample_dealt_card: DealtCard,
        stub_embedder_factory: object,
    ) -> None:
        embedder = stub_embedder_factory(4)  # type: ignore[operator]
        with VectorStore(":memory:", dimension=4) as store:
            ctx = build_per_card_context(
                dealt=sample_dealt_card,
                card=sample_card,
                spread=sample_spread,
                position=sample_position,
                vector_store=store,
                embedder=embedder,
            )
        assert ctx["retrieved_card_sections"] == "(none retrieved)"
        assert ctx["retrieved_position_text"] == "(none retrieved)"

    def test_retrieves_card_sections(
        self,
        sample_card: Card,
        sample_spread: Spread,
        sample_position: SpreadPosition,
        sample_dealt_card: DealtCard,
        stub_embedder_factory: object,
    ) -> None:
        embedder = stub_embedder_factory(4)  # type: ignore[operator]
        with VectorStore(":memory:", dimension=4) as store:
            store.add_chunks(
                [
                    _make_chunk(
                        "Pure potential, unformed.",
                        card_id="the-fool",
                        section=CardSection.DRIVE,
                    ),
                    _make_chunk(
                        "Spontaneity and curiosity.",
                        card_id="the-fool",
                        section=CardSection.LIGHT,
                    ),
                ]
            )
            ctx = build_per_card_context(
                dealt=sample_dealt_card,
                card=sample_card,
                spread=sample_spread,
                position=sample_position,
                vector_store=store,
                embedder=embedder,
                k=2,
            )
        assert "Pure potential" in ctx["retrieved_card_sections"]
        assert "Spontaneity" in ctx["retrieved_card_sections"]

    def test_retrieves_position_chunk(
        self,
        sample_card: Card,
        sample_spread: Spread,
        sample_position: SpreadPosition,
        sample_dealt_card: DealtCard,
        stub_embedder_factory: object,
    ) -> None:
        embedder = stub_embedder_factory(4)  # type: ignore[operator]
        with VectorStore(":memory:", dimension=4) as store:
            store.add_chunks(
                [
                    _make_chunk(
                        "What was set in motion before now.",
                        spread_id="new-moon-three-card",
                        position_index=0,
                    )
                ]
            )
            ctx = build_per_card_context(
                dealt=sample_dealt_card,
                card=sample_card,
                spread=sample_spread,
                position=sample_position,
                vector_store=store,
                embedder=embedder,
            )
        assert "set in motion" in ctx["retrieved_position_text"]

    def test_uses_position_index_from_spread(
        self,
        sample_card: Card,
        sample_spread: Spread,
        sample_dealt_card: DealtCard,
        stub_embedder_factory: object,
    ) -> None:
        embedder = stub_embedder_factory(4)  # type: ignore[operator]
        present_position = sample_spread.position_by_index(1)
        with VectorStore(":memory:", dimension=4) as store:
            store.add_chunks(
                [
                    _make_chunk(
                        "Present energy at the new moon.",
                        spread_id="new-moon-three-card",
                        position_index=1,
                    )
                ]
            )
            ctx = build_per_card_context(
                dealt=sample_dealt_card,
                card=sample_card,
                spread=sample_spread,
                position=present_position,
                vector_store=store,
                embedder=embedder,
            )
        assert "Present energy" in ctx["retrieved_position_text"]

    def test_uses_card_query_with_orientation(
        self,
        sample_card: Card,
        sample_spread: Spread,
        sample_position: SpreadPosition,
        sample_dealt_reversed: DealtCard,
    ) -> None:
        """The internal query is ``"{card_name} {orientation}"``."""
        with VectorStore(":memory:", dimension=4) as store:
            # Capture what the embedder sees by intercepting embed_query.
            captured: list[str] = []

            class _Spy(EmbedderSpy):  # type: ignore[misc, valid-type]
                def embed_query(self, text: str) -> list[float]:
                    captured.append(text)
                    return [0.0] * 4

            spy = _Spy()
            ctx = build_per_card_context(
                dealt=sample_dealt_reversed,
                card=sample_card,
                spread=sample_spread,
                position=sample_position,
                vector_store=store,
                embedder=spy,  # type: ignore[arg-type]
            )
        assert captured == ["The Fool reversed"]
        # And the dict is still well-formed.
        assert ctx["card_name"] == "The Fool"


# ---------------------------------------------------------------------------
# Tiny test helper — bare Embedder subclass that records calls.
# ---------------------------------------------------------------------------


class EmbedderSpy:
    """Duck-typed stand-in for :class:`Embedder` used to assert the chain
    passed the expected query string to ``embed_query``.

    Implements only the surface that :func:`build_per_card_context` uses.
    """

    def embed_query(self, text: str) -> list[float]:  # pragma: no cover - subclass overrides
        raise NotImplementedError

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 4 for _ in texts]

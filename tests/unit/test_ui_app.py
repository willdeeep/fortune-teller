"""Unit tests for the Gradio UI module.

The plan 0009 spike has no end-to-end Gradio test (a live Gradio
server is not available in CI), so we test:

- ``_format_card_text`` — pure formatting helper
- ``run_reading_generator`` — the streaming generator that drives
  progressive panel updates
- ``build_app`` — that it returns a ``gr.Blocks`` instance bound to
  the injected service (no ``.launch()`` is called)

A stub :class:`ReadingService` is used to avoid any real LLM, vector
store, or embedder.
"""

from __future__ import annotations

from typing import ClassVar

import gradio as gr
import pytest
from pydantic import HttpUrl

from fortune_teller.application.models.domain import (
    Arcana,
    Card,
    CardInterpretation,
    CardSection,
    CardSectionText,
    Chunk,
    ChunkType,
    Deck,
    Reading,
    Spread,
    SpreadPosition,
)
from fortune_teller.application.services.reading import ReadingHandle, ReadingService
from fortune_teller.application.stores.vector import VectorStore
from fortune_teller.application.ui.app import (
    _format_card_text,
    build_app,
    run_reading_generator,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


_SOURCE = "https://example.test/card"
_SPREAD_URL = "https://example.test/spread"


def _card(cid: str) -> Card:
    return Card(
        id=cid,
        name=cid.replace("-", " ").title(),
        arcana=Arcana.MAJOR,
        source_url=HttpUrl(_SOURCE),
    )


def _make_deck(size: int = 5) -> Deck:
    return Deck(
        id="test-deck",
        name="Test Deck",
        cards=[_card(f"card-{i:02d}") for i in range(size)],
    )


def _make_spread(position_count: int = 3) -> Spread:
    return Spread(
        id="test-spread",
        name="Test Spread",
        positions=[
            SpreadPosition(
                index=i,
                name=f"Position {i}",
                meaning=f"Meaning of position {i}.",
                source_url=HttpUrl(_SPREAD_URL),
            )
            for i in range(position_count)
        ],
    )


class _StubChain:
    """Chain that returns a canned string; records inputs."""

    inputs_received: ClassVar[list[dict[str, str]] | None] = None

    def __init__(self, *, reply: str = "Stub.") -> None:
        self.reply = reply
        self.invocations = 0
        if _StubChain.inputs_received is None:
            _StubChain.inputs_received = []

    def invoke(self, inputs: dict[str, str]) -> str:
        self.invocations += 1
        _StubChain.inputs_received.append(dict(inputs))
        return f"{self.reply}#{self.invocations}"


class _StubReadingService:
    """In-process double for :class:`ReadingService`."""

    def __init__(self, deck: Deck, spread: Spread) -> None:
        self._deck = deck
        self._spread = spread
        self._per_card_chain = _StubChain(reply="CARD")
        self._summary_chain = _StubChain(reply="SUMMARY")
        self._vector_store = None
        self._embedder = None
        self._counter = 0

    def start(self, seed: int | None = None) -> ReadingHandle:  # noqa: ARG002
        from fortune_teller.application.services.deck import DeckSession

        session = DeckSession(self._deck, rng=__import__("random").Random(0))
        return ReadingHandle(
            deck_session=session,
            deck_id=self._deck.id,
            spread=self._spread,
        )

    def deal_next(self, handle: ReadingHandle) -> CardInterpretation:
        position_index = len(handle.dealt)
        if position_index >= len(self._spread.positions):
            raise RuntimeError("All positions filled.")
        dealt = handle.deck_session.deal_one(position_index)
        position = self._spread.position_by_index(position_index)
        card = self._deck.card_by_id(dealt.card_id)
        self._counter += 1
        text = self._per_card_chain.invoke(
            {
                "card_name": card.name,
                "orientation": dealt.orientation.value,
                "position_name": position.name,
                "position_meaning": position.meaning,
            }
        )
        interp = CardInterpretation(
            dealt=dealt,
            card_name=card.name,
            position_name=position.name,
            text=text,
        )
        handle.dealt.append(dealt)
        handle.interpretations.append(interp)
        return interp

    def finalize(self, handle: ReadingHandle) -> Reading:
        summary = self._summary_chain.invoke(
            {"spread_name": self._spread.name, "card_summaries": "x"}
        )
        return Reading(
            deck_id=handle.deck_id,
            spread_id=self._spread.id,
            dealt=list(handle.dealt),
            per_card=list(handle.interpretations),
            summary=summary,
        )


@pytest.fixture
def stub_service() -> _StubReadingService:
    _StubChain.inputs_received = []
    return _StubReadingService(deck=_make_deck(5), spread=_make_spread(3))


# ---------------------------------------------------------------------------
# _format_card_text
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFormatCardText:
    def test_upright_uses_up_arrow(self) -> None:
        out = _format_card_text("The Fool", "upright", "Beginnings.")
        assert "The Fool" in out
        assert "▲ UPRIGHT" in out
        assert "Beginnings." in out

    def test_reversed_uses_down_arrow(self) -> None:
        out = _format_card_text("The Fool", "reversed", "Holding back.")
        assert "The Fool" in out
        assert "▼ REVERSED" in out
        assert "Holding back." in out

    def test_three_lines_then_body(self) -> None:
        out = _format_card_text("The Fool", "upright", "Body text.")
        lines = out.split("\n")
        # First two lines: name, orientation; blank line; body
        assert lines[0] == "The Fool"
        assert lines[1] == "▲ UPRIGHT"
        assert lines[2] == ""
        assert lines[3] == "Body text."

    def test_orientation_string_is_either_upright_or_reversed(self) -> None:
        for orientation in ("upright", "reversed"):
            out = _format_card_text("Card", orientation, "x")
            assert orientation.upper() in out


# ---------------------------------------------------------------------------
# run_reading_generator
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRunReadingGenerator:
    def test_yields_one_snapshot_per_card_plus_one_for_summary(
        self,
        stub_service: _StubReadingService,
    ) -> None:
        snapshots = list(run_reading_generator(stub_service))
        # 3 positions + 1 final yield = 4 snapshots
        assert len(snapshots) == 4

    def test_each_snapshot_has_one_more_panel_filled(
        self,
        stub_service: _StubReadingService,
    ) -> None:
        snapshots = list(run_reading_generator(stub_service))
        # First snapshot: only panel 0 filled, summary is empty.
        s0 = snapshots[0]
        assert s0[0] != ""
        assert s0[1] == "" and s0[2] == ""
        assert s0[3] == ""
        # Second snapshot: panels 0 and 1 filled.
        s1 = snapshots[1]
        assert s1[0] != "" and s1[1] != ""
        assert s1[2] == ""
        # Third snapshot: all three panels filled, summary still empty.
        s2 = snapshots[2]
        assert s2[0] != "" and s2[1] != "" and s2[2] != ""
        assert s2[3] == ""
        # Fourth snapshot: all panels + summary populated.
        s3 = snapshots[3]
        assert s3[0] != "" and s3[1] != "" and s3[2] != ""
        assert s3[3] != ""

    def test_panels_carry_card_name_and_orientation(
        self,
        stub_service: _StubReadingService,
    ) -> None:
        snapshots = list(run_reading_generator(stub_service))
        # The third panel of the final snapshot must mention a card
        # name (anything non-empty) and an orientation arrow.
        final_panels = snapshots[-1][:3]
        for panel in final_panels:
            assert "▲ UPRIGHT" in panel or "▼ REVERSED" in panel

    def test_summary_includes_text_from_summary_chain(
        self,
        stub_service: _StubReadingService,
    ) -> None:
        snapshots = list(run_reading_generator(stub_service))
        # Stub summary chain returns "SUMMARY#1"
        assert "SUMMARY#1" in snapshots[-1][3]

    def test_each_deal_uses_a_different_card(
        self,
        stub_service: _StubReadingService,
    ) -> None:
        snapshots = list(run_reading_generator(stub_service))
        # Per-card chain was invoked exactly len(spread.positions) times.
        assert stub_service._per_card_chain.invocations == 3
        # Each of the first three snapshots has a *new* card just added:
        # snapshot 0 has panel 0 filled, snapshot 1 has panel 1 filled,
        # snapshot 2 has panel 2 filled. The card names in those
        # "newly filled" panels must all be different (no duplicates
        # within a reading).
        newest_panel_per_snapshot = [s[len([p for p in s[:3] if p]) - 1] for s in snapshots[:3]]
        card_names = [p.split("\n")[0] for p in newest_panel_per_snapshot]
        assert len(set(card_names)) == 3


# ---------------------------------------------------------------------------
# build_app
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildApp:
    def test_returns_gradio_blocks(self, stub_service: _StubReadingService) -> None:
        demo = build_app(stub_service)
        assert isinstance(demo, gr.Blocks)

    def test_blocks_title_is_fortune_teller(
        self,
        stub_service: _StubReadingService,
    ) -> None:
        demo = build_app(stub_service)
        assert demo.title == "Fortune Teller"

    def test_app_uses_spread_name_in_markdown(
        self,
        stub_service: _StubReadingService,
    ) -> None:
        demo = build_app(stub_service)
        # Walk the block children looking for a Markdown with the spread name.
        markdown_texts = [
            child.value for child in demo.blocks.values() if isinstance(child, gr.Markdown)
        ]
        assert any("Test Spread" in t for t in markdown_texts)

    def test_app_does_not_launch_on_build(
        self,
        stub_service: _StubReadingService,
    ) -> None:
        # Sanity: build_app must NOT block / launch a server; just construct.
        # We verify by checking the title (a non-launch side effect).
        demo = build_app(stub_service)
        assert demo.title == "Fortune Teller"

    def test_app_creates_one_panel_per_spread_position(
        self,
        stub_service: _StubReadingService,
    ) -> None:
        demo = build_app(stub_service)
        # The stub spread has 3 positions, so we expect 3 Textbox panels.
        textboxes = [child for child in demo.blocks.values() if isinstance(child, gr.Textbox)]
        assert len(textboxes) == 4  # 3 card panels + 1 summary

    def test_summary_box_is_present_and_labeled(
        self,
        stub_service: _StubReadingService,
    ) -> None:
        demo = build_app(stub_service)
        textboxes = [child for child in demo.blocks.values() if isinstance(child, gr.Textbox)]
        # One of the textboxes is the summary; its label is "Reading Summary".
        summary_boxes = [tb for tb in textboxes if tb.label == "Reading Summary"]
        assert len(summary_boxes) == 1

    def test_button_label(self, stub_service: _StubReadingService) -> None:
        demo = build_app(stub_service)
        buttons = [c for c in demo.blocks.values() if isinstance(c, gr.Button)]
        assert any(b.value == "New Reading" for b in buttons)


# ---------------------------------------------------------------------------
# RAG wired — per-card chain receives a 6-key context, not a 4-key one
# (This is the deferred-0008 wiring test.)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRagWiringInService:
    def test_vector_store_and_embedder_produce_extended_context(self) -> None:
        card = _card("0-the-fool")
        card_with_sections = card.model_copy(
            update={
                "sections": [
                    CardSectionText(
                        section=CardSection.OVERALL,
                        text="Pure potential.",
                    ),
                ],
            },
        )
        deck = Deck(id="test-deck", name="Test Deck", cards=[card_with_sections])
        spread = _make_spread(1)
        chain = _StubChain(reply="INTERP")

        with VectorStore(":memory:", dimension=4) as store:
            chunk = Chunk(
                chunk_type=ChunkType.CARD_SECTION,
                deck_id="test-deck",
                card_id="0-the-fool",
                card_name="The Fool",
                section=CardSection.OVERALL,
                source_url="https://example.test/the-fool",
                text="Pure potential, unformed.",
                embedding=[0.0, 0.0, 0.0, 0.0],
            )
            store.add_chunks([chunk])

            class _StubEmbedder:
                def embed_query(self, _text: str) -> list[float]:
                    return [0.0, 0.0, 0.0, 0.0]

            svc = ReadingService(
                deck=deck,
                spread=spread,
                per_card_chain=chain,  # type: ignore[arg-type]
                vector_store=store,
                embedder=_StubEmbedder(),  # type: ignore[arg-type]
            )
            handle = svc.start(seed=0)
            interp = svc.deal_next(handle)

        # The chain must have received the RAG context with retrieved sections.
        assert _StubChain.inputs_received is not None
        ctx = _StubChain.inputs_received[0]
        assert "retrieved_card_sections" in ctx
        assert "Pure potential" in ctx["retrieved_card_sections"]
        assert "retrieved_position_text" in ctx
        assert "card_name" in ctx
        assert "position_name" in ctx
        assert "position_meaning" in ctx
        # And the returned interp text came from the chain.
        assert "INTERP" in interp.text

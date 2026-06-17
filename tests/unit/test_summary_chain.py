"""Unit tests for :mod:`fortune_teller.application.chains.summary`."""

from __future__ import annotations

import pytest
from langchain_core.runnables import RunnableLambda, RunnableSequence
from pydantic import HttpUrl

from fortune_teller.application.chains.summary import (
    build_summary_chain,
    build_summary_context,
    render_synergy_block,
    summary_prompt,
)
from fortune_teller.application.models.domain import (
    CardInterpretation,
    DealtCard,
    Orientation,
    Spread,
    SpreadPosition,
)
from fortune_teller.application.services.synergy import SynergyHit


def _make_spread() -> Spread:
    return Spread(
        id="new-moon-three-card",
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
            SpreadPosition(
                index=2,
                name="Future",
                meaning="What is being born.",
                source_url=HttpUrl("https://example.com/spread-new-moon/"),
            ),
        ],
    )


def _make_interpretation(
    position_index: int,
    position_name: str,
    card_name: str = "The Fool",
    orientation: Orientation = Orientation.UPRIGHT,
    text: str = "A new beginning.",
) -> CardInterpretation:
    return CardInterpretation(
        dealt=DealtCard(
            card_id=card_name.lower().replace(" ", "-"),
            orientation=orientation,
            position_index=position_index,
        ),
        card_name=card_name,
        position_name=position_name,
        text=text,
    )


# ---------------------------------------------------------------------------
# summary_prompt rendering
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSummaryPromptRenders:
    def test_renders_two_messages(self) -> None:
        msgs = summary_prompt.format_messages(
            spread_name="New Moon Three-Card",
            card_summaries="Position 0 — Past: new beginning.",
            spread_description="Past: What has been.",
            synergy_block="",
        )
        assert len(msgs) == 2
        assert msgs[0].type == "system"
        assert msgs[1].type == "human"

    def test_renders_spread_name(self) -> None:
        msgs = summary_prompt.format_messages(
            spread_name="New Moon Three-Card",
            card_summaries="Position 0 — Past: new beginning.",
            spread_description="Past: What has been.",
            synergy_block="",
        )
        assert "New Moon Three-Card" in msgs[1].content

    def test_renders_empty_card_summaries(self) -> None:
        msgs = summary_prompt.format_messages(
            spread_name="X",
            card_summaries="(no cards dealt yet)",
            spread_description="(no positions)",
            synergy_block="",
        )
        assert "(no cards dealt yet)" in msgs[1].content
        assert "(no positions)" in msgs[1].content


# ---------------------------------------------------------------------------
# build_summary_chain
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildSummaryChain:
    def test_returns_runnable_sequence(self, stub_llm: RunnableLambda) -> None:  # type: ignore[type-arg]
        chain = build_summary_chain(stub_llm)
        assert isinstance(chain, RunnableSequence)

    def test_invokes_to_string(self, stub_llm: RunnableLambda) -> None:  # type: ignore[type-arg]
        chain = build_summary_chain(stub_llm)
        result = chain.invoke(
            {
                "spread_name": "New Moon",
                "card_summaries": "Position 0 — Past: x.",
                "spread_description": "Past: y.",
                "synergy_block": "",
            }
        )
        assert isinstance(result, str)
        assert "Stub interpretation" in result


# ---------------------------------------------------------------------------
# build_summary_context
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildSummaryContext:
    def test_returns_dict_with_all_keys(self) -> None:
        spread = _make_spread()
        interps = [_make_interpretation(0, "Past"), _make_interpretation(1, "Present")]
        ctx = build_summary_context(interps, spread)
        assert set(ctx.keys()) == {
            "spread_name",
            "card_summaries",
            "spread_description",
            "synergy_block",
        }

    def test_spread_name(self) -> None:
        spread = _make_spread()
        ctx = build_summary_context([], spread)
        assert ctx["spread_name"] == "New Moon Three-Card Spread"

    def test_spread_description_includes_all_positions(self) -> None:
        spread = _make_spread()
        ctx = build_summary_context([], spread)
        assert "Past: What has been." in ctx["spread_description"]
        assert "Present: What is." in ctx["spread_description"]
        assert "Future: What is being born." in ctx["spread_description"]

    def test_card_summaries_include_position_and_card(self) -> None:
        spread = _make_spread()
        interps = [
            _make_interpretation(0, "Past", card_name="The Fool", text="A leap of faith."),
            _make_interpretation(1, "Present", card_name="The Magus", text="Active will."),
        ]
        ctx = build_summary_context(interps, spread)
        assert "Position 0 — Past (The Fool, upright):" in ctx["card_summaries"]
        assert "A leap of faith." in ctx["card_summaries"]
        assert "Position 1 — Present (The Magus, upright):" in ctx["card_summaries"]
        assert "Active will." in ctx["card_summaries"]

    def test_card_summaries_separated_by_blank_line(self) -> None:
        spread = _make_spread()
        interps = [
            _make_interpretation(0, "Past", text="first."),
            _make_interpretation(1, "Present", text="second."),
        ]
        ctx = build_summary_context(interps, spread)
        # Two newlines separate the per-card blocks.
        assert "first.\n\nPosition 1" in ctx["card_summaries"]

    def test_uses_dealt_position_index_not_enumerate_index(self) -> None:
        """The label uses ``interp.dealt.position_index``, not the list index."""
        spread = _make_spread()
        # Dealt in order, but with position 2 first in the list.
        interps = [
            _make_interpretation(2, "Future", text="third."),
            _make_interpretation(0, "Past", text="first."),
        ]
        ctx = build_summary_context(interps, spread)
        assert "Position 2 — Future" in ctx["card_summaries"]
        assert "Position 0 — Past" in ctx["card_summaries"]

    def test_orientation_rendered(self) -> None:
        spread = _make_spread()
        interps = [_make_interpretation(0, "Past", orientation=Orientation.REVERSED)]
        ctx = build_summary_context(interps, spread)
        assert "(The Fool, reversed):" in ctx["card_summaries"]

    def test_empty_interpretations_yields_empty_summaries(self) -> None:
        spread = _make_spread()
        ctx = build_summary_context([], spread)
        assert ctx["card_summaries"] == ""
        # spread_description is still populated
        assert "Past:" in ctx["spread_description"]

    def test_synergy_block_empty_when_no_synergies(self) -> None:
        spread = _make_spread()
        ctx = build_summary_context([], spread, synergies=[])
        assert ctx["synergy_block"] == ""

    def test_synergy_block_empty_when_synergies_none(self) -> None:
        spread = _make_spread()
        ctx = build_summary_context([], spread, synergies=None)
        assert ctx["synergy_block"] == ""

    def test_synergy_block_rendered_with_hits(self) -> None:
        spread = _make_spread()
        interps = [_make_interpretation(0, "Past")]
        hits = [
            SynergyHit(
                card_id_a="the-fool",
                card_id_b="the-magician",
                card_name_a="The Fool",
                card_name_b="The Magician",
                orientation_a=Orientation.UPRIGHT,
                orientation_b=Orientation.REVERSED,
                base="reinforce",
                effective="oppose",
            ),
        ]
        ctx = build_summary_context(interps, spread, synergies=hits)
        assert "Card synergies:" in ctx["synergy_block"]
        assert "The Fool" in ctx["synergy_block"]
        assert "The Magician" in ctx["synergy_block"]
        assert "oppose" in ctx["synergy_block"]


# ---------------------------------------------------------------------------
# render_synergy_block
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRenderSynergyBlock:
    def test_empty_list_returns_empty_string(self) -> None:
        assert render_synergy_block([]) == ""

    def test_single_reinforce_hit(self) -> None:
        hits = [
            SynergyHit(
                card_id_a="the-fool",
                card_id_b="the-magician",
                card_name_a="The Fool",
                card_name_b="The Magician",
                orientation_a=Orientation.UPRIGHT,
                orientation_b=Orientation.UPRIGHT,
                base="reinforce",
                effective="reinforce",
            ),
        ]
        result = render_synergy_block(hits)
        assert "Card synergies:" in result
        assert "The Fool (upright)" in result
        assert "The Magician (upright)" in result
        assert "reinforce" in result

    def test_oppose_hit_shows_oppose(self) -> None:
        hits = [
            SynergyHit(
                card_id_a="the-fool",
                card_id_b="the-tower",
                card_name_a="The Fool",
                card_name_b="The Tower",
                orientation_a=Orientation.UPRIGHT,
                orientation_b=Orientation.REVERSED,
                base="oppose",
                effective="reinforce",
            ),
        ]
        result = render_synergy_block(hits)
        assert "reinforce" in result

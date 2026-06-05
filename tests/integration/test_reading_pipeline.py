"""Integration tests for the per-card and summary interpretation chains.

End-to-end exercises both chains with a stub :class:`~langchain_core.runnables.Runnable`
LLM that never touches the network, and verifies:

- Per-card chain produces a non-empty string for a complete context dict.
- Summary chain produces a non-empty string from per-card interpretations.
- The chain prompt + stub LLM round-trip is reproducible.
- The chains do not depend on real vector stores or embedding models.

Marked ``@pytest.mark.integration`` because the test exercises the full
LangChain pipeline (prompt template + runnable sequence + output parser),
not just an isolated component.
"""

from __future__ import annotations

from typing import ClassVar

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from pydantic import HttpUrl

from fortune_teller.application.chains.per_card import (
    build_per_card_chain,
    per_card_prompt,
)
from fortune_teller.application.chains.summary import (
    build_summary_chain,
    build_summary_context,
    summary_prompt,
)
from fortune_teller.application.models.domain import (
    CardInterpretation,
    DealtCard,
    Orientation,
    Spread,
    SpreadPosition,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_STUB_PER_CARD_TEXT = "Stub per-card interpretation."
_STUB_SUMMARY_TEXT = "Stub reading summary."


def _stub_per_card_llm() -> RunnableLambda:  # type: ignore[type-arg]
    """Return a stub LLM that always produces the canned per-card text."""
    return RunnableLambda(lambda _: AIMessage(content=_STUB_PER_CARD_TEXT))


def _stub_summary_llm() -> RunnableLambda:  # type: ignore[type-arg]
    """Return a stub LLM that always produces the canned summary text."""
    return RunnableLambda(lambda _: AIMessage(content=_STUB_SUMMARY_TEXT))


def _build_spread() -> Spread:
    return Spread(
        id="new-moon-three-card",
        name="New Moon Three-Card Spread",
        positions=[
            SpreadPosition(
                index=0,
                name="Past",
                meaning="What has been.",
                source_url=HttpUrl("https://example.com/x/"),
            ),
            SpreadPosition(
                index=1,
                name="Present",
                meaning="What is.",
                source_url=HttpUrl("https://example.com/x/"),
            ),
            SpreadPosition(
                index=2,
                name="Future",
                meaning="What may come.",
                source_url=HttpUrl("https://example.com/x/"),
            ),
        ],
    )


def _build_interpretations() -> list[CardInterpretation]:
    return [
        CardInterpretation(
            dealt=DealtCard(card_id="the-fool", orientation=Orientation.UPRIGHT, position_index=0),
            card_name="The Fool",
            position_name="Past",
            text="A new beginning.",
        ),
        CardInterpretation(
            dealt=DealtCard(card_id="the-magus", orientation=Orientation.UPRIGHT, position_index=1),
            card_name="The Magus",
            position_name="Present",
            text="A call to action.",
        ),
        CardInterpretation(
            dealt=DealtCard(
                card_id="the-priestess", orientation=Orientation.REVERSED, position_index=2
            ),
            card_name="The Priestess",
            position_name="Future",
            text="Hidden knowledge emerging.",
        ),
    ]


# ---------------------------------------------------------------------------
# Per-card chain
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestPerCardChainEndToEnd:
    """End-to-end: per_card_prompt → stub LLM → StrOutputParser → str."""

    _CONTEXT: ClassVar[dict[str, str]] = {
        "card_name": "The Fool",
        "orientation": "upright",
        "position_name": "Past",
        "position_meaning": "What has been.",
        "retrieved_card_sections": "- Pure potential.\n- Spontaneity.",
        "retrieved_position_text": "- What has been.",
    }

    def test_chain_returns_non_empty_string(self) -> None:
        """The full per-card chain returns a non-empty string from a context dict."""
        chain = build_per_card_chain(_stub_per_card_llm())
        result = chain.invoke(self._CONTEXT)
        assert isinstance(result, str)
        assert result == _STUB_PER_CARD_TEXT
        assert len(result) > 0

    def test_chain_does_not_call_real_llm(self) -> None:
        """The chain works with a stub LLM, exercising no network endpoint."""
        # If this test takes any notable time, something is calling the network.
        # The stub LLM returns immediately, so the chain completes fast.
        chain = build_per_card_chain(_stub_per_card_llm())
        result = chain.invoke(self._CONTEXT)
        # The stub LLM doesn't know the prompt's content; it returns canned text.
        # If the real LLM had been called, we'd see LLM-generated text here.
        assert result == _STUB_PER_CARD_TEXT

    def test_chain_is_deterministic_with_stub(self) -> None:
        """The stub LLM is deterministic — same input, same output."""
        chain = build_per_card_chain(_stub_per_card_llm())
        first = chain.invoke(self._CONTEXT)
        second = chain.invoke(self._CONTEXT)
        assert first == second

    def test_chain_handles_different_contexts(self) -> None:
        """Different inputs flow through the chain without error."""
        chain = build_per_card_chain(_stub_per_card_llm())
        # Reversed orientation
        reversed_ctx = {**self._CONTEXT, "orientation": "reversed"}
        result = chain.invoke(reversed_ctx)
        assert result == _STUB_PER_CARD_TEXT
        # Different position
        present_ctx = {**self._CONTEXT, "position_name": "Present"}
        result = chain.invoke(present_ctx)
        assert result == _STUB_PER_CARD_TEXT

    def test_prompt_input_variables_match_chain_inputs(self) -> None:
        """The per-card chain's required inputs match the prompt's placeholders."""
        prompt_vars = set(per_card_prompt.input_variables)
        context_keys = set(self._CONTEXT.keys())
        assert prompt_vars <= context_keys, (
            f"Missing context keys for prompt placeholders: {prompt_vars - context_keys}"
        )


# ---------------------------------------------------------------------------
# Summary chain
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestSummaryChainEndToEnd:
    """End-to-end: summary_prompt → stub LLM → StrOutputParser → str."""

    _SPREAD = _build_spread()
    _INTERPRETATIONS = _build_interpretations()

    def _context(self) -> dict[str, str]:
        return build_summary_context(self._INTERPRETATIONS, self._SPREAD)

    def test_chain_returns_non_empty_string(self) -> None:
        """The full summary chain returns a non-empty string from real contexts."""
        chain = build_summary_chain(_stub_summary_llm())
        result = chain.invoke(self._context())
        assert isinstance(result, str)
        assert result == _STUB_SUMMARY_TEXT
        assert len(result) > 0

    def test_chain_includes_spread_name_in_prompt(self) -> None:
        """The prompt receives the spread name and card summaries."""
        ctx = self._context()
        assert ctx["spread_name"] == "New Moon Three-Card Spread"
        assert "The Fool" in ctx["card_summaries"]
        assert "The Magus" in ctx["card_summaries"]
        assert "The Priestess" in ctx["card_summaries"]
        assert "Past" in ctx["spread_description"]
        assert "Present" in ctx["spread_description"]
        assert "Future" in ctx["spread_description"]

    def test_chain_does_not_call_real_llm(self) -> None:
        """The chain works with a stub LLM, exercising no network endpoint."""
        chain = build_summary_chain(_stub_summary_llm())
        result = chain.invoke(self._context())
        assert result == _STUB_SUMMARY_TEXT

    def test_chain_is_deterministic_with_stub(self) -> None:
        """The stub LLM is deterministic — same input, same output."""
        chain = build_summary_chain(_stub_summary_llm())
        first = chain.invoke(self._context())
        second = chain.invoke(self._context())
        assert first == second

    def test_prompt_input_variables_match_chain_inputs(self) -> None:
        """The summary chain's required inputs match the prompt's placeholders."""
        prompt_vars = set(summary_prompt.input_variables)
        context_keys = set(self._context().keys())
        assert prompt_vars <= context_keys, (
            f"Missing context keys for prompt placeholders: {prompt_vars - context_keys}"
        )


# ---------------------------------------------------------------------------
# Full pipeline: per-card → summary
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestFullReadingPipeline:
    """Run a fake three-card reading through both chains end-to-end."""

    _SPREAD = _build_spread()
    _INTERPRETATIONS = _build_interpretations()

    def test_full_pipeline_produces_summary_from_interpretations(self) -> None:
        """The summary chain consumes the per-card outputs without error."""
        per_card_chain = build_per_card_chain(_stub_per_card_llm())
        summary_chain = build_summary_chain(_stub_summary_llm())

        # Per-card pass: build context, invoke chain, store the result
        per_card_outputs: list[str] = []
        for interp in self._INTERPRETATIONS:
            ctx = {
                "card_name": interp.card_name,
                "orientation": interp.dealt.orientation.value,
                "position_name": interp.position_name,
                "position_meaning": next(
                    p.meaning for p in self._SPREAD.positions if p.name == interp.position_name
                ),
                "retrieved_card_sections": "(none retrieved)",
                "retrieved_position_text": "(none retrieved)",
            }
            per_card_outputs.append(per_card_chain.invoke(ctx))

        # All per-card outputs are the stub text
        assert all(o == _STUB_PER_CARD_TEXT for o in per_card_outputs)

        # Build a fresh list of CardInterpretations using the chain output as text
        # and feed it into the summary chain.
        rebuilt = [
            CardInterpretation(
                dealt=interp.dealt,
                card_name=interp.card_name,
                position_name=interp.position_name,
                text=per_card_outputs[i],
            )
            for i, interp in enumerate(self._INTERPRETATIONS)
        ]
        summary_ctx = build_summary_context(rebuilt, self._SPREAD)
        summary_output = summary_chain.invoke(summary_ctx)

        assert summary_output == _STUB_SUMMARY_TEXT

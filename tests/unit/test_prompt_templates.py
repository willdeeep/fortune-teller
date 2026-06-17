"""Unit tests for the prompt template modules.

Verifies the :class:`~langchain_core.prompts.ChatPromptTemplate` instances
in :mod:`fortune_teller.application.chains.per_card` and
:mod:`fortune_teller.application.chains.summary` render correctly with
the expected placeholders, and fail loudly when a required placeholder
is missing.

These tests use the in-memory :func:`langchain_core.prompts.ChatPromptTemplate.format`
path — no LLM is invoked, so they are fast (unit-marked) and have no
network or filesystem dependencies.
"""

from __future__ import annotations

import inspect
from typing import ClassVar

import pytest
from langchain_core.prompts import ChatPromptTemplate
from pydantic import HttpUrl

from fortune_teller.application.chains.per_card import (
    build_per_card_context,
    per_card_prompt,
)
from fortune_teller.application.chains.summary import (
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

_CARD_NAME = "The Fool"
_ORIENTATION = Orientation.UPRIGHT.value
_POSITION_NAME = "Past"
_POSITION_MEANING = "What has been."
_CARD_SECTIONS = "- Pure potential.\n- Spontaneity."
_POSITION_TEXT = "- What has been."
_SPREAD_NAME = "New Moon Three-Card Spread"
_SPREAD_DESCRIPTION = "Past: What has been.\nPresent: What is."


def _full_per_card_dict() -> dict[str, str]:
    """Return a complete dict of placeholders for the per-card prompt."""
    return {
        "card_name": _CARD_NAME,
        "orientation": _ORIENTATION,
        "position_name": _POSITION_NAME,
        "position_meaning": _POSITION_MEANING,
        "retrieved_card_sections": _CARD_SECTIONS,
        "retrieved_position_text": _POSITION_TEXT,
    }


def _full_summary_dict() -> dict[str, str]:
    """Return a complete dict of placeholders for the summary prompt."""
    return {
        "spread_name": _SPREAD_NAME,
        "card_summaries": "Position 0 — Past (The Fool, upright):\nBeginnings.",
        "spread_description": _SPREAD_DESCRIPTION,
        "synergy_block": "",
    }


# ---------------------------------------------------------------------------
# Per-card prompt
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPerCardPromptTemplate:
    """Render and structural tests for :data:`per_card_prompt`."""

    _REQUIRED_KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "card_name",
            "orientation",
            "position_name",
            "position_meaning",
            "retrieved_card_sections",
            "retrieved_position_text",
        }
    )

    def test_is_a_chat_prompt_template(self) -> None:
        """The module-level prompt is a :class:`ChatPromptTemplate` instance."""
        assert isinstance(per_card_prompt, ChatPromptTemplate)

    def test_renders_both_system_and_human_messages(self) -> None:
        """Format with all placeholders yields a system + human message pair."""
        messages = per_card_prompt.format_messages(**_full_per_card_dict())
        # 2 messages: one system, one human
        assert len(messages) == 2
        types = [type(m).__name__ for m in messages]
        assert "SystemMessage" in types
        assert "HumanMessage" in types

    def test_human_message_contains_card_name_orientation_and_position(self) -> None:
        """The human message surfaces card_name, orientation, and position_name."""
        messages = per_card_prompt.format_messages(**_full_per_card_dict())
        human_text = str(messages[1].content)
        assert _CARD_NAME in human_text
        assert _ORIENTATION in human_text
        assert _POSITION_NAME in human_text
        assert _POSITION_MEANING in human_text

    def test_human_message_renders_retrieved_sections(self) -> None:
        """Retrieved context blocks are interpolated into the human message."""
        messages = per_card_prompt.format_messages(**_full_per_card_dict())
        human_text = str(messages[1].content)
        assert _CARD_SECTIONS in human_text
        assert _POSITION_TEXT in human_text

    def test_system_message_instructs_interpretation_role(self) -> None:
        """The system message frames the model as a Tarot reading assistant."""
        messages = per_card_prompt.format_messages(**_full_per_card_dict())
        system_text = str(messages[0].content)
        assert "Tarot" in system_text

    def test_missing_required_key_raises(self) -> None:
        """Omitting a required placeholder raises :class:`KeyError` on format."""
        incomplete = _full_per_card_dict()
        incomplete.pop("card_name")
        with pytest.raises(KeyError):
            per_card_prompt.format_messages(**incomplete)


# ---------------------------------------------------------------------------
# Summary prompt
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSummaryPromptTemplate:
    """Render and structural tests for :data:`summary_prompt`."""

    _REQUIRED_KEYS: ClassVar[frozenset[str]] = frozenset(
        {"spread_name", "card_summaries", "spread_description", "synergy_block"}
    )

    def test_is_a_chat_prompt_template(self) -> None:
        """The module-level prompt is a :class:`ChatPromptTemplate` instance."""
        assert isinstance(summary_prompt, ChatPromptTemplate)

    def test_renders_both_system_and_human_messages(self) -> None:
        """Format with all placeholders yields a system + human message pair."""
        messages = summary_prompt.format_messages(**_full_summary_dict())
        assert len(messages) == 2
        types = [type(m).__name__ for m in messages]
        assert "SystemMessage" in types
        assert "HumanMessage" in types

    def test_human_message_contains_spread_name_and_summaries(self) -> None:
        """The human message includes the spread name and per-card summaries."""
        messages = summary_prompt.format_messages(**_full_summary_dict())
        human_text = str(messages[1].content)
        assert _SPREAD_NAME in human_text
        assert "Position 0" in human_text
        assert "The Fool" in human_text
        assert "Beginnings." in human_text

    def test_human_message_includes_spread_description(self) -> None:
        """The spread description block is interpolated into the human message."""
        messages = summary_prompt.format_messages(**_full_summary_dict())
        human_text = str(messages[1].content)
        assert "Past" in human_text
        assert "What has been." in human_text
        assert "Present" in human_text

    def test_system_message_instructs_summary_role(self) -> None:
        """The system message frames the model as a summary generator."""
        messages = summary_prompt.format_messages(**_full_summary_dict())
        system_text = str(messages[0].content)
        assert "summary" in system_text.lower()

    def test_missing_required_key_raises(self) -> None:
        """Omitting a required placeholder raises :class:`KeyError` on format."""
        incomplete = _full_summary_dict()
        incomplete.pop("spread_name")
        with pytest.raises(KeyError):
            summary_prompt.format_messages(**incomplete)


# ---------------------------------------------------------------------------
# Cross-prompt — runtime contract with build_per_card_context / build_summary_context
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPromptContextBuildersMatchTemplates:
    """The context builders must produce dicts that match the prompt placeholders."""

    def test_per_card_context_keys_match_prompt_placeholders(self) -> None:
        """``build_per_card_context`` keys are exactly what ``per_card_prompt`` needs."""
        # Read the input variables off the prompt template
        # (ChatPromptTemplate exposes them via .input_variables)
        prompt_vars = set(per_card_prompt.input_variables)
        # The context builder should yield these and only these
        expected = {
            "card_name",
            "orientation",
            "position_name",
            "position_meaning",
            "retrieved_card_sections",
            "retrieved_position_text",
        }
        assert prompt_vars == expected
        # And the builder's signature should match (smoke test, not exhaustive)
        sig = inspect.signature(build_per_card_context)
        # 6 required positional args (dealt, card, spread, position, vector_store, embedder)
        # + 1 required keyword-only arg (deck_id)
        required_params = [
            p for p in sig.parameters.values() if p.default is inspect.Parameter.empty
        ]
        assert len(required_params) == 7

    def test_summary_context_keys_match_prompt_placeholders(self) -> None:
        """``build_summary_context`` keys are exactly what ``summary_prompt`` needs."""
        prompt_vars = set(summary_prompt.input_variables)
        expected = {"spread_name", "card_summaries", "spread_description", "synergy_block"}
        assert prompt_vars == expected
        sig = inspect.signature(build_summary_context)
        required_params = [
            p for p in sig.parameters.values() if p.default is inspect.Parameter.empty
        ]
        # 2 required positional args: interpretations, spread
        # (synergies is optional with default None)
        assert len(required_params) == 2


# ---------------------------------------------------------------------------
# Integration: build_summary_context renders into summary_prompt
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSummaryContextEndToEnd:
    """``build_summary_context`` output must render cleanly into ``summary_prompt``."""

    def test_build_summary_context_output_renders_prompt(self) -> None:
        """A real summary context dict renders the summary prompt without error."""
        spread = Spread(
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
            ],
        )
        interpretations = [
            CardInterpretation(
                dealt=DealtCard(
                    card_id="the-fool", orientation=Orientation.UPRIGHT, position_index=0
                ),
                card_name="The Fool",
                position_name="Past",
                text="A new beginning.",
            ),
            CardInterpretation(
                dealt=DealtCard(
                    card_id="the-magus", orientation=Orientation.REVERSED, position_index=1
                ),
                card_name="The Magus",
                position_name="Present",
                text="A pause for grounding.",
            ),
        ]
        ctx = build_summary_context(interpretations, spread)
        # Round-trip into the prompt — no KeyError means the keys match
        messages = summary_prompt.format_messages(**ctx)
        assert len(messages) == 2
        human_text = str(messages[1].content)
        assert "The Fool" in human_text
        assert "The Magus" in human_text
        assert "Past" in human_text
        assert "Present" in human_text
        assert "upright" in human_text
        assert "reversed" in human_text

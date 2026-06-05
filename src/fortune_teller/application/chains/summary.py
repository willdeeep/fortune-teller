"""Reading summary chain — see plan 0008 for full specification.

Produces a final summary that surfaces reinforcing or conflicting patterns
across the per-card interpretations. Like the per-card chain, this is a
pure ``prompt | llm | StrOutputParser`` pipeline — no retrieval happens
inside the chain. The prompt is fed a pre-rendered block of card summaries
plus a description of the spread.
"""

from __future__ import annotations

from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from fortune_teller.application.models.domain import (
    CardInterpretation,
    Spread,
)

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SUMMARY_SYSTEM = """\
You are a Tarot reading assistant producing a reading summary.

Rules:
- Use ONLY the text provided in the card and position context below.
- Identify any reinforcing themes (cards sharing keywords, arcana, elements).
- Identify any conflicting or tensioning themes.
- Do not invent symbolic meaning not present in the context.
- Respond in 4 to 8 sentences.
- Do not mention the word "context" in your response.
"""

SUMMARY_HUMAN = """\
Spread: {spread_name}

{card_summaries}

Spread description:
{spread_description}

Produce a summary reading.
"""

summary_prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
    [
        ("system", SUMMARY_SYSTEM),
        ("human", SUMMARY_HUMAN),
    ]
)


# ---------------------------------------------------------------------------
# Chain factory
# ---------------------------------------------------------------------------


def build_summary_chain(llm: Runnable[Any, Any]) -> Runnable[Any, Any]:
    """Return a chain ``summary_prompt | llm | StrOutputParser()``.

    Args:
        llm: A LangChain ``Runnable`` that accepts a list of messages
             (e.g. ``ChatOpenAI``) and returns an ``AIMessage``.
    """
    return summary_prompt | llm | StrOutputParser()


# ---------------------------------------------------------------------------
# Prompt-input builder
# ---------------------------------------------------------------------------


def build_summary_context(
    interpretations: list[CardInterpretation],
    spread: Spread,
) -> dict[str, str]:
    """Build the prompt input dict from finished card interpretations.

    Args:
        interpretations: Per-card interpretations in dealt order. The
            order, not the position index on the model, determines
            numbering in the rendered summary block.
        spread: The :class:`Spread` the reading is for.

    Returns:
        A dict with keys ``spread_name``, ``card_summaries``, and
        ``spread_description`` — the placeholders in
        :data:`summary_prompt`.
    """
    card_summaries = "\n\n".join(
        f"Position {interp.dealt.position_index} — "
        f"{interp.position_name} ({interp.card_name}, {interp.dealt.orientation}):\n"
        f"{interp.text}"
        for interp in interpretations
    )
    spread_description = "\n".join(f"{pos.name}: {pos.meaning}" for pos in spread.positions)
    return {
        "spread_name": spread.name,
        "card_summaries": card_summaries,
        "spread_description": spread_description,
    }

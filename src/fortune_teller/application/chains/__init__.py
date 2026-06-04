"""LangChain chains: per-card interpretation and reading summary.

The two chains are intentionally kept independent: each is a pure
``prompt | llm | StrOutputParser`` pipeline, with retrieval done by a
helper function that lives next to its chain. This makes the chains
easy to test with a stub LLM and keeps the LangChain surface small.
"""

from fortune_teller.application.chains.per_card import (
    PER_CARD_HUMAN,
    PER_CARD_SYSTEM,
    build_chat_model,
    build_per_card_chain,
    build_per_card_context,
    per_card_prompt,
)
from fortune_teller.application.chains.summary import (
    SUMMARY_HUMAN,
    SUMMARY_SYSTEM,
    build_summary_chain,
    build_summary_context,
    summary_prompt,
)

__all__ = [
    "PER_CARD_HUMAN",
    "PER_CARD_SYSTEM",
    "SUMMARY_HUMAN",
    "SUMMARY_SYSTEM",
    "build_chat_model",
    "build_per_card_chain",
    "build_per_card_context",
    "build_summary_chain",
    "build_summary_context",
    "per_card_prompt",
    "summary_prompt",
]

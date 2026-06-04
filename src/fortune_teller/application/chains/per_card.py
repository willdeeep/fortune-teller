"""Per-card interpretation chain — see plan 0008 for full specification.

This module wires a LangChain :class:`~langchain_core.prompts.ChatPromptTemplate`
to a chat model and an output parser, and exposes a helper that retrieves
relevant context from the vector store before invoking the chain.

The chain itself is a pure ``prompt | llm | StrOutputParser`` pipeline; the
retrieval step is kept separate so the chain remains easy to unit-test with
a stub LLM (no vector store needed in those tests).
"""

from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI

from fortune_teller.application.config import settings
from fortune_teller.application.models.domain import (
    Card,
    DealtCard,
    SearchHit,
    Spread,
    SpreadPosition,
)
from fortune_teller.application.stores.embeddings import Embedder
from fortune_teller.application.stores.vector import VectorStore

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

PER_CARD_SYSTEM = """\
You are a Tarot reading assistant interpreting a single card.

Rules:
- Use ONLY the information in the provided context sections.
- Do not add meaning beyond what is in the context.
- If the card is reversed, weight the REVERSED section and shadow themes.
- Relate the card meaning to its position meaning.
- Respond in 3 to 5 sentences.
- Do not mention the word "context" in your response.
"""

PER_CARD_HUMAN = """\
Card: {card_name} ({orientation})
Position: {position_name} — {position_meaning}

Card context sections:
{retrieved_card_sections}

Position context:
{retrieved_position_text}

Provide a brief interpretation of this card in this position.
"""

per_card_prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
    [
        ("system", PER_CARD_SYSTEM),
        ("human", PER_CARD_HUMAN),
    ]
)


# ---------------------------------------------------------------------------
# Chat model factory
# ---------------------------------------------------------------------------


def build_chat_model() -> ChatOpenAI:
    """Return a :class:`ChatOpenAI` configured for the local llama.cpp server.

    Settings are pulled from :class:`~fortune_teller.application.config.Settings`
    (env-driven). The model is initialised lazily by the caller — the first
    ``.invoke()`` triggers the HTTP connection.
    """
    return ChatOpenAI(
        base_url=settings.openai_base_url,
        api_key=settings.openai_api_key,
        model=settings.chat_model,
        temperature=0.0,
        timeout=60,
        max_retries=2,
    )


# ---------------------------------------------------------------------------
# Chain factory
# ---------------------------------------------------------------------------


def build_per_card_chain(llm: Runnable) -> Runnable:
    """Return a chain ``per_card_prompt | llm | StrOutputParser()``.

    Args:
        llm: A LangChain ``Runnable`` that accepts a list of messages
             (e.g. ``ChatOpenAI``) and returns an ``AIMessage``. Any
             compatible runnable works — pass a stub in tests.
    """
    return per_card_prompt | llm | StrOutputParser()


# ---------------------------------------------------------------------------
# Retrieval helper
# ---------------------------------------------------------------------------


def _format_chunks(hits: list[SearchHit], *, empty_marker: str = "(none retrieved)") -> str:
    """Format a list of :class:`SearchHit` as a bulleted block.

    Returns *empty_marker* when the list is empty, so the prompt always has
    something to render in the placeholder.
    """
    if not hits:
        return empty_marker
    return "\n".join(f"- {hit.chunk.text}" for hit in hits)


def build_per_card_context(
    dealt: DealtCard,
    card: Card,
    spread: Spread,
    position: SpreadPosition,
    vector_store: VectorStore,
    embedder: Embedder,
    *,
    k: int = 4,
) -> dict[str, str]:
    """Retrieve per-card and per-position context, then build a prompt dict.

    The returned dict has the keys expected by :data:`per_card_prompt`:
    ``card_name``, ``orientation``, ``position_name``, ``position_meaning``,
    ``retrieved_card_sections``, ``retrieved_position_text``.

    Args:
        dealt:           The card as dealt (carries orientation + position index).
        card:            The full :class:`Card` (provides name, id, etc.).
        spread:          The parent :class:`Spread` (needed for ``spread_id``).
        position:        The :class:`SpreadPosition` for this deal.
        vector_store:    An open :class:`VectorStore` for retrieval.
        embedder:        An :class:`Embedder` used to embed the card query.
        k:               Number of card-section chunks to retrieve.

    Returns:
        A dict suitable for ``per_card_chain.invoke(...)``.
    """
    card_query = f"{card.name} {dealt.orientation.value}"
    card_embedding = embedder.embed_query(card_query)
    card_hits = vector_store.search_card_section(
        card_id=card.id,
        query_embedding=card_embedding,
        k=k,
    )
    position_hits = vector_store.search_spread_position(
        spread_id=spread.id,
        position_index=position.index,
    )
    return {
        "card_name": card.name,
        "orientation": dealt.orientation.value,
        "position_name": position.name,
        "position_meaning": position.meaning,
        "retrieved_card_sections": _format_chunks(card_hits),
        "retrieved_position_text": _format_chunks(position_hits),
    }

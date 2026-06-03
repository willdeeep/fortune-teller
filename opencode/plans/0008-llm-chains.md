# 0008 — LangChain Chains

Modules:
- `fortune_teller.application.chains.per_card`
- `fortune_teller.application.chains.summary`
- `fortune_teller.application.config` — settings (env vars, defaults)

---

## Chat Model Initialisation

```python
from langchain_openai import ChatOpenAI
from fortune_teller.application.config import settings


def build_chat_model() -> ChatOpenAI:
    return ChatOpenAI(
        base_url=settings.openai_base_url,    # default: http://127.0.0.1:8080/v1
        api_key=settings.openai_api_key,       # default: "sk-no-key"
        model=settings.chat_model,             # configurable via env CHAT_MODEL
        temperature=0.0,
        timeout=60,
        max_retries=2,
    )
```

### `config.py` settings

```python
from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    openai_base_url: str = "http://127.0.0.1:8080/v1"
    openai_api_key: str = "sk-no-key"
    chat_model: str = "local-model"
    ft_data_dir: Path = Path("./data")
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    class Config:
        env_prefix = ""
        env_file = ".env"


settings = Settings()
```

---

## Per-Card Chain

### Purpose

For each dealt card, retrieve the top-k card section chunks and the spread
position chunk, then produce a grounded 3–5 sentence interpretation.

### Prompt Template

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


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

per_card_prompt = ChatPromptTemplate.from_messages([
    ("system", PER_CARD_SYSTEM),
    ("human", PER_CARD_HUMAN),
])


def build_per_card_chain(llm: ChatOpenAI) -> ...:
    return per_card_prompt | llm | StrOutputParser()
```

### Retrieval helper (called before chain)

```python
def build_per_card_context(
    dealt: DealtCard,
    card: Card,
    position: SpreadPosition,
    vector_store: VectorStore,
) -> dict[str, str]:
    card_chunks = vector_store.search_card_section(
        query=f"{card.name} {dealt.orientation}",
        card_id=card.id,
        k=4,
    )
    position_chunks = vector_store.search_spread_position(
        query=position.meaning,
        spread_id=...,
        position_index=position.index,
    )
    return {
        "card_name": card.name,
        "orientation": dealt.orientation.value,
        "position_name": position.name,
        "position_meaning": position.meaning,
        "retrieved_card_sections": _format_chunks(card_chunks),
        "retrieved_position_text": _format_chunks(position_chunks),
    }
```

---

## Summary Chain

### Purpose

After all three cards are dealt and interpreted, generate a summary that
surfaces reinforcing or conflicting patterns visible in the retrieved text.
The model does NOT invent meaning — it only calls out patterns present in the
retrieved sections.

### Prompt Template

```python
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

summary_prompt = ChatPromptTemplate.from_messages([
    ("system", SUMMARY_SYSTEM),
    ("human", SUMMARY_HUMAN),
])


def build_summary_chain(llm: ChatOpenAI) -> ...:
    return summary_prompt | llm | StrOutputParser()
```

### Input builder

```python
def build_summary_context(
    dealt_cards: list[DealtCard],
    interpretations: list[CardInterpretation],
    spread: Spread,
    vector_store: VectorStore,
) -> dict[str, str]:
    card_summaries = "\n\n".join(
        f"Position {i+1} — {interp.position_name} ({interp.card_name}, {interp.dealt.orientation}):\n"
        f"{interp.text}"
        for i, interp in enumerate(interpretations)
    )
    spread_description = "\n".join(
        f"{pos.name}: {pos.meaning}" for pos in spread.positions
    )
    return {
        "spread_name": spread.name,
        "card_summaries": card_summaries,
        "spread_description": spread_description,
    }
```

---

## Unit Tests

### Prompt rendering

```python
@pytest.mark.unit
def test_per_card_prompt_renders(sample_per_card_context: dict) -> None:
    messages = per_card_prompt.format_messages(**sample_per_card_context)
    assert len(messages) == 2
    assert "The Fool" in messages[1].content
    assert "reversed" in messages[1].content.lower()


@pytest.mark.unit
def test_summary_prompt_renders(sample_summary_context: dict) -> None:
    messages = summary_prompt.format_messages(**sample_summary_context)
    assert "New Moon" in messages[1].content
```

### Chain integration (stubbed LLM)

```python
from langchain_core.runnables import RunnableLambda

@pytest.mark.integration
def test_per_card_chain_returns_string(sample_per_card_context: dict) -> None:
    stub_llm = RunnableLambda(lambda _: FakeAIMessage(content="A test interpretation."))
    chain = per_card_prompt | stub_llm | StrOutputParser()
    result = chain.invoke(sample_per_card_context)
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.integration
def test_summary_chain_returns_string(sample_summary_context: dict) -> None:
    stub_llm = RunnableLambda(lambda _: FakeAIMessage(content="A test summary."))
    chain = summary_prompt | stub_llm | StrOutputParser()
    result = chain.invoke(sample_summary_context)
    assert isinstance(result, str)
```

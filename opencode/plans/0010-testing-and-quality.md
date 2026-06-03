# 0010 — Testing & Quality

---

## Philosophy

- Tests are first-class code: same ruff + mypy standards as `src/`.
- No magic numbers in assertions — use named constants or parametrize.
- No live HTTP, no live LLM, no writes to `data/` in tests.
- Fixtures are the documentation of the data shapes.

---

## Test Layout

```
tests/
├── conftest.py                     # shared fixtures: deck, spread, chunks, stubs
├── unit/
│   ├── test_domain_models.py       # pydantic validators, enums, round-trips
│   ├── test_deck_session.py        # DeckSession: no-replace, exhaustion, reset
│   ├── test_parser.py              # HTML → Card / Spread parsing (fixture HTML)
│   └── test_prompt_templates.py   # prompt rendering from known inputs
├── integration/
│   ├── test_vector_store.py        # DuckDB VSS: insert → search round-trip
│   └── test_reading_chain.py       # per-card + summary chains with stub LLM
└── fixtures/
    ├── html/
    │   └── thothreadings/
    │       ├── the-fool.html
    │       ├── the-magus.html
    │       ├── ace-of-wands.html
    │       └── spread-new-moon.html
    └── parsed/
        ├── book-of-thoth/
        │   └── the-fool.json
        └── spreads/
            └── new-moon-three-card.json
```

---

## Pytest Markers

```
unit        — pure logic, no I/O; must finish in <5s total; run in pre-commit
integration — DuckDB, chain stubs; may take longer; run in CI only
slow        — anything >5s individually; excluded from pre-commit and fast CI
```

Usage:

```python
@pytest.mark.unit
def test_deck_is_shuffled() -> None: ...

@pytest.mark.integration
def test_vector_store_round_trip(tmp_path) -> None: ...

@pytest.mark.slow
@pytest.mark.integration
def test_full_pipeline_end_to_end(tmp_path) -> None: ...
```

---

## Coverage Gate

- `--cov-fail-under=80` in `pyproject.toml` `[tool.pytest.ini_options]`.
- Branch coverage enabled.
- CI fails if coverage drops below 80%.
- Excluded from coverage: `pragma: no cover`, `if TYPE_CHECKING:`,
  `raise NotImplementedError`, Gradio `main()` entry points.

---

## Key `conftest.py` Fixtures

```python
import pytest
from fortune_teller.application.models.domain import Card, Deck, Spread, ...

@pytest.fixture
def small_deck() -> Deck:
    """A 10-card deck for fast property tests."""
    ...

@pytest.fixture
def full_78_card_deck() -> Deck:
    """Full deck loaded from fixture JSON files."""
    ...

@pytest.fixture
def new_moon_spread() -> Spread:
    """Loaded from tests/fixtures/parsed/spreads/new-moon-three-card.json"""
    ...

@pytest.fixture
def the_fool_html() -> str:
    """Raw HTML from tests/fixtures/html/thothreadings/the-fool.html"""
    ...

@pytest.fixture
def sample_chunks(small_deck, new_moon_spread) -> list[Chunk]:
    """Pre-built (not yet embedded) chunks for store tests."""
    ...

@pytest.fixture
def stub_llm():
    """RunnableLambda that returns a canned string without network."""
    from langchain_core.runnables import RunnableLambda
    from langchain_core.messages import AIMessage
    return RunnableLambda(lambda _: AIMessage(content="Test interpretation."))
```

---

## Per-Module Test Targets

### `test_domain_models.py`
- `Card` minor arcana without suit → `ValidationError`
- `Card` major arcana with suit → `ValidationError`
- `Spread` non-contiguous indices → `ValidationError`
- `Reading` duplicate card IDs → `ValidationError`
- All `StrEnum` values serialise and round-trip through `model_dump_json`
- `Deck.card_by_id` happy path and `KeyError`
- `CardSection` has exactly 11 members

### `test_deck_session.py`
- Dealt cards are pairwise distinct (Hypothesis property test, 78-card deck)
- Deal beyond deck size raises `DeckExhausted`
- `reset()` restores full deck and re-shuffles
- Inversion distribution is ~50/50 over 10k deals (chi-square)
- `deal_spread(3)` returns 3 `DealtCard`s with positions 0, 1, 2
- Two sessions with same seed produce same order
- `remaining_count` decrements correctly

### `test_parser.py`
- `parse_card_page(the_fool_html)` produces a `Card` with all 11 sections
- Card name and arcana are correct for The Fool, The Magus, Ace of Wands
- Section text is non-empty for all parsed sections
- `parse_spread_page(new_moon_html)` produces a `Spread` with 3 positions
  and contiguous indices
- Cached HTML path is used when file exists (no HTTP call)

### `test_prompt_templates.py`
- Per-card prompt renders both system and human messages correctly
- Human message contains card name, orientation, position name
- Summary prompt renders spread name and card summaries
- Missing required variable raises `KeyError`

### `test_vector_store.py`
- Insert 20 chunks → search by card_id returns only that card's chunks
- Top result is the most semantically relevant section
- Spread position search returns the correct position index
- Persistence: write + close + reopen → results still there

### `test_reading_chain.py`
- Per-card chain with stub LLM returns a non-empty string
- Summary chain with stub LLM returns a non-empty string
- Chain does not call the real LLM endpoint (no network in CI)

---

## Ruff & Mypy Quality

Both are enforced at three levels:
1. **Pre-commit**: runs on every `git commit`
2. **CI**: fails the build if any violation
3. **Manual**: `uv run ruff check && uv run mypy src`

Acceptable suppressions:
- `# noqa: <code>` — must have an inline comment explaining why
- `# type: ignore[<error-code>]` — must have an inline comment
- `ignore_missing_imports` overrides in `pyproject.toml` for untyped
  third-party stubs (sentence-transformers, duckdb, gradio, langchain)

# 0002 — Documentation

## Files to Create

| File | Purpose |
| ---- | ------- |
| `README.md` | Overview, quickstart, architecture, roadmap |
| `AGENTS.md` | Agent/AI coding conventions for this repo |
| `LICENSE` | MIT licence |
| `CONTRIBUTING.md` | Lightweight contributor guide |
| `docs/architecture.md` | Data-flow diagram, component responsibilities |
| `docs/decks/book-of-thoth.md` | Card schema, field definitions, source attribution |

---

## `README.md` — Section Outline

```
# Fortune Teller

> Local-first Tarot reading app powered by RAG, local embeddings, and a
> llama.cpp chat model. Runs fully offline on Apple Silicon (M2+).

## Features (spike)
- Book of Thoth deck (78 cards)
- New Moon three-card spread
- Auto-deal with no-replace guarantee within a reading
- Per-card interpretations grounded in scraped definitions
- Reading summary that surfaces cross-card patterns
- Runs entirely on-device — no external API keys required

## Requirements
- macOS 14+ (Apple Silicon recommended; Intel supported)
- Python 3.13 (managed by uv)
- uv — https://docs.astral.sh/uv/
- llama.cpp server running on http://127.0.0.1:8080

## Quickstart
  1. uv sync --extra dev --group test --group lint
  2. uv run pre-commit install
  3. uv run ft-scrape     # first run only — populates data/cache/
  4. uv run ft-parse      # first run only — writes data/parsed/
  5. uv run ft-embed      # first run only — writes embeddings
  6. uv run ft-build-index # first run only — writes data/duckdb/fortune.duckdb
  7. uv run fortune-teller  # opens http://127.0.0.1:7860

## llama.cpp Setup (M2)
  - Recommended model: a 4-bit quantised 7B or 8B instruct model
    (e.g. llama-3.2-8b-instruct.Q4_K_M.gguf)
  - Launch command:
      llama-server -m <model.gguf> --host 127.0.0.1 --port 8080 -ngl 99
  - Embeddings: handled locally by sentence-transformers
    (BAAI/bge-small-en-v1.5); llama.cpp server is only needed for chat.

## Configuration
  All runtime settings can be overridden with environment variables:
  - OPENAI_BASE_URL   — chat model server (default: http://127.0.0.1:8080/v1)
  - OPENAI_API_KEY    — any non-empty string (default: sk-no-key)
  - CHAT_MODEL        — model name as reported by the server
  - FT_DATA_DIR       — path to data/ directory (default: ./data)

## Architecture
  [see docs/architecture.md]

## Roadmap
  - v0.0.1-spike: single deck, single spread, auto-deal, no auth
  - v0.1.0: reading history persistence, SQLite storage
  - v0.2.0: user login (local, hashed password)
  - v0.3.0: multiple decks, multiple spreads
  - v0.4.0: manual card entry mode

## License
MIT — see LICENSE
```

---

## `AGENTS.md` — Section Outline

```
# AGENTS.md — Conventions for AI Coding Agents

## Package Layout
- All runtime code lives under src/fortune_teller/application/
- All developer tooling (scraping, index building) under src/fortune_teller/developer/
- Tests live under tests/ — never under src/
- Do NOT add new top-level packages; work within the existing tree.

## Package Manager
- Use uv ONLY. Never call pip, conda, or poetry directly.
- Add runtime deps:   uv add <package>
- Add dev CLIs:       uv add --optional dev <package>
- Add test tooling:   uv add --group test <package>
- Add lint tooling:   uv add --group lint <package>

## Code Standards
- ruff is the source of truth for formatting and linting. Never suppress
  ruff rules without a comment explaining why.
- mypy strict mode is required for all code under src/fortune_teller/.
  Add stubs or ignore_missing_imports overrides in pyproject.toml as needed.
- All public functions and classes must have docstrings.
- Type annotations required on every function signature.

## Test Rules
- Every new module must have a corresponding test module.
- Coverage gate is 80%. Do not merge if coverage drops below this.
- Mark all new tests with at least one marker: @pytest.mark.unit,
  @pytest.mark.integration, or @pytest.mark.slow.
- No live HTTP calls in tests. Use httpx.MockTransport or pytest-mock.
- No live LLM calls in tests. Stub ChatOpenAI with RunnableLambda.
- No writes to data/ in tests. Use pytest's tmp_path fixture.

## Forbidden Actions
- Do not commit anything under data/
- Do not bypass pre-commit hooks with --no-verify
- Do not add login/auth UI in the spike (tracked in roadmap v0.2.0)
- Do not hardcode API keys, paths, or model names — use config.py / env vars

## Adding a New Deck
1. Add slug list to src/fortune_teller/developer/scrape/seeds/<deck_id>.txt
2. Write a site-specific scraper in developer/scrape/<deck_id>.py
3. Write a section parser in developer/parse/<deck_id>.py
4. Add fixture HTML for ≥3 cards under tests/fixtures/html/<deck_id>/
5. Add golden parsed JSON under tests/fixtures/parsed/<deck_id>/
6. Add parser unit tests
7. Run ft-scrape, ft-parse, ft-embed, ft-build-index
8. Update docs/decks/<deck_id>.md with schema and attribution

## Adding a New Spread
1. Add a SpreadPosition list to application/models/domain.py (or a seed file)
2. Write parser logic in developer/parse/ if scraped from web
3. Add spread fixture and unit test
4. Wire into ReadingService
5. Add to Gradio UI selector (future work until multi-spread UI is built)

## Local LLM
- Chat model is accessed via ChatOpenAI with base_url pointing at llama.cpp.
- Default settings are in application/config.py, overridden by env vars.
- Embedding model (BAAI/bge-small-en-v1.5) runs locally via
  langchain-huggingface; no server required for embeddings.
```

---

## `docs/architecture.md` — Section Outline

### Data Flow (build-time)

```
thothreadings.com
      │  httpx (rate-limited, cached)
      ▼
data/cache/<host>/<slug>.html
      │  selectolax / bs4 parser
      ▼
data/parsed/<deck_id>/<slug>.json   (Card pydantic model JSON)
      │  HuggingFaceEmbeddings (bge-small-en-v1.5, MPS)
      ▼
data/duckdb/fortune.duckdb
  └── chunks table (id, metadata, text, embedding FLOAT[384])
       └── HNSW index (cosine)
```

### Data Flow (runtime)

```
Gradio UI
  │  "New Reading" click
  ▼
ReadingService.start(deck_id, spread_id)
  │  DeckSession.deal_one()  × 3  (no-replace, random orientation)
  ▼
for each DealtCard:
  VectorStore.search_card_section(query, card_id=…, k=3)
  VectorStore.search_spread_position(query, spread_id=…, position_index=…)
  │  PerCardChain (ChatPromptTemplate + ChatOpenAI @ temperature=0)
  ▼
  CardInterpretation.text → displayed in Gradio card panel

after all 3 cards:
  SummaryChain (all retrieved chunks bundled)
  │  ChatOpenAI @ temperature=0
  ▼
  Reading.summary → displayed in Gradio summary panel
```

---

## `docs/decks/book-of-thoth.md` — Section Outline

```
# Book of Thoth — Deck Schema

Source: https://thothreadings.com
Attribution: Content scraped from thothreadings.com under fair use for
             personal/educational purposes. Do not redistribute.

## Card Count
78 cards (22 Major Arcana + 56 Minor Arcana in suits: Wands, Cups,
Swords, Disks)

## Structured Fields per Card

| Field        | Section key    | Description |
| ------------ | -------------- | ----------- |
| Overall      | overall        | General meaning of the card |
| Drive        | drive          | The card's motivating energy |
| Light        | light          | Positive expression / upright themes |
| Shadow       | shadow         | Challenge / shadow expression |
| Reversed     | reversed       | Meaning when card is inverted |
| Keywords     | keywords       | Short thematic words |
| Advice       | advice         | Practical guidance from the card |
| Question     | question       | Reflective question the card poses |
| Proposal     | proposal       | What the card proposes or suggests |
| Confirmation | confirmation   | What the card confirms |
| Affirmation  | affirmation    | Positive affirmation statement |

## Embedding Strategy
Each field becomes one vector chunk with metadata:
  { deck_id, card_id, card_name, section, source_url }
Total: ~78 cards × ~11 sections ≈ ~858 chunks

## Card ID Format
Slugified card name, e.g.:
  "The Fool"        → "the-fool"
  "Ace of Wands"    → "ace-of-wands"
  "Queen of Cups"   → "queen-of-cups"
  "The High Priestess" → "the-high-priestess"
```

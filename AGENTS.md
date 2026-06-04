# AGENTS.md — Conventions for AI Coding Agents

This file defines conventions and rules for AI coding agents (and human
contributors) working in this repository.

## Setup & Environment

- **Python**: exactly `>=3.13,<3.14` (`.python-version` is `3.13`).
- **Package manager**: `uv` only. Never call `pip`, `conda`, or `poetry`.
- **Full dev install** (required before any work):
  ```bash
  uv sync --extra dev --group test --group lint
  uv run pre-commit install
  ```
- **llama.cpp server** must be running at `http://127.0.0.1:8080` for the app
  to work. Embeddings run locally via `sentence-transformers` and do **not**
  need a server.

## Package Layout

- Runtime code → `src/fortune_teller/application/`
- Developer tooling (scraping, parsing, embedding, index building) →
  `src/fortune_teller/developer/`
- Tests → `tests/` (subdirs: `tests/unit/`, `tests/integration/`). Never under `src/`.
- Do NOT add new top-level packages; work within the existing tree.

## Code Standards

- `ruff` is the source of truth for formatting, linting, and import sorting.
  Line length: 100. Never suppress rules without an inline comment explaining why.
- `mypy` strict mode for all code under `src/fortune_teller/` (not `tests/`).
  Add stubs or `ignore_missing_imports` overrides in `pyproject.toml`.
- All public functions and classes **must** have docstrings.
- Type annotations required on every function signature.
- All pydantic domain models are **frozen** (`ConfigDict(frozen=True)`). Use
  `model.model_copy(update={...})` for modified copies, never mutation.

## Verification Pipeline

Run in this order (matches CI):

```bash
uv run ruff check .                   # lint
uv run ruff format --check .          # formatting (or omit --check to fix)
uv run mypy src                       # type-check (src only)
uv run pytest                         # tests + coverage gate (≥80% branch)
```

Fast pre-commit subset (unit + not slow, no coverage gate):
```bash
uv run pytest -m "unit and not slow" -q --no-header --no-cov
```

## Test Rules

- Every new module needs a corresponding test module under `tests/`.
- Coverage gate is **80%** branch coverage (enforced via `pyproject.toml`
  `addopts` — just running `uv run pytest` checks it).
- Mark all tests: `@pytest.mark.unit`, `@pytest.mark.integration`,
  `@pytest.mark.slow` (>5s, excluded from pre-commit).
- **No live HTTP calls** — use `httpx.MockTransport` or `pytest-mock`.
- **No live LLM calls** — use the `stub_llm` fixture from `tests/conftest.py`.
- **No writes to `data/`** — use pytest's `tmp_path` fixture.
- **`.env` file auto-loads** via pydantic-settings — beware of stale `.env`
  values leaking into tests.

## Forbidden Actions

- Do not commit anything under `data/` (it's in `.gitignore`).
- Do not bypass pre-commit hooks with `--no-verify`.
- Do not add login/auth UI (roadmap v0.2.0).
- Do not hardcode API keys, paths, or model names — use
  `application/config.py` / env vars.
- Do not call `pip install` — always use `uv`.

## Configuration

All settings live in `src/fortune_teller/application/config.py` using
`pydantic-settings`. Override via environment variables or a `.env` file in the
project root. Key variables:

| Variable          | Default                      |
| ----------------- | ---------------------------- |
| `OPENAI_BASE_URL` | `http://127.0.0.1:8080/v1`   |
| `OPENAI_API_KEY`  | `sk-no-key`                  |
| `CHAT_MODEL`      | `local-model`                |
| `FT_DATA_DIR`     | `./data`                     |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5`     |

Access via the singleton: `from fortune_teller.application.config import settings`.

## Architecture Notes

- **Vector store is DuckDB VSS** (HNSW index, cosine metric, 384-dim vectors) —
  not FAISS or Chroma.
- Chat model accessed via `ChatOpenAI` with `base_url` pointing at llama.cpp.
  Embedding model runs locally via `langchain-huggingface` — no embedding server.
- Data pipeline is sequential and order-sensitive:
  `ft-scrape` → `ft-parse` → `ft-embed` → `ft-build-index`
- See `docs/architecture.md` for full data flow and design rationale.

## Branch & Commit Convention

- Branch names: `feat/**`, `fix/**`, `chore/**`, `docs/**` (matches CI triggers).
- Conventional Commits: `feat:`, `fix:`, `test:`, `docs:`, `chore:`, `refactor:`.

## Adding a New Deck

1. Add slug list to `src/fortune_teller/developer/scrape/seeds/<deck_id>.txt`
2. Write a site-specific scraper in `developer/scrape/<deck_id>.py`
3. Write a section parser in `developer/parse/<deck_id>.py`
4. Add fixture HTML for ≥3 cards under `tests/fixtures/html/<deck_id>/`
5. Add golden parsed JSON under `tests/fixtures/parsed/<deck_id>/`
6. Add parser unit tests
7. Run `ft-scrape`, `ft-parse`, `ft-embed`, `ft-build-index`
8. Update `docs/decks/<deck_id>.md` with schema and attribution

## Adding a New Spread

1. Add a `SpreadPosition` list in the models or a seed file
2. Write parser logic in `developer/parse/` if scraped from the web
3. Add spread fixture JSON and a unit test
4. Wire into `ReadingService`
5. Add to Gradio UI selector (deferred until multi-spread UI is built)

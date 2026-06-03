# AGENTS.md — Conventions for AI Coding Agents

This file defines conventions and rules for AI coding agents (and human
contributors) working in this repository.

## Package Layout

- All runtime code lives under `src/fortune_teller/application/`
- All developer tooling (scraping, index building) under `src/fortune_teller/developer/`
- Tests live under `tests/` — **never** under `src/`
- Do NOT add new top-level packages; work within the existing tree

## Package Manager

Use `uv` **only**. Never call `pip`, `conda`, or `poetry` directly.

```bash
uv add <package>                         # add runtime dependency
uv add --optional dev <package>          # add developer extra
uv add --group test <package>            # add test tooling
uv add --group lint <package>            # add lint tooling
uv sync --extra dev --group test --group lint  # full dev env
```

## Code Standards

- `ruff` is the source of truth for formatting, linting, and import sorting.
  Never suppress ruff rules without an inline comment explaining why.
- `mypy` strict mode is required for all code under `src/fortune_teller/`.
  Add stubs or `ignore_missing_imports` overrides in `pyproject.toml`.
- All public functions and classes **must** have docstrings.
- Type annotations required on every function signature.
- Line length: 100 characters.

## Test Rules

- Every new module must have a corresponding test module.
- Coverage gate is **80%** (branch coverage). Do not merge if coverage drops.
- Mark all new tests with at least one marker:
  - `@pytest.mark.unit` — fast, pure logic, no I/O (runs in pre-commit)
  - `@pytest.mark.integration` — touches DuckDB/SQLite/HTTP fixtures
  - `@pytest.mark.slow` — >5s; excluded from pre-commit and fast CI matrix
- **No live HTTP calls in tests** — use `httpx.MockTransport` or `pytest-mock`.
- **No live LLM calls in tests** — stub `ChatOpenAI` with `RunnableLambda`.
- **No writes to `data/`** — use pytest's `tmp_path` fixture.

## Forbidden Actions

- Do not commit anything under `data/`
- Do not bypass pre-commit hooks with `--no-verify`
- Do not add login/auth UI in the spike (roadmap v0.2.0)
- Do not hardcode API keys, paths, or model names — use `config.py` / env vars
- Do not call `pip install` directly — always use `uv`

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

## Local LLM

- Chat model is accessed via `ChatOpenAI` with `base_url` pointing at llama.cpp.
- Default settings are in `application/config.py`, overridden by env vars.
- Embedding model (`BAAI/bge-small-en-v1.5`) runs locally via
  `langchain-huggingface`; no server required for embeddings.

## Commit Convention

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add DeckSession no-replace logic
fix: handle missing card section in parser
test: add property tests for deck exhaustion
docs: update architecture diagram
chore: pin ruff to latest in pre-commit
refactor: extract chunk builder from vector store
```

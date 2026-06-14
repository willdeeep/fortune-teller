# AGENTS.md

## Setup

```bash
uv sync --extra dev --group test --group lint
uv run pre-commit install
```

Python 3.13 only (`>=3.13,<3.14`). Package manager is `uv` — never pip/poetry/conda.
A llama.cpp server at `http://127.0.0.1:8080` is required at runtime; embeddings run locally.

## Verification (run in this order)

```bash
uv run ruff check .                # lint
uv run ruff format --check .        # format (omit --check to fix)
uv run mypy src                     # type-check (src only, not tests)
uv run pytest                       # full suite + 80% branch coverage gate
```

Fast pre-commit subset:

```bash
uv run pytest -m "unit and not slow" -q --no-header --no-cov
```

## Package layout

- `src/fortune_teller/application/` — runtime (UI, chains, services, models, stores)
- `src/fortune_teller/developer/` — offline tooling (scrape, parse, embed, fetch-images, fetch-models, build-index)
- `tests/unit/`, `tests/integration/` — never under `src/`
- No new top-level packages

## Key gotchas

- **Pydantic models are frozen** (`ConfigDict(frozen=True)`). Use `model.model_copy(update={...})`, never mutate.
- **Vector store is DuckDB VSS** (HNSW, cosine, 384-dim) — not FAISS or Chroma.
- **ChatOpenAI timeout is 180s** (not the default 60s) — CPU-only llama.cpp can take 120s on summary prompts.
- **Images are deck-scoped**: `settings.images_dir / deck_id / card_id.<ext>`. Always pass `images_dir / service.deck_id`, never bare `images_dir`.
- **Gradio `allowed_paths`** must include the deck image directory, or Gradio refuses to serve images.
- **Typer CLI entry points** reference the Typer `app` object (e.g. `cli:app`), not a bare `main()` function — calling the function directly skips click argument parsing.
- **Lazy imports** in `services/reading.py` and `ui/app.py` are intentional (`# noqa: PLC0415`) — they defer config and heavy service loads so test patches work correctly.
- **`data/` is gitignored** except `data/models/` (has `!data/models/` exception for the offline embedding model).
- **`.env` auto-loads** via pydantic-settings; stale values leak into tests. Clean it if tests behave strangely.
- **Data pipeline is order-sensitive**: `ft-scrape` → `ft-parse` → `ft-embed` → `ft-build-index`
- Config singleton: `from fortune_teller.application.config import settings`

## Test conventions

- Mark all tests: `@pytest.mark.unit`, `@pytest.mark.integration`, `@pytest.mark.slow` (>5s)
- **No live HTTP** — use `httpx.MockTransport` or `pytest-mock`
- **No live LLM** — use `stub_llm` / `stub_embedder` from `tests/conftest.py`
- **No writes to `data/`** — use `tmp_path`
- Every new module needs a test module under `tests/`
- Coverage gate is 80% branch (enforced by `pyproject.toml` addopts)

## Code standards

- `ruff` for format + lint + import sort. Line length 100. Never suppress rules without an inline comment explaining why.
- `mypy` strict for `src/fortune_teller/` (pydantic plugin enabled, `ignore_missing_imports` in `pyproject.toml` for third-party stubs)
- Google-style docstrings on public modules, classes, functions
- Full type hints on every signature (`X | None`, not `Optional`)

## Branch & release flow

Three long-lived branches: `dev` → `staging` → `main`.

- Working branches: `feature/<slug>`, `bugfix/<slug>`, `refactor/<slug>`, `docs/<slug>` (off `dev`)
- Hotfix branches off `main`
- Conventional Commits: `feat:`, `fix:`, `test:`, `docs:`, `chore:`, `refactor:`
- Promotion uses the staged-release-flow skill (see global rules) — do not push directly to `staging` or `main`

## Forbidden

- Do not commit anything under `data/` (gitignored)
- Do not bypass pre-commit hooks with `--no-verify`
- Do not hardcode API keys, paths, or model names — use `settings` / env vars
- Do not add login/auth UI
- Do not call `pip install`

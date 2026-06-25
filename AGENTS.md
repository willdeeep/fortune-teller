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
- `src/fortune_teller/developer/` — offline tooling (scrape, parse, normalize, embed, fetch-images, fetch-models, build-index)
- `tests/unit/`, `tests/integration/` — never under `src/`
- No new top-level packages

## Data pipeline (order matters)

```bash
ft-fetch-models      # download embedding model (one-time)
ft-scrape            # scrape all decks (--source thoth|learntarot|all, default all)
ft-parse             # parse all decks (--source thoth|learntarot|all, default all)
ft-normalize-rw      # Rider-Waite only: RawCard → Card via LLM (--no-llm for deterministic dry run)
ft-normalize-thoth   # Thoth only: synthesize reinforce/oppose IDs via LLM (--no-llm for dry run)
ft-normalize         # umbrella: runs both RW + Thoth normalizers (--deck rw|thoth|all, default all)
ft-embed             # embed all decks (skips meta.json)
ft-build-index       # build DuckDB vector index (all decks)
ft-fetch-images      # download card artwork (--deck book-of-thoth|rider-waite|all, default all)
```

`ft-scrape`/`ft-parse` use `--source`; `ft-fetch-images` uses `--deck`; `ft-normalize` uses `--deck` — different flags for the same concept.

## Key gotchas

- **Pydantic models are frozen** (`ConfigDict(frozen=True)`). Use `model.model_copy(update={...})`, never mutate.
- **Vector store is DuckDB VSS** (HNSW, cosine, 384-dim) — not FAISS or Chroma.
- **`search_card_section` requires `deck_id`** — the SQL filters `AND deck_id = ?`; passing the wrong deck returns nothing.
- **ChatOpenAI timeout is 180s** (not the default 60s) — CPU-only llama.cpp can take 120s on summary prompts.
- **Images are deck-scoped**: `settings.images_dir / deck_id / card_id.<ext>`. Always pass `images_dir / service.deck_id`, never bare `images_dir`.
- **NiceGUI serves card images** via `app.add_static_files("/images", <deck image dir>)` in `build_app`; resolve URLs as `/images/<file>`, never filesystem paths in the browser.
- **Typer CLI entry points** reference the Typer `app` object (e.g. `cli:app`), not a bare `main()` function — calling the function directly skips click argument parsing.
- **Lazy imports** in `services/reading.py` and `ui/nicegui_app.py` are intentional (`# noqa: PLC0415`) — they defer config and heavy service loads so test patches work correctly.
- **`langchain-anthropic` is a dev/optional dependency** — imported lazily inside functions in `developer/normalize/`, not at module level. The app runs without it; only `ft-normalize-rw` needs it.
- **`Runnable[Any, Any]` not bare `Runnable`** — `langchain_core.runnables.Runnable` is generic; bare `Runnable` triggers mypy `type-arg` error.
- **`data/` is gitignored** except `data/models/` (has `!data/models/` exception for the offline embedding model).
- **`.env` auto-loads** via pydantic-settings; stale values leak into tests. Clean it if tests behave strangely.
- **`Suit.PENTACLES`** is for Rider-Waite; Thoth uses `Suit.DISKS`. Both are valid enum values; minor-arcana cards must carry the suit matching their deck.
- **`meta.json`** in each deck directory (`data/parsed/<deck_id>/meta.json`) carries `name`, `source_url`, `attribution`, `description`. `load_deck` reads it; `list_decks()` uses it for display names. Card JSON files are loaded alongside it (excluded by glob).
- **Normalization provenance** is stored in sidecar files (`data/parsed/rider-waite/_norm/<id>.json`), not in the `Card` domain model. A `_normalization_report.md` is also generated in the deck directory.
- **`Card.image_url`** is `str | None` — populated by the parser from HTML, carried through normalization. Not all cards have images.
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

# Fortune Teller

> Local-first Tarot reading app powered by RAG, local embeddings, and a
> llama.cpp chat model. Runs fully offline on Apple Silicon (M2+).

## Features (spike v0.0.1)

- Book of Thoth deck (78 cards)
- New Moon three-card spread (Past · Present · Future)
- Auto-deal with no-replace guarantee within a reading
- Per-card interpretations grounded in scraped card definitions
- Reading summary that surfaces cross-card reinforcing and conflicting themes
- Readings run entirely on-device — no external API keys or network needed at runtime (the optional Rider-Waite deck backfill, `ft-normalize-rw`, is a one-time build step that calls the Anthropic API)

## Requirements

- macOS 14+ (Apple Silicon recommended; Intel supported)
- Python 3.13 (managed by `uv`)
- [`uv`](https://docs.astral.sh/uv/) — `curl -LsSf https://astral.sh/uv/install.sh | sh`
- `llama.cpp` server running at `http://127.0.0.1:8080` (for chat only)

## Quickstart

```bash
# 1. Install full dev environment
uv sync --extra dev --group test --group lint

# 2. Install git hooks
uv run pre-commit install

# 3. One-time data pipeline (scrape → parse → normalise → embed → index)
uv run ft-fetch-models      # download embedding model for offline use
uv run ft-scrape            # scrape all decks (Book of Thoth + Rider-Waite)
uv run ft-parse             # parse all decks → card JSON (Thoth) / raw JSON (RW)
uv run ft-normalize-rw      # Rider-Waite backfill: RawCard → Card via the Anthropic API
                            #   • requires ANTHROPIC_API_KEY in .env (see Configuration)
                            #   • review data/parsed/rider-waite/_normalization_report.md
                            #   • --no-llm for a free deterministic dry run
uv run ft-embed             # embed all decks
uv run ft-build-index       # build the DuckDB vector index (all decks)
uv run ft-fetch-images      # download card artwork for all parsed decks into data/images/

# 4. Start the app
uv run fortune-teller
# → opens http://127.0.0.1:7860
```

## llama.cpp Setup (M2)

```bash
# Example: llama-3.2-8b-instruct Q4 quant
llama-server -hf bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q5_K_M \
  --host 127.0.0.1 --port 8080 -ngl 99
```

Embeddings are handled locally by `sentence-transformers` (`BAAI/bge-small-en-v1.5`)
and do **not** require a server.

## Configuration

Override any setting with an environment variable or `.env` file:

| Variable               | Default                           | Description                                          |
| ---------------------- | --------------------------------- | ---------------------------------------------------- |
| `OPENAI_BASE_URL`      | `http://127.0.0.1:8080/v1`        | llama.cpp server                                     |
| `OPENAI_API_KEY`       | `sk-no-key`                       | Any non-empty string                                 |
| `CHAT_MODEL`           | `local-model`                     | Model name as reported by server                     |
| `FT_DATA_DIR`          | `./data`                          | Path to data directory                               |
| `SQLITE_PATH`          | `./data/sqlite/fortune.db`        | Path to SQLite reading-history database              |
| `IMAGES_DIR`           | `./data/images`                   | Directory for card artwork images                    |
| `EMBEDDING_MODEL`      | `BAAI/bge-small-en-v1.5`          | HuggingFace embedding model (name or local path)     |
| `EMBEDDING_MODEL_PATH` | `./data/models/bge-small-en-v1.5` | Local path for offline embedding model               |
| `ANTHROPIC_API_KEY`    | _(unset)_                         | Anthropic key — required by `ft-normalize-rw`        |
| `NORMALIZE_PROVIDER`   | `api`                             | `ft-normalize-rw` backend: `api` (Claude) or `local` |
| `NORMALIZE_MODEL`      | `claude-sonnet-4-6`               | Model id used when `NORMALIZE_PROVIDER=api`          |

## Architecture

See [`docs/architecture.md`](docs/architecture.md).

## Development

```bash
uv run pytest                          # run all tests (≥80% coverage required)
uv run pytest -m "unit and not slow"   # fast subset only
uv run ruff check && uv run ruff format --check
uv run mypy src
```

## Roadmap

| Version        | Scope                                                                                                                                                                                                 |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `v0.0.1-spike` | Single deck, single spread, auto-deal, no auth                                                                                                                                                        |
| `v0.1.0`       | Bundle local embeddings model so the app runs fully offline after install (no HF Hub fetch)                                                                                                           |
| `v0.2.0`       | Reading history persistence (SQLite)                                                                                                                                                                  |
| `v0.3.0`       | Scrape, store, and serve card images; UI overlay with card artwork                                                                                                                                    |
| `v0.4.0`       | Interactive detail views — click a card name for a popup with its full structured entry + image + source attribution; hover a position title for a floating definition with a source-attribution link |
| `v0.5.0`       | Multiple decks, multiple spreads (Rider-Waite deck added)                                                                                                                                             |
| `v0.6.0`       | Manual card entry mode                                                                                                                                                                                |
| `v0.7.0`       | User login (local, hashed password)                                                                                                                                                                   |

## License

MIT — see [LICENSE](LICENSE).

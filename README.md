# Fortune Teller

> Local-first Tarot reading app powered by RAG, local embeddings, and a
> llama.cpp chat model. Runs fully offline on Apple Silicon (M2+).

## Features (spike v0.0.1)

- Book of Thoth deck (78 cards)
- New Moon three-card spread (Past · Present · Future)
- Auto-deal with no-replace guarantee within a reading
- Per-card interpretations grounded in scraped card definitions
- Reading summary that surfaces cross-card reinforcing and conflicting themes
- Runs entirely on-device — no external API keys required

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

# 3. One-time data pipeline (scrape → parse → embed → index)
uv run ft-scrape
uv run ft-parse
uv run ft-embed
uv run ft-build-index

# 4. Start the app
uv run fortune-teller
# → opens http://127.0.0.1:7860
```

## llama.cpp Setup (M2)

```bash
# Example: llama-3.2-8b-instruct Q4 quant
llama-server -m ~/models/llama-3.2-8b-instruct.Q4_K_M.gguf \
  --host 127.0.0.1 --port 8080 -ngl 99
```

Embeddings are handled locally by `sentence-transformers` (`BAAI/bge-small-en-v1.5`)
and do **not** require a server.

## Configuration

Override any setting with an environment variable or `.env` file:

| Variable           | Default                        | Description |
| ------------------ | ------------------------------ | ----------- |
| `OPENAI_BASE_URL`  | `http://127.0.0.1:8080/v1`     | llama.cpp server |
| `OPENAI_API_KEY`   | `sk-no-key`                    | Any non-empty string |
| `CHAT_MODEL`       | `local-model`                  | Model name as reported by server |
| `FT_DATA_DIR`      | `./data`                       | Path to data directory |
| `EMBEDDING_MODEL`  | `BAAI/bge-small-en-v1.5`       | HuggingFace embedding model |

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

| Version | Scope |
| ------- | ----- |
| `v0.0.1-spike` | Single deck, single spread, auto-deal, no auth |
| `v0.1.0` | Reading history persistence (SQLite) |
| `v0.2.0` | User login (local, hashed password) |
| `v0.3.0` | Multiple decks, multiple spreads |
| `v0.4.0` | Manual card entry mode |

## License

MIT — see [LICENSE](LICENSE).

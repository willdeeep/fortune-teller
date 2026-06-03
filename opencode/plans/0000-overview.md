# Fortune Teller — Project Plan Overview

A small, local-first Tarot reading app for Apple Silicon (M2+) using
RAG over scraped card definitions, a local llama.cpp chat model, and
local sentence-transformers embeddings.

## Locked Decisions

| Topic                | Decision                                                                  |
| -------------------- | ------------------------------------------------------------------------- |
| Python               | 3.13 (latest stable supported by all listed libs)                         |
| Env / packages       | `uv` only                                                                 |
| Repo layout          | Single package, PEP 735 `[dependency-groups]` + `[project.optional-deps]` |
| Chat model           | llama.cpp OpenAI-compatible server via `langchain-openai`                 |
| Embeddings           | `langchain-huggingface` + `BAAI/bge-small-en-v1.5` (local)               |
| Vector store         | DuckDB + `vss` extension (HNSW)                                           |
| Local storage        | SQLite (readings, future user prefs)                                      |
| UI                   | Gradio                                                                    |
| Domain validation    | pydantic v2                                                               |
| Lint/format          | `ruff`                                                                    |
| Type-check           | `mypy` (strict on `src/fortune_teller`)                                   |
| Tests                | `pytest` + `pytest-cov`, ≥80% gate                                        |
| Pre-commit           | ruff + hygiene + mypy + fast pytest subset                                |
| Login                | Not in spike                                                              |
| GitHub repo          | Private `fortune-teller` under user                                       |
| First deck           | Book of Thoth (`thothreadings.com`)                                       |
| First spread         | New Moon three-card                                                       |
| Spike dealing        | Auto-deal only, no-replace within reading, reset on new reading           |

## Plan File Index

| File | Topic |
| ---- | ----- |
| 0001 | Tooling & layout |
| 0002 | Documentation |
| 0003 | MCP servers |
| 0004 | Domain model |
| 0005 | Deck & reading engine |
| 0006 | Scraping & parsing |
| 0007 | Embeddings & vector store |
| 0008 | LLM chains |
| 0009 | Gradio UI |
| 0010 | Testing & quality |
| 0011 | CI & pre-commit |
| 0012 | GitHub & release |
| 0013 | Spike acceptance |

## Execution Order

1. Tooling & layout (0001)
2. Docs skeleton (0002)
3. MCP servers (0003)
4. Domain model (0004)
5. Deck/Reading no-replace (0005)
6. Scraper + parser w/ HTML fixtures (0006)
7. Embedding pipeline + DuckDB VSS (0007)
8. RAG chains with stubbed LLM (0008)
9. Gradio UI (0009)
10. Tests passing ≥80% + CI green (0010, 0011)
11. Push private repo (0012)
12. Verify spike acceptance (0013)

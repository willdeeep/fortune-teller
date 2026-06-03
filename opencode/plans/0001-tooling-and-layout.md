# 0001 — Tooling & Repository Layout

## Directory Layout

```
fortune-teller/
├── pyproject.toml
├── uv.lock
├── README.md
├── AGENTS.md
├── LICENSE
├── .python-version          # 3.13
├── .gitignore
├── .pre-commit-config.yaml
├── .github/workflows/ci.yml
├── .opencode/opencode.json
├── docs/
│   ├── architecture.md
│   └── decks/book-of-thoth.md
├── opencode/plans/          # this folder
├── src/fortune_teller/
│   ├── __init__.py
│   ├── application/
│   │   ├── __init__.py
│   │   ├── ui/
│   │   │   ├── __init__.py
│   │   │   └── app.py           # gradio entry point
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── deck.py          # DeckSession, DeckExhausted
│   │   │   └── reading.py       # ReadingService
│   │   ├── chains/
│   │   │   ├── __init__.py
│   │   │   ├── per_card.py      # per-card interpretation chain
│   │   │   └── summary.py       # reading summary chain
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   └── domain.py        # all pydantic models
│   │   ├── stores/
│   │   │   ├── __init__.py
│   │   │   ├── vector.py        # DuckDB VSS wrapper
│   │   │   └── sqlite.py        # SQLite reading history
│   │   └── config.py            # settings (env vars, paths)
│   └── developer/
│       ├── __init__.py
│       ├── scrape/
│       │   ├── __init__.py
│       │   ├── cli.py           # ft-scrape entry point
│       │   ├── thothreadings.py # site-specific scraper
│       │   └── seeds/
│       │       └── book_of_thoth.txt  # 78 slugs + spread slug
│       ├── parse/
│       │   ├── __init__.py
│       │   ├── cli.py           # ft-parse entry point
│       │   └── thothreadings.py # section-aware parser
│       ├── embed/
│       │   ├── __init__.py
│       │   └── cli.py           # ft-embed entry point
│       └── build_index/
│           ├── __init__.py
│           └── cli.py           # ft-build-index entry point
├── tests/
│   ├── conftest.py
│   ├── unit/
│   │   ├── test_domain_models.py
│   │   ├── test_deck_session.py
│   │   ├── test_parser.py
│   │   └── test_prompt_templates.py
│   ├── integration/
│   │   ├── test_vector_store.py
│   │   └── test_reading_chain.py
│   └── fixtures/
│       ├── html/                # cached thothreadings.com pages
│       └── parsed/              # golden parsed JSON
└── data/                        # gitignored
    ├── cache/                   # raw scraped HTML
    ├── duckdb/
    └── sqlite/
```

## `pyproject.toml` (complete)

```toml
[project]
name = "fortune-teller"
version = "0.0.1"
description = "Local-first Tarot reading app — RAG over scraped definitions"
readme = "README.md"
license = { text = "MIT" }
requires-python = ">=3.13,<3.14"
dependencies = [
    "langchain>=0.3",
    "langchain-core>=0.3",
    "langchain-openai>=0.2",
    "langchain-huggingface>=0.1",
    "sentence-transformers>=3.0",
    "pydantic>=2.7",
    "duckdb>=1.1",
    "gradio>=5.0",
    "httpx>=0.27",
    "platformdirs>=4.2",
]

[project.optional-dependencies]
dev = [
    "selectolax>=0.3",
    "beautifulsoup4>=4.12",
    "tenacity>=9.0",
    "rich>=13.7",
    "typer>=0.12",
]

[project.scripts]
fortune-teller   = "fortune_teller.application.ui.app:main"
ft-scrape        = "fortune_teller.developer.scrape.cli:main"
ft-parse         = "fortune_teller.developer.parse.cli:main"
ft-embed         = "fortune_teller.developer.embed.cli:main"
ft-build-index   = "fortune_teller.developer.build_index.cli:main"

[dependency-groups]
test = [
    "pytest>=8.3",
    "pytest-cov>=5.0",
    "pytest-mock>=3.14",
    "hypothesis>=6.112",
    "coverage[toml]>=7.6",
]
lint = [
    "ruff>=0.6",
    "mypy>=1.11",
    "pre-commit>=3.8",
    "types-beautifulsoup4",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/fortune_teller"]

[tool.ruff]
target-version = "py313"
line-length = 100
src = ["src", "tests"]

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "SIM", "RUF", "N", "PL", "ARG", "TID"]
ignore = ["PLR0913"]

[tool.ruff.lint.per-file-ignores]
"tests/**" = ["PLR2004", "S101"]

[tool.ruff.format]
docstring-code-format = true

[tool.mypy]
python_version = "3.13"
strict = true
files = ["src/fortune_teller"]
plugins = ["pydantic.mypy"]

[[tool.mypy.overrides]]
module = [
    "sentence_transformers.*",
    "duckdb.*",
    "selectolax.*",
    "gradio.*",
    "langchain.*",
    "langchain_core.*",
    "langchain_openai.*",
    "langchain_huggingface.*",
]
ignore_missing_imports = true

[tool.pytest.ini_options]
addopts = "-ra --strict-markers --cov=fortune_teller --cov-branch --cov-report=term-missing --cov-fail-under=80"
testpaths = ["tests"]
markers = [
    "unit: fast unit tests (pre-commit subset)",
    "integration: tests touching DuckDB/SQLite/HTTP fixtures",
    "slow: slow tests skipped from pre-commit",
]

[tool.coverage.run]
branch = true
source = ["fortune_teller"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "if TYPE_CHECKING:",
    "raise NotImplementedError",
]
```

## Install / Run Recipes

```bash
# pin python
uv python install 3.13

# runtime only
uv sync

# runtime + developer CLIs (scraping, index building)
uv sync --extra dev

# full dev environment (runtime + dev extras + test + lint groups)
uv sync --extra dev --group test --group lint

# run app
uv run fortune-teller

# one-time data pipeline (requires --extra dev)
uv run ft-scrape
uv run ft-parse
uv run ft-embed
uv run ft-build-index

# tests
uv run pytest

# linting
uv run ruff check && uv run ruff format --check

# type-checking
uv run mypy src
```

# 0011 — CI & Pre-commit

---

## Pre-commit Configuration

File: `.pre-commit-config.yaml`

```yaml
repos:
  # --- Ruff: format + lint + import sort -----------------------------------
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.9
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  # --- General file hygiene ------------------------------------------------
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v5.0.0
    hooks:
      - id: end-of-file-fixer
      - id: trailing-whitespace
      - id: check-yaml
      - id: check-toml
      - id: check-merge-conflict
      - id: mixed-line-ending
        args: [--fix=lf]
      - id: check-added-large-files
        args: [--maxkb=500]     # block accidental data/ commits

  # --- Mypy: type-checking -------------------------------------------------
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.11.2
    hooks:
      - id: mypy
        additional_dependencies:
          - pydantic>=2.7
          - pydantic-settings>=2.4
        args: [--config-file=pyproject.toml]
        files: ^src/fortune_teller/

  # --- Fast pytest subset (unit tests only) --------------------------------
  - repo: local
    hooks:
      - id: pytest-fast
        name: pytest (unit, fast)
        entry: uv run pytest -m "unit and not slow" -q --no-header
        language: system
        pass_filenames: false
        stages: [pre-commit]
```

### Install pre-commit

```bash
uv run pre-commit install        # installs git hooks
uv run pre-commit run --all-files  # validate full repo on first install
```

---

## GitHub Actions CI

File: `.github/workflows/ci.yml`

```yaml
name: CI

on:
  push:
    branches: ["main", "feat/**"]
  pull_request:
    branches: ["main"]

jobs:
  test:
    name: Test (${{ matrix.os }}, Python ${{ matrix.python-version }})
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [macos-14, ubuntu-latest]
        python-version: ["3.13"]

    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v3
        with:
          enable-cache: true
          cache-dependency-glob: "uv.lock"

      - name: Set up Python ${{ matrix.python-version }}
        run: uv python install ${{ matrix.python-version }}

      - name: Install dependencies (full dev environment)
        run: uv sync --extra dev --group test --group lint

      - name: Ruff lint check
        run: uv run ruff check .

      - name: Ruff format check
        run: uv run ruff format --check .

      - name: Mypy type check
        run: uv run mypy src

      - name: Pytest (unit + integration, coverage gate ≥80%)
        run: uv run pytest --tb=short

      - name: Upload coverage report
        if: matrix.os == 'ubuntu-latest'
        uses: actions/upload-artifact@v4
        with:
          name: coverage-report
          path: .coverage
          retention-days: 7
```

---

## Caching Notes

- `astral-sh/setup-uv@v3` automatically caches `~/.cache/uv` keyed on
  `uv.lock` hash. This means installs are fast after the first run.
- HuggingFace model weights are NOT cached in CI — integration tests that
  require a real embedding model are marked `@pytest.mark.slow` and excluded
  from the standard CI run. Only stub-based integration tests run in CI.
- To run slow tests locally: `uv run pytest -m slow`

---

## Branch Protection (post repo creation)

On `main`:
- Require status checks: `test (macos-14, 3.13)` and `test (ubuntu-latest, 3.13)`
- Require branches to be up-to-date before merging
- Do not allow force-pushes
- Do not allow branch deletion

---

## Suggested Commit Convention

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add DeckSession no-replace logic
fix: handle missing card section in parser
test: add property tests for deck exhaustion
docs: update architecture diagram
chore: pin ruff to v0.6.9 in pre-commit
refactor: extract chunk builder from vector store
```

This convention makes the changelog and release notes straightforward
to generate later.

# 0013 — Spike Acceptance Criteria

The spike (`v0.0.1-spike`) is **done** only when ALL of the following
are verified:

---

## Environment & Install

- [ ] `git clone <repo> && cd fortune-teller` succeeds on macOS 14+ (Apple Silicon).
- [ ] `uv sync --extra dev --group test --group lint` completes without errors.
- [ ] `uv run pre-commit install` installs hooks without errors.
- [ ] `uv run pre-commit run --all-files` passes cleanly on a fresh checkout.

---

## Data Pipeline (developer package)

- [ ] `uv run ft-scrape` produces exactly **78 card HTML files** under
  `data/cache/thothreadings.com/` plus `spread-new-moon.html`.
  (Re-run reads from cache, no duplicate HTTP requests.)
- [ ] `uv run ft-parse` produces exactly **78 valid `Card` JSON files**
  under `data/parsed/book-of-thoth/` plus
  `data/parsed/spreads/new-moon-three-card.json`.
  Each JSON file passes `Card.model_validate_json()` / `Spread.model_validate_json()`.
- [ ] `uv run ft-embed` augments all parsed files with an `embedding`
  field (list of 384 floats).
- [ ] `uv run ft-build-index` populates
  `data/duckdb/fortune.duckdb` with **≥858 chunks**
  (78 cards × ~11 sections) plus 3 spread position chunks.

---

## Application

- [ ] `uv run fortune-teller` starts without errors and opens Gradio at
  `http://127.0.0.1:7860`.
- [ ] The page title reads "Fortune Teller" and shows the New Moon spread
  description.
- [ ] Clicking "New Reading":
  - [ ] Deals exactly **3 distinct cards** (no duplicates within the reading).
  - [ ] Each card panel shows: card name, orientation (▲ UPRIGHT or ▼ REVERSED),
    and a grounded interpretation (3–5 sentences, non-empty).
  - [ ] Cards appear **progressively** (Past → Present → Future) without
    waiting for all three.
  - [ ] The **Reading Summary** panel populates after the third card
    (4–8 sentences, non-empty).
- [ ] Clicking "New Reading" a second time:
  - [ ] Deals a **different combination** of cards (with high probability).
  - [ ] The full deck of 78 cards is available again (reset confirmed by
    running ≥10 successive readings without `DeckExhausted`).
- [ ] Interpretations and summary are grounded in card/spread definition
  text — they do not hallucinate card meanings not present in the
  vector store.

---

## Tests & Quality

- [ ] `uv run pytest` exits green (all tests pass).
- [ ] Coverage is **≥80%** (branch coverage) — reported by pytest-cov.
- [ ] `uv run ruff check .` exits 0 (no lint violations).
- [ ] `uv run ruff format --check .` exits 0 (all files formatted).
- [ ] `uv run mypy src` exits 0 (no type errors under strict mode).

---

## CI

- [ ] GitHub Actions CI passes on both `macos-14` and `ubuntu-latest`
  runners for all steps: ruff, mypy, pytest.
- [ ] The branch protection rule on `main` is enforced (direct pushes
  blocked; CI required).

---

## Documentation

- [ ] `README.md` quickstart, followed verbatim by a new user on a clean
  machine, results in a working reading within the Gradio UI.
- [ ] `AGENTS.md` accurately describes the conventions currently in use.
- [ ] `docs/architecture.md` reflects the actual component structure
  as built.
- [ ] `docs/decks/book-of-thoth.md` lists all 11 section types and
  includes source attribution.

---

## Definition of "Done"

All checkboxes above are ticked. Tag `v0.0.1-spike`, create a GitHub
pre-release, and update `README.md` badge links to point at the tag.

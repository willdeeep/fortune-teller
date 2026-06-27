# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Since `v0.5.1` the version is derived automatically from the git tag (hatch-vcs),
so a release is created by tagging — there is no version field to bump.

## [Unreleased]

## [0.7.1] — 2026-06-27

### Fixed

- **Reading resilience to LLM failures** (plan 0038, #45) — the per-card and
  summary chains now use separate LLM clients with independent timeouts. The
  summary timeout scales with spread size (`summary_timeout_base` 120s +
  `summary_timeout_per_card` 12s × positions — a 10-card Celtic Cross gets
  240s) instead of the old single 180s ceiling, so large spreads no longer fail
  on slower local models. `_run_reading` now catches reading failures and shows
  them in the summary area plus a toast instead of escaping and crashing
  NiceGUI's background-task handler; `build_app` registers a global
  `app.on_exception` safety net. New settings: `per_card_timeout`,
  `summary_timeout_base`, `summary_timeout_per_card`.
- **Thoth cards scraped from full definition pages** (plan 0039, #42) — the
  Book of Thoth scraper now fetches card pages from the site root
  (`thothreadings.com/<slug>/`) instead of the truncated `/blog/<slug>/`
  summaries, so card definitions (and their embeddings) are complete and the
  "View source" links resolve to the full pages. Removes the `/blog/` path
  handling. Requires a deck-data rebuild to take effect.
- **Rider-Waite reinforce/oppose synergy parsing** (plan 0040, #40) — the
  learntarot parser now resolves reinforcing/opposing cards by the link's
  **href slug** (the authoritative card id) via DOM extraction, instead of
  parsing display text. Fixes silently-dropped links caused by split/mangled
  headings, prose leaks, missing "of" (`Ten Wands`), source typos
  (`Four o Swords`), and leaked markup, and removes the fragile name-resolution
  layer (`resolve_card_names`, `_normalize_card_name`, `_split_names`, and the
  overrides/digit maps). Requires a deck-data rebuild to take effect.
- **Source-attribution links open in a new tab** (#41) — the card-detail
  "View source" and position-meaning "Source" links now open in a new browser
  tab (`target="_blank"`), so the reading isn't navigated away from.

## [0.7.0] — 2026-06-25

### Added

- **UI deck selection** (plan 0023) — a deck `ui.select` dropdown backed by
  `list_decks` lets the reader choose which deck a reading uses (e.g. Book of
  Thoth vs Rider-Waite). Services are cached per `(deck_id, spread_id)` pair;
  image URLs resolve from the parent `images_dir` using the current deck ID
  (`/images/<deck_id>/<file>`). Single-deck setups render no selector
  (back-compat). The title and history detail surface the deck name.
- **Reversed card rotation** (plan 0025) — a reversed card's artwork is now
  displayed rotated 180° via a native CSS `transform` on the image element
  (`rotation_style` helper, applied in `_run_reading`); upright cards are
  unchanged and the transform is cleared when a slot is re-dealt upright. No new
  dependency (Pillow not required).
- **Interactive detail views** (plan 0024) — clicking a position title opens a
  `ui.dialog` showing the position's meaning + source link (grid and row
  layouts; the grid title click stops propagation so it doesn't also open the
  card detail). The card-detail dialog now lists reinforcing/opposing card
  names, surfacing the v0.6.0 synergy data in the UI. Replaces the static
  position-meanings markdown block with interactive popups.
- **Complex spreads + Celtic Cross** (plan 0030) — `SpreadPosition` gains
  optional `row`/`col`/`rotation` layout fields (linear row layout when absent).
  The NiceGUI UI renders 2D spreads via a CSS grid (with a 90°-rotated crossing
  card) and gains a spread selector (`ui.select`) so readers can switch spreads;
  `list_spreads` exposes `(id, name)` options and a service factory builds a
  reading service per spread. Ships the authored `celtic-cross.json` spread
  (10 positions).

### Changed

- **Usable spread layout** (plan 0036) — replaced the overflowing in-cell
  interpretation text with a single spread-agnostic renderer: a spatial grid of
  fixed-size card boxes (CSS card backs that flip to face images on deal, with a
  true centred 90° crossing card) plus a numbered interpretation list below as
  the source of truth for text. Fixes the unusable Celtic Cross spread that
  blocked v0.7.0. Adds an opt-in, non-gating NiceGUI `Screen` screenshot harness
  (`uv run pytest -m screen --no-cov`) for visual verification.
- **Roadmap re-sequenced** (README Roadmap — the committed source of truth)
  after an in-browser UX review. **v0.7.0** is scoped to a _usable_ UI and gains
  a Celtic Cross layout fix (0036) — the grid previously rendered each card's
  full interpretation inside a 100px cell, so text overran and overlapped. UX
  polish (dark theme, History on its own tab + a compact expandable table,
  header reorganisation) splits into **v0.8.0** (0037), which also pulls in
  post-reading **user notes** (0034) so the History redesign ships with notes.
  **Added functionality** (manual card entry, framing question) moves to
  **v0.9.0**; **single-user login** (0031) becomes its own **v0.10.0**.
- **UI migrated from Gradio to NiceGUI** (plan 0035) — the `fortune-teller`
  console script now launches a NiceGUI app (`application/ui/nicegui_app.py`,
  on FastAPI). Card detail views use a real `ui.dialog` overlay instead of the
  always-visible Markdown panel; the reading sequence updates progressively via
  `asyncio.to_thread` for the blocking LLM calls. Framework-agnostic formatters
  and the reading generator carried over unchanged. `gradio` is replaced by
  `nicegui` in the dependencies. Establishes the framework for the v0.7.0 UI
  work (0023/0024/0025/0030/0031).

## [0.6.1] — 2026-06-17

### Changed

- **Untracked `.opencode/opencode.json`** — removed local editor config from
  version control (`git rm --cached`); the `.gitignore` `.opencode` rule keeps
  `.opencode/` and `.claude/` out going forward. A `.gitignore` rule can only
  ignore files git is _not already tracking_, so the file had to be untracked
  explicitly. Dropped the redundant `.opencode/*` ignore line.

### Added

- **`ft-normalize-rw` CLI test coverage** — `tests/unit/test_normalize_cli.py`
  covers the `developer/normalize/cli.py` entry point (provider/no-llm/only
  flags + missing-raw-dir exit), taking it from 0% to 100%.

## [0.6.0] — 2026-06-17

### Added

- **Reinforce/Oppose domain model** — `Card.reinforcing_ids` / `opposing_ids`
  fields; `CardSection.REINFORCING` / `OPPOSING` enum values; orientation XOR
  rule (`effective_relationship`): exactly one reversed card flips the
  relationship (reinforce↔oppose).
- **RW name→ID resolver** — `resolve_card_names()` normalises learntarot display
  names (strips "The", maps digit ranks, singularises suits, overrides map) to
  internal card IDs. Unresolvable names are logged and reported in
  `_normalization_report.md`.
- **Reading-time synergy** — `compute_synergies()` finds all reinforce/oppose
  pairs among dealt cards; `render_synergy_block()` surfaces them in the summary
  prompt using card display names; `ReadingService.finalize()` wires synergies
  into the summary chain.
- **Thoth reinforce/oppose LLM synthesis** — `synthesize_card_synergies()` and
  `synthesize_deck_synergies()` use an LLM to synthesize reinforcing/opposing
  card IDs for each Thoth card, with validation (max 5 per category,
  self-reference removal, deck membership check). Writes `_synergy_report.md`.
- **`ft-normalize-thoth`** CLI — `--provider`, `--model`, `--only`, `--no-llm`.
- **`ft-normalize`** umbrella CLI — `--deck rw|thoth|all`, inherits other flags.

### Fixed

- **RW resolver silently dropped most references** — the initial
  `resolve_card_names` used simple lowercase matching, which lost every
  major-arcana reference (learntarot omits "The") and all numeric minors
  ("2 of Wands" vs "Two of Wands"). Fixed with multi-step normalisation +
  overrides map + unresolved-names reporting.

## [0.5.2] — 2026-06-16

### Added

- This `CHANGELOG.md`, documenting the release history to date.

### Changed

- **Roadmap re-prioritised** (README Roadmap table — the committed source of
  truth): **v0.6.0** Reinforce/Oppose synergy (orientation-aware; Thoth
  relationships LLM-backfilled) · **v0.7.0** UI improvement (framework
  evaluation, complex spreads incl. Celtic Cross, interactive detail views,
  deck picker, reversed-card rotation, single-user login) · **v0.8.0** added
  functionality (manual card entry, framing question, post-reading notes).
  Tracked in GitHub issues #26 / #9 / #7 and detailed in local planning docs
  `.opencode/plans/0021`–`0034` (not committed).

## [0.5.1] — 2026-06-15

### Changed

- **Automatic versioning** via `hatch-vcs`: the git tag is the single source of
  truth; `__version__` is read from package metadata. Removes manual
  version-file bumps and prior tag/`pyproject`/`__init__` drift. CI fetches full
  history/tags so the version resolves on build.

### Added

- Quickstart instructions for creating a `.env` with `ANTHROPIC_API_KEY`
  (only needed for the `ft-normalize-rw` step).

## [0.5.0] — 2026-06-15

### Added

- **Rider-Waite tarot deck** — a second deck, end to end: scrape
  (`learntarot.com`) → parse → LLM normalisation (Claude) → embed → index →
  card images.
- **Deck isolation** — vector retrieval scoped by `deck_id`; per-deck
  `meta.json`; `list_decks`. The two decks never bleed into one another.
- `ft-normalize-rw` (RawCard → Card); all-sources `ft-scrape` / `ft-parse`;
  `ft-fetch-images` covers every parsed deck.

### Fixed

- Reading-history detail view crash — use gradio `SelectData.row_value`
  instead of the non-existent `.row`.

## [0.4.0] — 2026-06-14

### Added

- **Interactive detail views** — a per-card/position detail panel showing the
  full structured entry, image, and source attribution.
- UI overlay composing card artwork with the interpretation text.

## [0.3.0] — 2026-06-14

### Added

- **Card images** — scrape, store, and serve Book of Thoth artwork; basic
  image display alongside each card's interpretation (`ft-fetch-images`).

## [0.2.0] — 2026-06-13

> Note: tag **backfilled** on 2026-06-16 — the feature shipped on 2026-06-13
> (commit `71c3e81`) but was not tagged at the time.

### Added

- **Reading history persistence (SQLite)** — readings are autosaved; a History
  tab lists past readings and shows their detail.

## [0.1.0] — 2026-06-13

### Added

- **Fully-offline embeddings** — `ft-fetch-models` downloads the embedding model
  for offline use; runtime no longer contacts the HuggingFace Hub.

## [0.0.1-spike] — 2026-06-11

### Added

- Initial spike: Book of Thoth deck (78 cards), New Moon three-card spread,
  auto-deal with no-replace, RAG per-card interpretations + a cross-card summary,
  local llama.cpp chat model, DuckDB (VSS) vector store, and a Gradio UI.

[Unreleased]: https://github.com/willdeeep/fortune-teller/compare/v0.5.2...HEAD
[0.5.2]: https://github.com/willdeeep/fortune-teller/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/willdeeep/fortune-teller/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/willdeeep/fortune-teller/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/willdeeep/fortune-teller/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/willdeeep/fortune-teller/compare/v0.1.0...v0.3.0
[0.2.0]: https://github.com/willdeeep/fortune-teller/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/willdeeep/fortune-teller/compare/v0.0.1-spike...v0.1.0
[0.0.1-spike]: https://github.com/willdeeep/fortune-teller/releases/tag/v0.0.1-spike

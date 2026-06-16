# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Since `v0.5.1` the version is derived automatically from the git tag (hatch-vcs),
so a release is created by tagging — there is no version field to bump.

## [Unreleased]

_Nothing yet._

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

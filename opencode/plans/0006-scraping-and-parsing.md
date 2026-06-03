# 0006 — Scraping & Parsing (developer package)

Modules:
- `fortune_teller.developer.scrape.thothreadings` — async HTTP scraper
- `fortune_teller.developer.scrape.cli` — `ft-scrape` entry point
- `fortune_teller.developer.parse.thothreadings` — section-aware HTML parser
- `fortune_teller.developer.parse.cli` — `ft-parse` entry point
- `fortune_teller.developer.scrape.seeds/book_of_thoth.txt` — 78 card slugs

---

## Politeness & Caching

Before the first scrape:
1. Check `https://thothreadings.com/robots.txt` — verify no disallow rules
   for the card and spread paths.
2. Commit the result of the check to `docs/decks/book-of-thoth.md` under a
   "robots.txt" section.

Runtime behaviour:
- `User-Agent: fortune-teller/0.0.1 (+https://github.com/<user>/fortune-teller)`
- Rate limit: **1 request per second** (hard-coded minimum delay).
- Retry: `tenacity.retry` with exponential backoff (3 attempts, wait 2–30s).
- HTML cached to `data/cache/thothreadings.com/<slug>.html`.
- Re-run reads from cache unless `--refresh` flag is passed to `ft-scrape`.

---

## Target URLs

### Card pages
Pattern: `https://thothreadings.com/<slug>/`

All 78 slugs committed to:
`src/fortune_teller/developer/scrape/seeds/book_of_thoth.txt`

One slug per line, e.g.:
```
the-fool
the-magus
the-high-priestess
the-empress
the-emperor
...
```

### Spread page
`https://thothreadings.com/spread-new-moon/`

---

## Scraper Implementation Sketch

```python
import asyncio
import time
from pathlib import Path

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential


CACHE_DIR = Path("data/cache/thothreadings.com")
BASE_URL = "https://thothreadings.com"
USER_AGENT = "fortune-teller/0.0.1"
REQUEST_DELAY_SECONDS = 1.0


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=30))
async def fetch_page(client: httpx.AsyncClient, slug: str, *, refresh: bool = False) -> str:
    cache_path = CACHE_DIR / f"{slug}.html"
    if cache_path.exists() and not refresh:
        return cache_path.read_text(encoding="utf-8")
    await asyncio.sleep(REQUEST_DELAY_SECONDS)
    response = await client.get(f"{BASE_URL}/{slug}/")
    response.raise_for_status()
    html = response.text
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(html, encoding="utf-8")
    return html


async def scrape_deck(slugs: list[str], *, refresh: bool = False) -> None:
    async with httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT},
        follow_redirects=True,
        timeout=30.0,
    ) as client:
        for slug in slugs:
            await fetch_page(client, slug, refresh=refresh)
```

---

## Parser — Section Extraction Strategy

The thothreadings.com card pages follow a consistent heading structure.
The parser locates each heading keyword and captures the text block that
follows it, until the next heading.

### Target sections and heading keywords

| Section key  | Heading text (case-insensitive) |
| ------------ | ------------------------------- |
| overall      | (main article intro before first section heading) |
| drive        | "Drive" |
| light        | "Light" |
| shadow       | "Shadow" |
| reversed     | "Reversed" |
| keywords     | "Keywords" |
| advice       | "Advice" or "Instructions" |
| question     | "Question" |
| proposal     | "Proposal" |
| confirmation | "Confirmation" |
| affirmation  | "Affirmation" |

### Parser implementation sketch

```python
from selectolax.parser import HTMLParser
from fortune_teller.application.models.domain import (
    Card, CardSectionText, CardSection, Arcana,
)


HEADING_TO_SECTION: dict[str, CardSection] = {
    "drive": CardSection.DRIVE,
    "light": CardSection.LIGHT,
    "shadow": CardSection.SHADOW,
    "reversed": CardSection.REVERSED,
    "keywords": CardSection.KEYWORDS,
    "advice": CardSection.ADVICE,
    "instructions": CardSection.ADVICE,
    "question": CardSection.QUESTION,
    "proposal": CardSection.PROPOSAL,
    "confirmation": CardSection.CONFIRMATION,
    "affirmation": CardSection.AFFIRMATION,
}


def parse_card_page(html: str, slug: str, source_url: str) -> Card:
    tree = HTMLParser(html)
    sections: list[CardSectionText] = []
    # ... extract article body, walk nodes, match headings ...
    return Card(
        id=slug,
        name=_slug_to_name(slug),
        arcana=_detect_arcana(slug),
        suit=_detect_suit(slug),
        sections=sections,
        source_url=source_url,  # type: ignore[arg-type]
    )
```

### Spread parser

```python
def parse_spread_page(html: str, source_url: str) -> Spread:
    """
    Parses the New Moon spread page.
    Extracts three position blocks: their names and meaning text.
    """
    ...
```

---

## Output Files

- Parsed cards: `data/parsed/book-of-thoth/<slug>.json`
  (each file is a `Card` model serialised with `model.model_dump_json()`)
- Parsed spread: `data/parsed/spreads/new-moon-three-card.json`
  (a `Spread` model)

---

## Tests (no live HTTP)

- Commit golden HTML for at least:
  - `tests/fixtures/html/thothreadings/the-fool.html`
  - `tests/fixtures/html/thothreadings/the-magus.html`
  - `tests/fixtures/html/thothreadings/ace-of-wands.html`
  - `tests/fixtures/html/thothreadings/spread-new-moon.html`
- Commit golden parsed JSON:
  - `tests/fixtures/parsed/book-of-thoth/the-fool.json`
  - `tests/fixtures/parsed/spreads/new-moon-three-card.json`

### Test cases

```python
@pytest.mark.unit
def test_parse_the_fool_has_all_sections(the_fool_html: str) -> None:
    card = parse_card_page(the_fool_html, "the-fool", "https://thothreadings.com/the-fool/")
    section_keys = {s.section for s in card.sections}
    assert CardSection.DRIVE in section_keys
    assert CardSection.AFFIRMATION in section_keys

@pytest.mark.unit
def test_parse_the_fool_name(the_fool_html: str) -> None:
    card = parse_card_page(the_fool_html, "the-fool", "https://thothreadings.com/the-fool/")
    assert card.name == "The Fool"
    assert card.arcana == Arcana.MAJOR

@pytest.mark.unit
def test_parse_ace_of_wands_is_minor(ace_of_wands_html: str) -> None:
    card = parse_card_page(ace_of_wands_html, "ace-of-wands", "https://...")
    assert card.arcana == Arcana.MINOR
    assert card.suit == Suit.WANDS

@pytest.mark.unit
def test_parse_spread_has_three_positions(new_moon_spread_html: str) -> None:
    spread = parse_spread_page(new_moon_spread_html, "https://thothreadings.com/spread-new-moon/")
    assert len(spread.positions) == 3
    assert spread.positions[0].index == 0

@pytest.mark.unit
def test_scraper_uses_cache(tmp_path: Path, monkeypatch) -> None:
    # Write a fake HTML file to cache, assert no HTTP call is made
    ...
```

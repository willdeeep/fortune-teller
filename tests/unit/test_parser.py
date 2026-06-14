"""Unit tests for the thothreadings.com HTML parser.

All tests are pure logic using committed fixture HTML files.
No live HTTP calls are made.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import fortune_teller.developer.scrape.thothreadings as scrape_mod
from fortune_teller.application.config import Settings
from fortune_teller.application.models.domain import (
    Arcana,
    Card,
    CardSection,
    Spread,
    Suit,
)
from fortune_teller.developer.parse.thothreadings import (
    _slug_to_name,
    parse_card_page,
    parse_spread_page,
)
from fortune_teller.developer.scrape.thothreadings import (
    _build_url,
    fetch_page,
    load_slugs,
    scrape_slugs,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_FIXTURES = Path(__file__).parent.parent / "fixtures"
_HTML_DIR = _FIXTURES / "html" / "thothreadings"
_PARSED_DIR = _FIXTURES / "parsed"


def _html(name: str) -> str:
    path = _HTML_DIR / name
    if not path.exists():
        pytest.skip(f"HTML fixture not found: {path}")
    return path.read_text(encoding="utf-8")


def _golden(subdir: str, name: str) -> dict:  # type: ignore[type-arg]
    path = _PARSED_DIR / subdir / name
    if not path.exists():
        pytest.skip(f"Golden JSON fixture not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# _slug_to_name
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSlugToName:
    def test_digit_prefix_stripped(self) -> None:
        assert _slug_to_name("0-the-fool") == "The Fool"

    def test_roman_i_stripped(self) -> None:
        assert _slug_to_name("i-the-magician") == "The Magician"

    def test_roman_ii_stripped(self) -> None:
        assert _slug_to_name("ii-the-high-priestess") == "The High Priestess"

    def test_roman_viii_stripped(self) -> None:
        assert _slug_to_name("viii-adjustment") == "Adjustment"

    def test_roman_xxi_stripped(self) -> None:
        assert _slug_to_name("xxi-the-universe") == "The Universe"

    def test_minor_no_prefix(self) -> None:
        assert _slug_to_name("ace-of-wands") == "Ace of Wands"

    def test_minor_with_subtitle(self) -> None:
        assert _slug_to_name("two-of-wands-dominion") == "Two of Wands Dominion"

    def test_court_card(self) -> None:
        assert _slug_to_name("princess-of-wands") == "Princess of Wands"

    def test_of_stays_lowercase(self) -> None:
        # "of" is a stop word and stays lowercase wherever it appears
        result = _slug_to_name("ace-of-wands")
        assert " of " in result

    def test_the_is_capitalised(self) -> None:
        # "the" is NOT a stop word — always capitalised
        result = _slug_to_name("0-the-fool")
        assert result.startswith("The")


# ---------------------------------------------------------------------------
# parse_card_page — The Fool (0-the-fool.html)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestParseTheFool:
    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self.html = _html("0-the-fool.html")
        self.card = parse_card_page(self.html, "0-the-fool")

    def test_name(self) -> None:
        assert self.card.name == "The Fool"

    def test_id(self) -> None:
        assert self.card.id == "0-the-fool"

    def test_arcana_is_major(self) -> None:
        assert self.card.arcana == Arcana.MAJOR

    def test_suit_is_none(self) -> None:
        assert self.card.suit is None

    def test_number_is_zero(self) -> None:
        assert self.card.number == 0

    def test_has_overall_section(self) -> None:
        section_keys = {s.section for s in self.card.sections}
        assert CardSection.OVERALL in section_keys

    def test_has_drive_section(self) -> None:
        assert CardSection.DRIVE in {s.section for s in self.card.sections}

    def test_has_light_section(self) -> None:
        assert CardSection.LIGHT in {s.section for s in self.card.sections}

    def test_has_shadow_section(self) -> None:
        assert CardSection.SHADOW in {s.section for s in self.card.sections}

    def test_has_keywords_section(self) -> None:
        assert CardSection.KEYWORDS in {s.section for s in self.card.sections}

    def test_has_advice_section(self) -> None:
        assert CardSection.ADVICE in {s.section for s in self.card.sections}

    def test_has_question_section(self) -> None:
        assert CardSection.QUESTION in {s.section for s in self.card.sections}

    def test_has_affirmation_section(self) -> None:
        assert CardSection.AFFIRMATION in {s.section for s in self.card.sections}

    def test_all_section_texts_non_empty(self) -> None:
        for s in self.card.sections:
            assert s.text.strip(), f"Section {s.section} has empty text"

    def test_no_duplicate_sections(self) -> None:
        keys = [s.section for s in self.card.sections]
        assert len(keys) == len(set(keys))

    def test_source_url_contains_slug(self) -> None:
        assert "0-the-fool" in str(self.card.source_url)

    def test_source_url_contains_blog(self) -> None:
        assert "/blog/" in str(self.card.source_url)

    def test_drive_text_content(self) -> None:
        text = self.card.section_text(CardSection.DRIVE)
        assert text is not None
        assert len(text) > 5

    def test_model_validates(self) -> None:
        # Round-trip through JSON to confirm pydantic validation passes
        reloaded = Card.model_validate_json(self.card.model_dump_json())
        assert reloaded.id == self.card.id

    def test_image_url_is_populated(self) -> None:
        """End-to-end: parse_card_page populates Card.image_url."""
        assert self.card.image_url is not None
        assert "wp-content/uploads" in str(self.card.image_url)

    def test_matches_golden_json(self) -> None:
        golden = _golden("book-of-thoth", "0-the-fool.json")
        assert golden["id"] == self.card.id
        assert golden["name"] == self.card.name
        assert golden["arcana"] == self.card.arcana.value
        assert len(golden["sections"]) == len(self.card.sections)


# ---------------------------------------------------------------------------
# parse_card_page — The Magician (i-the-magician.html)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestParseTheMagician:
    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self.html = _html("i-the-magician.html")
        self.card = parse_card_page(self.html, "i-the-magician")

    def test_name(self) -> None:
        assert self.card.name == "The Magician"

    def test_arcana_is_major(self) -> None:
        assert self.card.arcana == Arcana.MAJOR

    def test_number_is_one(self) -> None:
        assert self.card.number == 1

    def test_has_minimum_sections(self) -> None:
        assert len(self.card.sections) >= 5

    def test_no_duplicate_sections(self) -> None:
        keys = [s.section for s in self.card.sections]
        assert len(keys) == len(set(keys))

    def test_all_section_texts_non_empty(self) -> None:
        for s in self.card.sections:
            assert s.text.strip()


# ---------------------------------------------------------------------------
# parse_card_page — Ace of Wands (ace-of-wands-blog.html)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestParseAceOfWands:
    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self.html = _html("ace-of-wands-blog.html")
        self.card = parse_card_page(self.html, "ace-of-wands")

    def test_name(self) -> None:
        assert self.card.name == "Ace of Wands"

    def test_arcana_is_minor(self) -> None:
        assert self.card.arcana == Arcana.MINOR

    def test_suit_is_wands(self) -> None:
        assert self.card.suit == Suit.WANDS

    def test_number_is_one(self) -> None:
        assert self.card.number == 1

    def test_has_drive_section(self) -> None:
        assert CardSection.DRIVE in {s.section for s in self.card.sections}

    def test_has_light_section(self) -> None:
        assert CardSection.LIGHT in {s.section for s in self.card.sections}

    def test_has_shadow_section(self) -> None:
        assert CardSection.SHADOW in {s.section for s in self.card.sections}

    def test_has_keywords_section(self) -> None:
        assert CardSection.KEYWORDS in {s.section for s in self.card.sections}

    def test_no_duplicate_sections(self) -> None:
        keys = [s.section for s in self.card.sections]
        assert len(keys) == len(set(keys))

    def test_model_validates(self) -> None:
        reloaded = Card.model_validate_json(self.card.model_dump_json())
        assert reloaded.suit == Suit.WANDS


# ---------------------------------------------------------------------------
# parse_card_page — error handling
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestParseCardErrors:
    def test_missing_entry_content_raises(self) -> None:
        minimal_html = "<html><body><div class='other'>no content here</div></body></html>"
        with pytest.raises(ValueError, match="entry-content"):
            parse_card_page(minimal_html, "0-the-fool")

    def test_empty_html_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_card_page("", "0-the-fool")


# ---------------------------------------------------------------------------
# parse_spread_page — New Moon spread
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestParseNewMoonSpread:
    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self.html = _html("spread-new-moon.html")
        self.spread = parse_spread_page(
            self.html, "new-moon-three-card", "New Moon Three-Card Spread"
        )

    def test_spread_id(self) -> None:
        assert self.spread.id == "new-moon-three-card"

    def test_spread_name(self) -> None:
        assert self.spread.name == "New Moon Three-Card Spread"

    def test_has_three_positions(self) -> None:
        assert len(self.spread.positions) == 3

    def test_positions_are_zero_based(self) -> None:
        indices = [p.index for p in self.spread.positions]
        assert sorted(indices) == [0, 1, 2]

    def test_positions_contiguous(self) -> None:
        indices = sorted(p.index for p in self.spread.positions)
        assert indices == list(range(len(self.spread.positions)))

    def test_first_position_name(self) -> None:
        pos = self.spread.position_by_index(0)
        assert len(pos.name) > 0

    def test_all_positions_have_meaning(self) -> None:
        for pos in self.spread.positions:
            assert pos.meaning.strip()

    def test_all_positions_have_source_url(self) -> None:
        for pos in self.spread.positions:
            assert "thothreadings.com" in str(pos.source_url)

    def test_model_validates(self) -> None:
        reloaded = Spread.model_validate_json(self.spread.model_dump_json())
        assert reloaded.id == self.spread.id

    def test_matches_golden_json(self) -> None:
        golden = _golden("spreads", "new-moon-three-card.json")
        assert golden["id"] == self.spread.id
        assert len(golden["positions"]) == len(self.spread.positions)


# ---------------------------------------------------------------------------
# parse_spread_page — error handling
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestParseSpreadErrors:
    def test_missing_entry_content_raises(self) -> None:
        minimal_html = "<html><body><p>nothing here</p></body></html>"
        with pytest.raises(ValueError, match="entry-content"):
            parse_spread_page(minimal_html, "test", "Test")

    def test_no_position_headings_raises(self) -> None:
        html = '<html><body><div class="entry-content"><p>no cards here</p></div></body></html>'
        with pytest.raises(ValueError, match="No spread positions"):
            parse_spread_page(html, "test", "Test")


# ---------------------------------------------------------------------------
# Scraper helpers — load_slugs
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLoadSlugs:
    def test_loads_non_empty_non_comment_lines(self, tmp_path: Path) -> None:
        seeds = tmp_path / "seeds.txt"
        seeds.write_text("# comment\nthe-fool\n\nace-of-wands\n", encoding="utf-8")
        result = load_slugs(seeds)
        assert result == ["the-fool", "ace-of-wands"]

    def test_empty_file_returns_empty_list(self, tmp_path: Path) -> None:
        seeds = tmp_path / "seeds.txt"
        seeds.write_text("# just comments\n\n", encoding="utf-8")
        result = load_slugs(seeds)
        assert result == []

    def test_spread_prefix_preserved_in_raw_list(self, tmp_path: Path) -> None:
        seeds = tmp_path / "seeds.txt"
        seeds.write_text("spread:spread-new-moon\n", encoding="utf-8")
        result = load_slugs(seeds)
        assert "spread:spread-new-moon" in result


# ---------------------------------------------------------------------------
# Scraper — URL building
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildUrl:
    def test_card_uses_blog_prefix(self) -> None:
        url = _build_url("0-the-fool")
        assert url == "https://thothreadings.com/blog/0-the-fool/"

    def test_spread_uses_root(self) -> None:
        url = _build_url("spread-new-moon")
        assert url == "https://thothreadings.com/spread-new-moon/"

    def test_minor_card_uses_blog_prefix(self) -> None:
        url = _build_url("ace-of-wands")
        assert "blog" in url


# ---------------------------------------------------------------------------
# Scraper — cache behaviour (no live HTTP)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_scraper_reads_from_cache(tmp_path: Path) -> None:
    """Scraper returns cached HTML without making HTTP calls."""
    cached_html = "<html><body>Cached content</body></html>"
    cache_file = tmp_path / "0-the-fool.html"
    cache_file.write_text(cached_html, encoding="utf-8")

    class _NeverCalledClient:
        async def get(self, url: str) -> None:  # pragma: no cover  # noqa: ARG002
            raise AssertionError("HTTP should not be called when cache exists")

    result = asyncio.run(
        fetch_page(
            _NeverCalledClient(),  # type: ignore[arg-type]
            "0-the-fool",
            tmp_path,
            refresh=False,
        )
    )
    assert result == cached_html


@pytest.mark.unit
def test_scraper_strips_spread_prefix() -> None:
    """Scraper normalises 'spread:slug' lines by stripping the prefix."""
    fake_html = "<html><body>Spread page</body></html>"

    async def _mock_fetch(
        client: object,  # noqa: ARG001
        slug: str,  # noqa: ARG001
        cache_dir: object,  # noqa: ARG001
        *,
        refresh: bool,  # noqa: ARG001
    ) -> str:
        return fake_html

    original = scrape_mod.fetch_page
    scrape_mod.fetch_page = _mock_fetch  # type: ignore[assignment]
    try:
        results = asyncio.run(scrape_slugs(["spread:spread-new-moon"], MagicMock(), refresh=False))
    finally:
        scrape_mod.fetch_page = original

    assert "spread-new-moon" in results
    assert results["spread-new-moon"] == fake_html


@pytest.mark.unit
def test_scraper_skips_comment_and_blank_lines() -> None:
    """Comment lines and blank lines are not included in scrape results."""

    async def _mock_fetch(
        client: object,  # noqa: ARG001
        slug: str,  # noqa: ARG001
        cache_dir: object,  # noqa: ARG001
        *,
        refresh: bool,  # noqa: ARG001
    ) -> str:
        return "<html/>"

    original = scrape_mod.fetch_page
    scrape_mod.fetch_page = _mock_fetch  # type: ignore[assignment]
    try:
        results = asyncio.run(
            scrape_slugs(["# comment", "", "  ", "0-the-fool"], MagicMock(), refresh=False)
        )
    finally:
        scrape_mod.fetch_page = original

    assert list(results.keys()) == ["0-the-fool"]


@pytest.mark.unit
def test_fetch_page_refresh_overwrites_cache(tmp_path: Path) -> None:
    """When refresh=True, the cached file is overwritten with fresh content."""
    old_html = "<html>old</html>"
    new_html = "<html>new</html>"
    cache_file = tmp_path / "0-the-fool.html"
    cache_file.write_text(old_html, encoding="utf-8")

    class _FakeResponse:
        text = new_html

        def raise_for_status(self) -> None:
            pass

    class _FakeClient:
        async def get(self, url: str) -> _FakeResponse:  # noqa: ARG002
            return _FakeResponse()

    result = asyncio.run(
        fetch_page(_FakeClient(), "0-the-fool", tmp_path, refresh=True)  # type: ignore[arg-type]
    )
    assert result == new_html
    assert cache_file.read_text(encoding="utf-8") == new_html


@pytest.mark.unit
def test_config_settings_defaults() -> None:
    """Settings loads without error and exposes expected default fields."""
    s = Settings()
    assert "127.0.0.1" in s.openai_base_url
    assert s.openai_api_key == "sk-no-key"
    assert s.chat_model == "local-model"
    assert "bge-small" in s.embedding_model

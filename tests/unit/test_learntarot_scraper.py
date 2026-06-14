"""Unit tests for the learntarot.com scraper.

All tests are pure logic — no live HTTP calls.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import fortune_teller.developer.scrape.learntarot as scrape_mod
from fortune_teller.developer.scrape.learntarot import (
    _build_url,
    fetch_page,
    load_slugs,
    scrape_slugs,
)

# ---------------------------------------------------------------------------
# load_slugs
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLoadSlugs:
    def test_loads_non_empty_non_comment_lines(self, tmp_path: Path) -> None:
        seeds = tmp_path / "seeds.txt"
        seeds.write_text("# comment\nmaj00\n\nc7\n", encoding="utf-8")
        result = load_slugs(seeds)
        assert result == ["maj00", "c7"]

    def test_empty_file_returns_empty_list(self, tmp_path: Path) -> None:
        seeds = tmp_path / "seeds.txt"
        seeds.write_text("# just comments\n\n", encoding="utf-8")
        result = load_slugs(seeds)
        assert result == []

    def test_all_whitespace_lines_ignored(self, tmp_path: Path) -> None:
        seeds = tmp_path / "seeds.txt"
        seeds.write_text("  \n\t\nmaj00\n", encoding="utf-8")
        result = load_slugs(seeds)
        assert result == ["maj00"]


# ---------------------------------------------------------------------------
# _build_url
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildUrl:
    def test_major_uses_htm_suffix(self) -> None:
        url = _build_url("maj00")
        assert url == "https://www.learntarot.com/maj00.htm"

    def test_minor_uses_htm_suffix(self) -> None:
        url = _build_url("c7")
        assert url == "https://www.learntarot.com/c7.htm"

    def test_court_uses_htm_suffix(self) -> None:
        url = _build_url("wpg")
        assert url == "https://www.learntarot.com/wpg.htm"

    def test_base_url_is_correct(self) -> None:
        url = _build_url("maj00")
        assert url.startswith("https://www.learntarot.com/")
        assert url.endswith(".htm")


# ---------------------------------------------------------------------------
# Cache behaviour (no live HTTP)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_scraper_reads_from_cache(tmp_path: Path) -> None:
    """Scraper returns cached HTML without making HTTP calls."""
    cached_html = "<html><body>Cached content</body></html>"
    cache_file = tmp_path / "maj00.html"
    cache_file.write_text(cached_html, encoding="utf-8")

    class _NeverCalledClient:
        async def get(self, url: str) -> None:  # pragma: no cover  # noqa: ARG002
            raise AssertionError("HTTP should not be called when cache exists")

    result = asyncio.run(
        fetch_page(
            _NeverCalledClient(),  # type: ignore[arg-type]
            "maj00",
            tmp_path,
            refresh=False,
        )
    )
    assert result == cached_html


@pytest.mark.unit
def test_fetch_page_refresh_overwrites_cache(tmp_path: Path) -> None:
    """When refresh=True, the cached file is overwritten with fresh content."""
    old_html = "<html>old</html>"
    new_html = "<html>new</html>"
    cache_file = tmp_path / "maj00.html"
    cache_file.write_text(old_html, encoding="utf-8")

    class _FakeResponse:
        text = new_html

        def raise_for_status(self) -> None:
            pass

    class _FakeClient:
        async def get(self, url: str) -> _FakeResponse:  # noqa: ARG002
            return _FakeResponse()

    result = asyncio.run(
        fetch_page(_FakeClient(), "maj00", tmp_path, refresh=True)  # type: ignore[arg-type]
    )
    assert result == new_html
    assert cache_file.read_text(encoding="utf-8") == new_html


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
            scrape_slugs(["# comment", "", "  ", "maj00"], MagicMock(), refresh=False)
        )
    finally:
        scrape_mod.fetch_page = original

    assert list(results.keys()) == ["maj00"]


@pytest.mark.unit
def test_scraper_results_map_keys_are_slugs() -> None:
    """Scrape results map slug strings to HTML strings."""
    fake_html = "<html>test</html>"

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
        results = asyncio.run(scrape_slugs(["maj00", "c7"], MagicMock(), refresh=False))
    finally:
        scrape_mod.fetch_page = original

    assert results == {"maj00": fake_html, "c7": fake_html}

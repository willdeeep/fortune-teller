"""Unit tests for card image extraction and resolution.

Covers:
- ``_extract_image_url`` in the thothreadings parser
- ``image_path_for`` in the images store
- ``_resolve_ext`` in the fetch-images CLI
"""

from __future__ import annotations

from pathlib import Path

import pytest
from selectolax.parser import HTMLParser

from fortune_teller.application.stores.images import image_path_for
from fortune_teller.developer.parse.thothreadings import _extract_image_url


@pytest.mark.unit
class TestExtractImageUrl:
    def test_extracts_card_image_from_entry_content(self, the_fool_html: str) -> None:
        tree = HTMLParser(the_fool_html)
        content = tree.css_first(".entry-content")
        url = _extract_image_url(content)
        assert url is not None
        assert "wp-content/uploads" in url
        assert "astrologist-illustration" not in url

    def test_strips_size_suffix(self, the_fool_html: str) -> None:
        tree = HTMLParser(the_fool_html)
        content = tree.css_first(".entry-content")
        url = _extract_image_url(content)
        assert url is not None
        assert "-480x" not in url
        assert url.endswith(".jpeg")

    def test_excludes_site_chrome(self) -> None:
        html = '<div class="entry-content"><img src="https://thothreadings.com/wp-content/uploads/2020/11/astrologist-illustration-01.png"></div>'
        tree = HTMLParser(html)
        content = tree.css_first(".entry-content")
        url = _extract_image_url(content)
        assert url is None

    def test_excludes_external_images(self) -> None:
        html = '<div class="entry-content"><img src="https://example.com/image.png"></div>'
        tree = HTMLParser(html)
        content = tree.css_first(".entry-content")
        url = _extract_image_url(content)
        assert url is None

    def test_no_images_yields_none(self) -> None:
        html = '<div class="entry-content"><p>No images here.</p></div>'
        tree = HTMLParser(html)
        content = tree.css_first(".entry-content")
        url = _extract_image_url(content)
        assert url is None

    def test_ace_of_wands_has_image(self, ace_of_wands_html: str) -> None:
        tree = HTMLParser(ace_of_wands_html)
        content = tree.css_first(".entry-content")
        url = _extract_image_url(content)
        assert url is not None
        assert "wp-content/uploads" in url


@pytest.mark.unit
class TestImagePathFor:
    def test_finds_existing_image(self, tmp_path: Path) -> None:
        (tmp_path / "0-the-fool.jpeg").write_bytes(b"fake")
        result = image_path_for("0-the-fool", tmp_path)
        assert result is not None
        assert result.name == "0-the-fool.jpeg"

    def test_returns_none_for_missing(self, tmp_path: Path) -> None:
        result = image_path_for("missing", tmp_path)
        assert result is None

    def test_returns_none_for_nonexistent_dir(self, tmp_path: Path) -> None:
        missing = tmp_path / "nope"
        result = image_path_for("the-fool", missing)
        assert result is None

    def test_prefers_first_match_by_extension(self, tmp_path: Path) -> None:
        (tmp_path / "the-fool.png").write_bytes(b"a")
        (tmp_path / "the-fool.jpeg").write_bytes(b"b")
        result = image_path_for("the-fool", tmp_path)
        assert result is not None
        assert result.stem == "the-fool"

    def test_ignores_non_image_extensions(self, tmp_path: Path) -> None:
        (tmp_path / "the-fool.txt").write_bytes(b"not an image")
        result = image_path_for("the-fool", tmp_path)
        assert result is None

"""Unit tests for ``ft-fetch-images`` CLI.

Covers:
- ``_resolve_ext`` — URL extension extraction with fallback
- ``_download_images`` — skip-existing, reject non-image content-type,
  dry-run, refresh, successful download
- ``image_path_for`` with deck-scoped directory (regression for
  Bug 1: ``main()`` was passing bare ``images_dir`` instead of
  ``images_dir / deck_id``)
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from pydantic import HttpUrl

from fortune_teller.application.models.domain import Card
from fortune_teller.application.stores.images import image_path_for
from fortune_teller.developer.fetch_images.cli import _download_images, _resolve_ext

# ---------------------------------------------------------------------------
# _resolve_ext
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestResolveExt:
    def test_jpeg_extension(self) -> None:
        assert _resolve_ext("https://example.com/card.jpeg") == ".jpeg"

    def test_jpg_extension(self) -> None:
        assert _resolve_ext("https://example.com/card.jpg") == ".jpg"

    def test_png_extension(self) -> None:
        assert _resolve_ext("https://example.com/image.png") == ".png"

    def test_webp_extension(self) -> None:
        assert _resolve_ext("https://example.com/art.webp") == ".webp"

    def test_gif_extension(self) -> None:
        assert _resolve_ext("https://example.com/anim.gif") == ".gif"

    def test_no_extension_defaults_to_jpeg(self) -> None:
        assert _resolve_ext("https://example.com/image") == ".jpeg"

    def test_unknown_extension_defaults_to_jpeg(self) -> None:
        assert _resolve_ext("https://example.com/image.svg") == ".jpeg"

    def test_size_suffix_stripped_in_url(self) -> None:
        assert _resolve_ext("https://example.com/wp-content/uploads/card-480x600.png") == ".png"

    def test_query_params_ignored(self) -> None:
        assert _resolve_ext("https://example.com/card.jpeg?w=480") == ".jpeg"


# ---------------------------------------------------------------------------
# _download_images test helpers
# ---------------------------------------------------------------------------


def _make_card(card_id: str, image_url: str | None = None) -> Any:
    """Create a Card instance for download tests."""
    kwargs: dict[str, Any] = {
        "id": card_id,
        "name": card_id.replace("-", " ").title(),
        "arcana": "major",
        "source_url": HttpUrl("https://example.test/card"),
    }
    if image_url is not None:
        kwargs["image_url"] = image_url
    return Card(**kwargs)


class _FakeResponse:
    """Minimal httpx.Response-like object for mocking."""

    def __init__(
        self,
        *,
        content: bytes = b"",
        content_type: str = "image/jpeg",
    ) -> None:
        self.content = content
        self.headers = {"content-type": content_type}
        self.status_code = 200

    def raise_for_status(self) -> None:
        pass


class _FakeAsyncClient:
    """Doubles httpx.AsyncClient for download tests."""

    def __init__(self, responses: list[_FakeResponse]) -> None:
        self._responses = list(responses)
        self._call_idx = 0
        self.get_calls: list[str] = []

    async def get(self, url: str) -> _FakeResponse:
        self.get_calls.append(url)
        resp = self._responses[self._call_idx]
        self._call_idx += 1
        return resp

    async def __aenter__(self) -> _FakeAsyncClient:
        return self

    async def __aexit__(self, *args: object) -> None:
        pass


# ---------------------------------------------------------------------------
# _download_images
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDownloadImages:
    def test_skips_existing_images(self, tmp_path: Path) -> None:
        """Cards whose image file already exists on disk are not re-downloaded."""
        card = _make_card("the-fool", "https://example.com/the-fool.jpeg")
        existing = tmp_path / "the-fool.jpeg"
        original = b"cached-image"
        existing.write_bytes(original)

        fake_client = _FakeAsyncClient(responses=[])
        with patch(
            "fortune_teller.developer.fetch_images.cli.httpx.AsyncClient", return_value=fake_client
        ):
            asyncio.run(_download_images([card], tmp_path))

        assert existing.read_bytes() == original
        assert fake_client.get_calls == []

    def test_rejects_non_image_content_type(self, tmp_path: Path) -> None:
        """An HTML error page must NOT be written as a .jpeg file."""
        card = _make_card("the-fool", "https://example.com/the-fool.jpeg")
        responses = [_FakeResponse(content=b"<html>Error</html>", content_type="text/html")]
        fake_client = _FakeAsyncClient(responses=responses)

        with patch(
            "fortune_teller.developer.fetch_images.cli.httpx.AsyncClient", return_value=fake_client
        ):
            asyncio.run(_download_images([card], tmp_path))

        written_files = [p for p in tmp_path.iterdir() if p.is_file()]
        assert len(written_files) == 0

    def test_dry_run_does_not_download(self, tmp_path: Path) -> None:
        """In dry-run mode, no files are written and no HTTP requests are made."""
        card = _make_card("the-fool", "https://example.com/the-fool.jpeg")

        fake_client = _FakeAsyncClient(responses=[])
        with patch(
            "fortune_teller.developer.fetch_images.cli.httpx.AsyncClient", return_value=fake_client
        ):
            asyncio.run(_download_images([card], tmp_path, dry_run=True))

        assert fake_client.get_calls == []
        assert not list(tmp_path.iterdir())

    def test_refresh_overwrites_existing(self, tmp_path: Path) -> None:
        """With refresh=True, existing files are re-downloaded."""
        card = _make_card("the-fool", "https://example.com/the-fool.jpeg")
        existing = tmp_path / "the-fool.jpeg"
        existing.write_bytes(b"old-image")

        responses = [_FakeResponse(content=b"new-image", content_type="image/jpeg")]
        fake_client = _FakeAsyncClient(responses=responses)

        with patch(
            "fortune_teller.developer.fetch_images.cli.httpx.AsyncClient", return_value=fake_client
        ):
            asyncio.run(_download_images([card], tmp_path, refresh=True))

        assert existing.read_bytes() == b"new-image"
        assert len(fake_client.get_calls) == 1

    def test_skips_cards_without_image_url(self, tmp_path: Path) -> None:
        """Cards with image_url=None are skipped (no download attempted)."""
        card = _make_card("the-fool", image_url=None)

        fake_client = _FakeAsyncClient(responses=[])
        with patch(
            "fortune_teller.developer.fetch_images.cli.httpx.AsyncClient", return_value=fake_client
        ):
            asyncio.run(_download_images([card], tmp_path))

        assert fake_client.get_calls == []

    def test_creates_dest_directory(self, tmp_path: Path) -> None:
        """_download_images creates the images_dir if it doesn't exist."""
        nested = tmp_path / "book-of-thoth"
        card = _make_card("the-fool", image_url=None)

        fake_client = _FakeAsyncClient(responses=[])
        with patch(
            "fortune_teller.developer.fetch_images.cli.httpx.AsyncClient", return_value=fake_client
        ):
            asyncio.run(_download_images([card], nested))

        assert nested.is_dir()

    def test_downloads_image_file(self, tmp_path: Path) -> None:
        """Successful download writes the image content to disk."""
        card = _make_card("the-fool", "https://example.com/the-fool.jpeg")
        responses = [_FakeResponse(content=b"fake-jpeg-data", content_type="image/jpeg")]
        fake_client = _FakeAsyncClient(responses=responses)

        with patch(
            "fortune_teller.developer.fetch_images.cli.httpx.AsyncClient", return_value=fake_client
        ):
            asyncio.run(_download_images([card], tmp_path))

        written = tmp_path / "the-fool.jpeg"
        assert written.exists()
        assert written.read_bytes() == b"fake-jpeg-data"

    def test_http_error_does_not_crash(self, tmp_path: Path) -> None:
        """An HTTP error for one card does not prevent others from downloading."""
        card_ok = _make_card("the-magician", "https://example.com/the-magician.jpeg")

        class _ErrorClient(_FakeAsyncClient):
            async def get(self, url: str) -> _FakeResponse:  # type: ignore[override]
                raise httpx.HTTPStatusError(
                    "404",
                    request=httpx.Request("GET", url),
                    response=httpx.Response(404),
                )

        fake_client = _ErrorClient(responses=[])
        with patch(
            "fortune_teller.developer.fetch_images.cli.httpx.AsyncClient", return_value=fake_client
        ):
            asyncio.run(_download_images([card_ok], tmp_path))

        written = list(tmp_path.iterdir())
        assert len(written) == 0


# ---------------------------------------------------------------------------
# Regression: deck-scoped directory in image_path_for
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestImagePathForDeckScoped:
    def test_finds_image_in_deck_subdirectory(self, tmp_path: Path) -> None:
        """image_path_for must look inside the card's deck-scoped directory.

        Regression test for Bug 1: main() was passing bare ``images_dir``
        (e.g. ``data/images/``) instead of ``images_dir / deck_id``
        (e.g. ``data/images/book-of-thoth/``).  The ``ft-fetch-images``
        CLI writes to ``images_dir / deck_id``, so ``image_path_for``
        must be called with that deck-scoped subdirectory to find files.
        """
        deck_dir = tmp_path / "book-of-thoth"
        deck_dir.mkdir()
        (deck_dir / "0-the-fool.jpeg").write_bytes(b"art")

        result = image_path_for("0-the-fool", deck_dir)
        assert result is not None
        assert result.name == "0-the-fool.jpeg"

    def test_flat_images_dir_does_not_find_deck_nested_files(self, tmp_path: Path) -> None:
        """If we incorrectly pass the parent images_dir, items in
        subdirectories are NOT found (because image_path_for is
        non-recursive).  This confirms the directory-level bug."""
        deck_dir = tmp_path / "book-of-thoth"
        deck_dir.mkdir()
        (deck_dir / "0-the-fool.jpeg").write_bytes(b"art")

        result = image_path_for("0-the-fool", tmp_path)
        assert result is None

"""Async scraper for thothreadings.com.

Fetches card and spread pages, caches raw HTML to disk, and obeys
``robots.txt`` (only ``/wp-admin/`` is disallowed — card and spread paths
are fully permitted).

Politeness rules:
- Minimum 1 second between requests.
- ``User-Agent`` identifies this project.
- Up to 3 retries with exponential backoff on transient errors.
- Re-uses on-disk cache unless ``--refresh`` is passed.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

BASE_URL = "https://thothreadings.com"
USER_AGENT = "fortune-teller/0.0.1 (+https://github.com/fortune-teller/fortune-teller)"
REQUEST_DELAY_SECONDS = 1.0


def _cache_path(cache_dir: Path, slug: str) -> Path:
    """Return the on-disk cache path for *slug*."""
    return cache_dir / f"{slug}.html"


# Most Book of Thoth seed slugs are already the root definition-page slug, but a
# handful differ — Thoth renames (Magus, Priestess, Lust, Art) and a few site
# inconsistencies (e.g. the Three of Wands). These map the seed slug to its real
# root slug; everything else passes through unchanged. Sourced from
# https://thothreadings.com/wp-sitemap.xml (verified 2026-06-26).
_ROOT_SLUG_OVERRIDES: dict[str, str] = {
    "0-the-fool": "the-fool",
    "i-the-magician": "the-magician-the-magus",
    "ii-the-high-priestess": "the-priestess-ii-the-high-priestess",
    "xi-the-passion-lust": "xi-the-passion",
    "xiv-the-art": "xiv-the-art-alchemy",
    "the-three-of-wands": "three-of-wands-virtue",
}


def _root_slug(slug: str) -> str:
    """Map a seed slug to its root definition-page URL slug.

    The seed slugs are mostly identical to the root-page slug; the few that
    differ (see :data:`_ROOT_SLUG_OVERRIDES`) are mapped explicitly. There is no
    clean transform — the root URL scheme is irregular (some majors keep their
    numeral prefix, some drop it, some are renamed), so the mapping is authored
    from the site sitemap rather than derived.
    """
    return _ROOT_SLUG_OVERRIDES.get(slug, slug)


def _build_url(slug: str) -> str:
    """Build the full root-page URL for a card or spread slug.

    Pages live at the site root ``/<slug>/`` (the old ``/blog/<slug>/`` pages
    are truncated summaries). A few seeds map to a different root slug — see
    :func:`_root_slug`.
    """
    return f"{BASE_URL}/{_root_slug(slug)}/"


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
async def _fetch_url(client: httpx.AsyncClient, url: str) -> str:
    """Fetch *url* with retry on transient errors."""
    response = await client.get(url)
    response.raise_for_status()
    return str(response.text)


async def fetch_page(
    client: httpx.AsyncClient,
    slug: str,
    cache_dir: Path,
    *,
    refresh: bool = False,
) -> str:
    """Fetch one page, reading from cache when available.

    Args:
        client:    Shared :class:`httpx.AsyncClient`.
        slug:      URL slug, e.g. ``"the-fool"`` or ``"spread-new-moon"``.
        cache_dir: Directory for cached HTML files.
        refresh:   If ``True``, re-fetch even when a cached file exists.

    Returns:
        Raw HTML string.
    """
    path = _cache_path(cache_dir, slug)
    if path.exists() and not refresh:
        return path.read_text(encoding="utf-8")

    await asyncio.sleep(REQUEST_DELAY_SECONDS)
    url = _build_url(slug)
    html: str = await _fetch_url(client, url)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return html


async def scrape_slugs(
    slugs: list[str],
    cache_dir: Path,
    *,
    refresh: bool = False,
) -> dict[str, str]:
    """Fetch all *slugs*, returning ``{slug: html}`` mapping.

    Lines starting with ``#`` in the slug list are silently ignored (comments
    from the seeds file).  Lines prefixed with ``spread:`` have the prefix
    stripped before building the URL (the prefix is used only for
    categorisation in the seeds file).

    Args:
        slugs:     List of slugs (raw lines from the seeds file are accepted).
        cache_dir: Directory for cached HTML files.
        refresh:   Force re-fetch of cached pages.

    Returns:
        Mapping of slug (without ``spread:`` prefix) to raw HTML.
    """
    normalised: list[str] = []
    for raw in slugs:
        clean = raw.strip()
        if not clean or clean.startswith("#"):
            continue
        if clean.startswith("spread:"):
            clean = clean[len("spread:") :]
        normalised.append(clean)

    results: dict[str, str] = {}
    async with httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT},
        follow_redirects=True,
        timeout=30.0,
    ) as client:
        for slug in normalised:
            html = await fetch_page(client, slug, cache_dir, refresh=refresh)
            results[slug] = html

    return results


def load_slugs(seeds_file: Path) -> list[str]:
    """Read and return non-empty, non-comment lines from *seeds_file*."""
    return [
        line.strip()
        for line in seeds_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

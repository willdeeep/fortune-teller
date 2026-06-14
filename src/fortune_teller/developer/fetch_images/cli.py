"""CLI entry point: ``ft-fetch-images``.

Usage::

    uv run ft-fetch-images               # download missing card images
    uv run ft-fetch-images --refresh      # re-download all images
    uv run ft-fetch-images --dry-run      # list cards + resolved image URLs

Images are stored in ``<FT_DATA_DIR>/images/<deck_id>/<card_id>.<ext>``.
Cards with ``image_url is None`` are skipped and reported.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from urllib.parse import urlparse

import httpx
import typer
from rich.console import Console

from fortune_teller.application.config import settings
from fortune_teller.application.models.domain import Card
from fortune_teller.application.services.loading import load_deck

app = typer.Typer(add_completion=False)
console = Console()


def _resolve_ext(url: str) -> str:
    """Extract file extension from a URL, defaulting to ``.jpeg``."""
    path = urlparse(url).path
    if "." in path.split("/")[-1]:
        ext = "." + path.split("/")[-1].rsplit(".", 1)[-1].lower()
        if ext in (".jpeg", ".jpg", ".png", ".webp", ".gif"):
            return ext
    return ".jpeg"


async def _download_images(
    cards: list[Card],
    images_dir: Path,
    *,
    refresh: bool = False,
    dry_run: bool = False,
) -> None:
    """Download card images for cards with an ``image_url``."""
    images_dir.mkdir(parents=True, exist_ok=True)
    skipped = 0
    downloaded = 0
    cached = 0

    async with httpx.AsyncClient(
        headers={"User-Agent": "fortune-teller/0.3.0 (developer tool)"},
        follow_redirects=True,
        timeout=30.0,
    ) as client:
        for card in cards:
            if card.image_url is None:
                console.print(f"  [dim]{card.id}: no image_url, skipping[/dim]")
                skipped += 1
                continue

            ext = _resolve_ext(card.image_url)
            dest = images_dir / f"{card.id}{ext}"

            if dest.exists() and not refresh:
                console.print(f"  [dim]{card.id}: cached ({dest.name})[/dim]")
                cached += 1
                continue

            if dry_run:
                console.print(f"  {card.id}: {card.image_url}")
                continue

            try:
                response = await client.get(card.image_url)
                response.raise_for_status()
                content_type = response.headers.get("content-type", "")
                if not content_type.startswith("image/"):
                    console.print(
                        f"  [red]{card.id}: rejected (Content-Type: {content_type})[/red]"
                    )
                    continue
                dest.write_bytes(response.content)
                console.print(f"  [green]{card.id}: downloaded → {dest.name}[/green]")
                downloaded += 1
            except httpx.HTTPError as exc:
                console.print(f"  [red]{card.id}: HTTP error — {exc}[/red]")

    console.print(
        f"\n[bold]Summary:[/bold] {downloaded} downloaded, "
        f"{cached} cached, {skipped} skipped (no image_url)."
    )


@app.command()
def main(
    refresh: bool = typer.Option(False, "--refresh", help="Re-download all images."),
    dry_run: bool = typer.Option(False, "--dry-run", help="List cards + URLs without downloading."),
    deck_id: str = typer.Option("book-of-thoth", "--deck", help="Deck ID to fetch images for."),
) -> None:
    """Download card artwork images from the source site."""
    parsed_dir = settings.ft_data_dir / "parsed"
    deck = load_deck(parsed_dir, deck_id)
    images_dir = settings.images_dir / deck_id

    console.print(f"[bold]Fetching images for {deck.name} ({len(deck.cards)} cards)[/bold]")
    console.print(f"  Images dir: {images_dir}")

    asyncio.run(_download_images(deck.cards, images_dir, refresh=refresh, dry_run=dry_run))

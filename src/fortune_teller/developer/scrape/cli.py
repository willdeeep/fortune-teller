"""CLI entry point: ``ft-scrape``.

Usage::

    uv run ft-scrape                  # scrape all Book of Thoth slugs
    uv run ft-scrape --refresh        # force re-fetch, ignoring cache
    uv run ft-scrape --dry-run        # list slugs without fetching

Cached HTML is written to ``<FT_DATA_DIR>/cache/thothreadings.com/``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.console import Console

from fortune_teller.application.config import settings
from fortune_teller.developer.scrape.thothreadings import load_slugs, scrape_slugs

app = typer.Typer(add_completion=False)
console = Console()

_SEEDS_FILE = Path(__file__).parent / "seeds" / "book_of_thoth.txt"


_DEFAULT_SEEDS = str(_SEEDS_FILE)  # string keeps typer happy (avoids B008)


@app.command()
def main(
    refresh: bool = typer.Option(False, "--refresh", help="Force re-fetch cached pages."),
    dry_run: bool = typer.Option(False, "--dry-run", help="List slugs without fetching."),
    seeds: Path = typer.Option(_DEFAULT_SEEDS, "--seeds", help="Path to slugs seed file."),  # noqa: B008
) -> None:
    """Scrape card and spread pages from thothreadings.com."""
    cache_dir = settings.ft_data_dir / "cache" / "thothreadings.com"

    slugs = load_slugs(seeds)
    console.print(f"[bold]Loaded {len(slugs)} slugs from {seeds.name}[/bold]")

    if dry_run:
        for slug in slugs:
            console.print(f"  {slug}")
        return

    cache_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"Cache dir: {cache_dir}")

    results = asyncio.run(scrape_slugs(slugs, cache_dir, refresh=refresh))
    console.print(f"[green]Scraped {len(results)} pages.[/green]")

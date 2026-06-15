"""CLI entry point: ``ft-scrape``.

Usage::

    uv run ft-scrape                       # scrape all Book of Thoth slugs
    uv run ft-scrape --refresh             # force re-fetch, ignoring cache
    uv run ft-scrape --dry-run             # list slugs without fetching
    uv run ft-scrape --source learntarot   # scrape Rider-Waite from learntarot.com

Cached HTML is written to ``<FT_DATA_DIR>/cache/<source>.com/``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.console import Console

from fortune_teller.application.config import settings

app = typer.Typer(add_completion=False)
console = Console()

_SEEDS_DIR = Path(__file__).parent / "seeds"

_SOURCE_CONFIG: dict[str, tuple[str, str, str]] = {
    "thothreadings": (
        "thothreadings.com",
        "book_of_thoth.txt",
        "fortune_teller.developer.scrape.thothreadings",
    ),
    "learntarot": (
        "learntarot.com",
        "rider_waite.txt",
        "fortune_teller.developer.scrape.learntarot",
    ),
}

_DEFAULT_SOURCE = "thothreadings"


@app.command()
def main(
    refresh: bool = typer.Option(False, "--refresh", help="Force re-fetch cached pages."),
    dry_run: bool = typer.Option(False, "--dry-run", help="List slugs without fetching."),
    seeds: Path | None = typer.Option(None, "--seeds", help="Path to slugs seed file."),  # noqa: B008
    source: str = typer.Option(_DEFAULT_SOURCE, "--source", help="Scraping source."),
) -> None:
    """Scrape card pages from the specified source."""
    source_cfg = _SOURCE_CONFIG.get(source)
    if source_cfg is None:
        console.print(f"[red]Unknown source: {source!r}. Choose from: {list(_SOURCE_CONFIG)}[/red]")
        raise typer.Exit(1)

    cache_name, seeds_name, module_path = source_cfg

    # Lazy import to keep CLI fast
    if module_path == "fortune_teller.developer.scrape.thothreadings":  # pragma: no cover
        from fortune_teller.developer.scrape.thothreadings import load_slugs, scrape_slugs  # noqa: PLC0415, I001
    else:
        from fortune_teller.developer.scrape.learntarot import load_slugs, scrape_slugs  # noqa: PLC0415, I001

    cache_dir = settings.ft_data_dir / "cache" / cache_name
    seeds_file = seeds or (_SEEDS_DIR / seeds_name)

    slugs = load_slugs(seeds_file)
    console.print(f"[bold]Loaded {len(slugs)} slugs from {seeds_file.name}[/bold]")

    if dry_run:
        for slug in slugs:
            console.print(f"  {slug}")
        return

    cache_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"Cache dir: {cache_dir}")

    results = asyncio.run(scrape_slugs(slugs, cache_dir, refresh=refresh))
    console.print(f"[green]Scraped {len(results)} pages.[/green]")

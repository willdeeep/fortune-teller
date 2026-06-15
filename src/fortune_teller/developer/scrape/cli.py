"""CLI entry point: ``ft-scrape``.

Usage::

    uv run ft-scrape                         # scrape ALL sources (Thoth + Rider-Waite)
    uv run ft-scrape --source thothreadings  # just the Book of Thoth
    uv run ft-scrape --source learntarot     # just the Rider-Waite deck
    uv run ft-scrape --refresh               # force re-fetch, ignoring cache
    uv run ft-scrape --dry-run               # list slugs without fetching

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

#: ``--source`` default. ``"all"`` scrapes every configured source.
_DEFAULT_SOURCE = "all"


def _scrape_source(
    source: str,
    *,
    refresh: bool,
    dry_run: bool,
    seeds: Path | None,
) -> None:
    """Scrape a single configured *source* (a key of :data:`_SOURCE_CONFIG`)."""
    cache_name, seeds_name, module_path = _SOURCE_CONFIG[source]

    # Lazy import to keep CLI start-up fast.
    if module_path == "fortune_teller.developer.scrape.thothreadings":  # pragma: no cover
        from fortune_teller.developer.scrape.thothreadings import load_slugs, scrape_slugs  # noqa: PLC0415, I001
    else:
        from fortune_teller.developer.scrape.learntarot import load_slugs, scrape_slugs  # noqa: PLC0415, I001

    cache_dir = settings.ft_data_dir / "cache" / cache_name
    seeds_file = seeds or (_SEEDS_DIR / seeds_name)

    slugs = load_slugs(seeds_file)
    console.print(f"[bold]{source} — loaded {len(slugs)} slugs from {seeds_file.name}[/bold]")

    if dry_run:
        for slug in slugs:
            console.print(f"  {slug}")
        return

    cache_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"  Cache dir: {cache_dir}")

    results = asyncio.run(scrape_slugs(slugs, cache_dir, refresh=refresh))
    console.print(f"[green]{source} — scraped {len(results)} pages.[/green]")


@app.command()
def main(
    refresh: bool = typer.Option(False, "--refresh", help="Force re-fetch cached pages."),
    dry_run: bool = typer.Option(False, "--dry-run", help="List slugs without fetching."),
    seeds: Path | None = typer.Option(  # noqa: B008
        None, "--seeds", help="Seed file override (requires a single --source)."
    ),
    source: str = typer.Option(
        _DEFAULT_SOURCE,
        "--source",
        help="Source: 'all' (default), 'thothreadings', or 'learntarot'.",
    ),
) -> None:
    """Scrape card pages. With no ``--source`` it scrapes every configured source."""
    if source == "all":
        if seeds is not None:
            console.print("[red]--seeds requires a single --source (not 'all').[/red]")
            raise typer.Exit(1)
        for src in _SOURCE_CONFIG:
            _scrape_source(src, refresh=refresh, dry_run=dry_run, seeds=None)
        return

    if source not in _SOURCE_CONFIG:
        console.print(
            f"[red]Unknown source: {source!r}. Choose from: {['all', *_SOURCE_CONFIG]}[/red]"
        )
        raise typer.Exit(1)

    _scrape_source(source, refresh=refresh, dry_run=dry_run, seeds=seeds)

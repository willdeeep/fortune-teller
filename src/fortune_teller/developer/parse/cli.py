"""CLI entry point: ``ft-parse``.

Usage::

    uv run ft-parse                   # parse all cached HTML in data/cache/
    uv run ft-parse --deck book-of-thoth
"""

from __future__ import annotations

import typer
from rich.console import Console

from fortune_teller.application.config import settings
from fortune_teller.developer.parse.thothreadings import parse_card_page, parse_spread_page

app = typer.Typer(add_completion=False)
console = Console()

_SPREAD_SLUGS = {"spread-new-moon"}
_SPREAD_META: dict[str, tuple[str, str]] = {
    "spread-new-moon": ("new-moon-three-card", "New Moon Three-Card Spread"),
}


@app.command()
def main(
    deck: str = typer.Option("book-of-thoth", "--deck", help="Deck identifier."),
) -> None:
    """Parse cached HTML into structured card and spread JSON."""
    cache_dir = settings.ft_data_dir / "cache" / "thothreadings.com"
    card_out_dir = settings.ft_data_dir / "parsed" / deck
    spread_out_dir = settings.ft_data_dir / "parsed" / "spreads"
    card_out_dir.mkdir(parents=True, exist_ok=True)
    spread_out_dir.mkdir(parents=True, exist_ok=True)

    html_files = sorted(cache_dir.glob("*.html"))
    if not html_files:
        console.print(f"[red]No HTML files found in {cache_dir}[/red]")
        raise typer.Exit(1)

    cards_written = spreads_written = errors = 0

    for html_path in html_files:
        slug = html_path.stem
        html = html_path.read_text(encoding="utf-8")

        try:
            if slug in _SPREAD_SLUGS:
                spread_id, spread_name = _SPREAD_META[slug]
                spread = parse_spread_page(html, spread_id, spread_name)
                out = spread_out_dir / f"{spread_id}.json"
                out.write_text(spread.model_dump_json(indent=2), encoding="utf-8")
                spreads_written += 1
                console.print(f"  [cyan]spread[/cyan] {slug} -> {out.name}")
            else:
                card = parse_card_page(html, slug)
                out = card_out_dir / f"{slug}.json"
                out.write_text(card.model_dump_json(indent=2), encoding="utf-8")
                cards_written += 1
                console.print(f"  [green]card[/green]   {slug} -> {out.name}")
        except Exception as exc:
            console.print(f"  [red]ERROR[/red]  {slug}: {exc}")
            errors += 1

    console.print(
        f"\n[bold]Done.[/bold] {cards_written} cards, {spreads_written} spreads, {errors} errors."
    )
    if errors:
        raise typer.Exit(1)

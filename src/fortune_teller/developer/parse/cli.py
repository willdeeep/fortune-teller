"""CLI entry point: ``ft-parse``.

Usage::

    uv run ft-parse                           # parse all cached thothreadings HTML
    uv run ft-parse --deck book-of-thoth
    uv run ft-parse --source learntarot       # parse Rider-Waite from learntarot.com
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import typer
from rich.console import Console

from fortune_teller.application.config import settings

app = typer.Typer(add_completion=False)
console = Console()

_SPREAD_SLUGS = {"spread-new-moon"}
_SPREAD_META: dict[str, tuple[str, str]] = {
    "spread-new-moon": ("new-moon-three-card", "New Moon Three-Card Spread"),
}

_SOURCE_CONFIG: dict[str, tuple[str, str]] = {
    "thothreadings": ("thothreadings.com", "fortune_teller.developer.parse.thothreadings"),
    "learntarot": ("learntarot.com", "fortune_teller.developer.parse.learntarot"),
}

_DEFAULT_SOURCE = "thothreadings"


def _import_parser(
    source: str,
) -> tuple[Callable[..., Any], Callable[..., Any] | None]:
    """Lazy-import the parser module for *source*.

    Returns:
        ``(parse_card_page, parse_spread_page_or_None)``.
    """
    if source == "thothreadings":  # pragma: no cover
        from fortune_teller.developer.parse.thothreadings import (  # noqa: PLC0415, I001
            parse_card_page as _tr_parse,
            parse_spread_page as _tr_spread,
        )

        return cast("Callable[..., Any]", _tr_parse), cast("Callable[..., Any]", _tr_spread)

    from fortune_teller.developer.parse.learntarot import parse_card_page as _rw_parse  # noqa: PLC0415, I001

    return cast("Callable[..., Any]", _rw_parse), None


def _parse_one_card(
    html: str,
    slug: str,
    source: str,
    parse_card_page: Callable[..., Any],
    card_out_dir: Path,
) -> str | None:
    """Parse a single HTML file and write the output.

    Returns:
        The output filename on success, or ``None`` if the page was a spread.
    """
    card = parse_card_page(html, slug)
    if source == "learntarot":
        out = card_out_dir / f"{card.id}.json"
    else:
        out = card_out_dir / f"{slug}.json"
    out.write_text(card.model_dump_json(indent=2), encoding="utf-8")
    return out.name


def _parse_one_spread(
    html: str,
    slug: str,
    parse_spread_page: Callable[..., Any],
) -> str | None:
    """Parse a spread HTML file and write the output.

    Returns:
        The output filename on success.
    """
    spread_id, spread_name = _SPREAD_META[slug]
    spread = parse_spread_page(html, spread_id, spread_name)
    out = settings.ft_data_dir / "parsed" / "spreads" / f"{spread_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(spread.model_dump_json(indent=2), encoding="utf-8")
    return out.name


@app.command()
def main(
    deck: str = typer.Option("book-of-thoth", "--deck", help="Deck identifier."),
    source: str = typer.Option(_DEFAULT_SOURCE, "--source", help="Parsing source."),
) -> None:
    """Parse cached HTML into structured card JSON."""
    source_cfg = _SOURCE_CONFIG.get(source)
    if source_cfg is None:
        console.print(f"[red]Unknown source: {source!r}. Choose from: {list(_SOURCE_CONFIG)}[/red]")
        raise typer.Exit(1)

    cache_name, _module_path = source_cfg
    cache_dir = settings.ft_data_dir / "cache" / cache_name

    parse_card_page, parse_spread_page = _import_parser(source)

    if source == "learntarot":
        card_out_dir = settings.ft_data_dir / "raw" / "rider-waite"
    else:
        card_out_dir = settings.ft_data_dir / "parsed" / deck

    card_out_dir.mkdir(parents=True, exist_ok=True)

    html_files = sorted(cache_dir.glob("*.html"))
    if not html_files:
        console.print(f"[red]No HTML files found in {cache_dir}[/red]")
        raise typer.Exit(1)

    cards_written = errors = 0

    for html_path in html_files:
        slug = html_path.stem
        html = html_path.read_text(encoding="utf-8")

        try:
            if source == "thothreadings" and slug in _SPREAD_SLUGS:
                out_name = _parse_one_spread(html, slug, parse_spread_page)  # type: ignore[arg-type]
                console.print(f"  [cyan]spread[/cyan] {slug} -> {out_name}")
            else:
                out_name = _parse_one_card(html, slug, source, parse_card_page, card_out_dir)
                cards_written += 1
                console.print(f"  [green]card[/green]   {slug} -> {out_name}")
        except Exception as exc:
            console.print(f"  [red]ERROR[/red]  {slug}: {exc}")
            errors += 1

    console.print(f"\n[bold]Done.[/bold] {cards_written} cards, {errors} errors.")
    if errors:
        raise typer.Exit(1)

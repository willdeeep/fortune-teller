"""CLI entry point: ``ft-normalize-rw``.

Usage::

    uv run ft-normalize-rw                    # normalise all RW cards (API)
    uv run ft-normalize-rw --provider local   # use local llama-server
    uv run ft-normalize-rw --only the-fool,seven-of-cups  # re-run specific cards
"""

from __future__ import annotations

import typer
from rich.console import Console

from fortune_teller.application.config import settings
from fortune_teller.developer.normalize.rider_waite import build_normalize_model, normalize_deck

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    provider: str = typer.Option(
        "api", "--provider", help="LLM provider: 'api' (Claude) or 'local' (llama-server)."
    ),
    model: str = typer.Option("claude-sonnet-4-6", "--model", help="Model identifier."),
    only: str | None = typer.Option(None, "--only", help="Comma-separated card IDs to re-run."),
    no_llm: bool = typer.Option(False, "--no-llm", help="Run deterministic stage only (no LLM)."),
) -> None:
    raw_dir = settings.ft_data_dir / "raw" / "rider-waite"
    out_dir = settings.ft_data_dir / "parsed" / "rider-waite"

    if not raw_dir.is_dir():
        console.print(f"[red]Raw directory not found: {raw_dir}[/red]")
        console.print(
            "[dim]Run ft-scrape --source learntarot and ft-parse --source learntarot first.[/dim]"
        )
        raise typer.Exit(1)

    llm = None
    if not no_llm:
        llm = build_normalize_model(provider, model)

    only_ids = set(only.split(",")) if only else None

    results = normalize_deck(raw_dir, out_dir, llm=llm, only=only_ids)
    console.print(f"[green]Normalized {len(results)} cards.[/green]")

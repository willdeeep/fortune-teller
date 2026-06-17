"""CLI entry point: ``ft-normalize-thoth``.

Usage::

    uv run ft-normalize-thoth                    # synthesize Thoth synergies (API)
    uv run ft-normalize-thoth --provider local    # use local llama-server
    uv run ft-normalize-thoth --only the-fool,the-magician  # re-run specific cards
    uv run ft-normalize-thoth --no-llm           # dry run (no LLM, IDs stay empty)
"""

from __future__ import annotations

import typer
from rich.console import Console

from fortune_teller.application.config import settings
from fortune_teller.developer.normalize.rider_waite import build_normalize_model
from fortune_teller.developer.normalize.thoth import synthesize_deck_synergies

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    provider: str = typer.Option(
        "api", "--provider", help="LLM provider: 'api' (Claude) or 'local' (llama-server)."
    ),
    model: str = typer.Option("claude-sonnet-4-6", "--model", help="Model identifier."),
    only: str | None = typer.Option(None, "--only", help="Comma-separated card IDs to re-run."),
    no_llm: bool = typer.Option(False, "--no-llm", help="Dry run — no LLM, IDs stay empty."),
) -> None:
    parsed_dir = settings.ft_data_dir / "parsed"

    if not (parsed_dir / "book-of-thoth").is_dir():
        console.print(
            f"[red]Thoth parsed directory not found: {parsed_dir / 'book-of-thoth'}[/red]"
        )
        console.print("[dim]Run ft-scrape --source thoth and ft-parse --source thoth first.[/dim]")
        raise typer.Exit(1)

    llm = None
    if not no_llm:
        llm = build_normalize_model(provider, model)

    only_ids = set(only.split(",")) if only else None

    results = synthesize_deck_synergies(parsed_dir, llm=llm, only=only_ids)
    console.print(f"[green]Synthesized synergies for {len(results)} Thoth cards.[/green]")

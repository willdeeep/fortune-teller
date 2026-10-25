"""CLI entry point: ``ft-normalize`` — umbrella command for all normalizers.

Runs the Rider-Waite normalizer and the Thoth synergy synthesizer in sequence.

Usage::

    uv run ft-normalize                        # normalize all decks (API)
    uv run ft-normalize --deck rw              # Rider-Waite only
    uv run ft-normalize --deck thoth           # Thoth synergies only
    uv run ft-normalize --provider local       # use local llama-server
    uv run ft-normalize --no-llm               # dry run (no LLM calls)
"""

from __future__ import annotations

import typer
from rich.console import Console

from fortune_teller.application.config import settings
from fortune_teller.developer.normalize.reporting import report_outcome, retry_command
from fortune_teller.developer.normalize.rider_waite import build_normalize_model, normalize_deck
from fortune_teller.developer.normalize.thoth import synthesize_deck_synergies

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    provider: str = typer.Option(
        "api", "--provider", help="LLM provider: 'api' (Claude) or 'local' (llama-server)."
    ),
    model: str = typer.Option("claude-sonnet-4-6", "--model", help="Model identifier."),
    deck: str = typer.Option("all", "--deck", help="Deck to normalize: 'rw', 'thoth', or 'all'."),
    only: str | None = typer.Option(None, "--only", help="Comma-separated card IDs to re-run."),
    no_llm: bool = typer.Option(False, "--no-llm", help="Dry run — no LLM calls."),
) -> None:
    llm = None
    if not no_llm:
        llm = build_normalize_model(provider, model)

    only_ids = set(only.split(",")) if only else None
    any_failed = False

    if deck in ("rw", "all"):
        raw_dir = settings.ft_data_dir / "raw" / "rider-waite"
        out_dir = settings.ft_data_dir / "parsed" / "rider-waite"

        if not raw_dir.is_dir():
            console.print(f"[red]RW raw directory not found: {raw_dir}[/red]")
            console.print(
                "[dim]Run ft-scrape --source learntarot and "
                "ft-parse --source learntarot first.[/dim]"
            )
            if deck == "rw":
                raise typer.Exit(1)
        else:
            outcome = normalize_deck(raw_dir, out_dir, llm=llm, only=only_ids)
            retry = retry_command(
                "uv run ft-normalize --deck rw", provider=provider, model=model, no_llm=no_llm
            )
            any_failed |= report_outcome(
                console, outcome, action="Rider-Waite: normalized", retry=retry
            )

    if deck in ("thoth", "all"):
        parsed_dir = settings.ft_data_dir / "parsed"

        if not (parsed_dir / "book-of-thoth").is_dir():
            console.print(
                f"[red]Thoth parsed directory not found: {parsed_dir / 'book-of-thoth'}[/red]"
            )
            console.print(
                "[dim]Run ft-scrape --source thoth and ft-parse --source thoth first.[/dim]"
            )
            if deck == "thoth":
                raise typer.Exit(1)
        else:
            outcome = synthesize_deck_synergies(parsed_dir, llm=llm, only=only_ids)
            retry = retry_command(
                "uv run ft-normalize --deck thoth", provider=provider, model=model, no_llm=no_llm
            )
            any_failed |= report_outcome(
                console, outcome, action="Thoth: synthesized synergies for", retry=retry
            )

    # Every requested deck has been attempted; now signal any partial failure.
    if any_failed:
        raise typer.Exit(1)

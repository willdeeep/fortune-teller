"""CLI entry point for the embedder: ``ft-embed``.

Reads parsed card and spread JSON from ``<FT_DATA_DIR>/parsed/``, chunks
each document into one :class:`Chunk` per section/position, embeds the
text using :class:`Embedder`, and writes per-source JSON files to
``<FT_DATA_DIR>/embedded/`` ready for ``ft-build-index``.

Usage::

    uv run ft-embed                    # embed everything in data/parsed/
    uv run ft-embed --deck book-of-thoth  # limit to one deck

Per architecture (``docs/architecture.md``):

    data/embedded/<slug>.json            ← Chunks with embedding vectors
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from fortune_teller.application.config import settings
from fortune_teller.application.models.domain import Card, Chunk, Spread
from fortune_teller.application.stores.embeddings import Embedder
from fortune_teller.developer.embed.chunker import (
    attach_embeddings,
    chunks_from_card,
    chunks_from_spread,
)

app = typer.Typer(add_completion=False)
console = Console()


def _write_chunks(out_path: Path, chunks: list[Chunk]) -> None:
    """Persist a list of chunks as a JSON array."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [chunk.model_dump(mode="json") for chunk in chunks]
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _process_deck(
    deck_dir: Path,
    out_root: Path,
    embedder: Embedder,
) -> tuple[int, int]:
    """Embed every card JSON in *deck_dir*. Returns (card_count, error_count)."""
    card_count = errors = 0
    for card_path in sorted(deck_dir.glob("*.json")):
        try:
            card = Card.model_validate_json(card_path.read_text(encoding="utf-8"))
            chunks = chunks_from_card(card, deck_id=deck_dir.name)
            if not chunks:
                continue
            embeddings = embedder.embed_texts([c.text for c in chunks])
            embedded = attach_embeddings(chunks, embeddings)
            out_path = out_root / deck_dir.name / f"{card.id}.json"
            _write_chunks(out_path, embedded)
            card_count += 1
            console.print(f"  [green]card[/green]   {card.id} ({len(embedded)} sections)")
        except Exception as exc:
            errors += 1
            console.print(f"  [red]ERROR[/red]  {card_path.name}: {exc}")
    return card_count, errors


def _process_spreads(
    spreads_dir: Path,
    out_root: Path,
    embedder: Embedder,
) -> tuple[int, int]:
    """Embed every spread JSON in *spreads_dir*. Returns (spread_count, error_count)."""
    spread_count = errors = 0
    if not spreads_dir.exists():
        return spread_count, errors

    for spread_path in sorted(spreads_dir.glob("*.json")):
        try:
            spread = Spread.model_validate_json(spread_path.read_text(encoding="utf-8"))
            chunks = chunks_from_spread(spread)
            if not chunks:
                continue
            embeddings = embedder.embed_texts([c.text for c in chunks])
            embedded = attach_embeddings(chunks, embeddings)
            # Per architecture: data/embedded/<slug>.json (no deck subdir).
            out_path = out_root / "spreads" / f"{spread.id}.json"
            _write_chunks(out_path, embedded)
            spread_count += 1
            console.print(f"  [cyan]spread[/cyan] {spread.id} ({len(embedded)} positions)")
        except Exception as exc:
            errors += 1
            console.print(f"  [red]ERROR[/red]  {spread_path.name}: {exc}")
    return spread_count, errors


@app.command()
def main(
    deck: str | None = typer.Option(
        None,
        "--deck",
        help="Limit to one deck directory under data/parsed/ (e.g. book-of-thoth).",
    ),
) -> None:
    """Embed parsed cards and spreads into per-source JSON files."""
    data_dir = settings.ft_data_dir
    parsed_dir = data_dir / "parsed"
    out_root = data_dir / "embedded"
    out_root.mkdir(parents=True, exist_ok=True)

    if not parsed_dir.exists():
        console.print(f"[red]No parsed data at {parsed_dir}[/red]")
        raise typer.Exit(1)

    # No inputs to process → nothing to embed. Treat as an error so the
    # user notices (e.g. they forgot to run ft-parse first).
    if not any(parsed_dir.iterdir()):
        console.print(f"[red]No parsed data under {parsed_dir}[/red]")
        raise typer.Exit(1)

    embedder = Embedder()
    console.print(f"[bold]Embedding model:[/bold] {embedder.model_name}")

    card_count = spread_count = errors = 0

    # --- Cards -----------------------------------------------------------
    if deck is not None:
        deck_dir = parsed_dir / deck
        if not deck_dir.is_dir():
            console.print(f"[red]Deck directory not found: {deck_dir}[/red]")
            raise typer.Exit(1)
        c, e = _process_deck(deck_dir, out_root, embedder)
        card_count += c
        errors += e
    else:
        for entry in sorted(parsed_dir.iterdir()):
            if not entry.is_dir() or entry.name == "spreads":
                continue
            c, e = _process_deck(entry, out_root, embedder)
            card_count += c
            errors += e

    # --- Spreads ---------------------------------------------------------
    s, e = _process_spreads(parsed_dir / "spreads", out_root, embedder)
    spread_count += s
    errors += e

    console.print(
        f"\n[bold]Done.[/bold] {card_count} cards, {spread_count} spreads, "
        f"{errors} errors. Output: {out_root}"
    )
    if errors:
        raise typer.Exit(1)

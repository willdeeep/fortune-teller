"""CLI entry point for the index builder: ``ft-build-index``.

Reads embedded chunks from ``<FT_DATA_DIR>/embedded/``, writes them into a
DuckDB VSS vector store at ``<FT_DATA_DIR>/duckdb/fortune.duckdb`` with
an HNSW cosine-similarity index.

Usage::

    uv run ft-build-index           # rebuild from scratch (idempotent)
    uv run ft-build-index --no-rebuild  # append to existing index

The default ``--rebuild`` mode drops and recreates the chunks table —
safe because the source of truth is the embedded JSON. This keeps
re-runs of the data pipeline deterministic.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from fortune_teller.application.config import settings
from fortune_teller.application.models.domain import Chunk
from fortune_teller.application.stores.vector import VectorStore

app = typer.Typer(add_completion=False)
console = Console()


def _load_embedded_chunks(embedded_dir: Path) -> list[Chunk]:
    """Load every JSON file under *embedded_dir* and return a flat chunk list."""
    chunks: list[Chunk] = []
    for path in sorted(embedded_dir.rglob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for entry in payload:
            chunks.append(Chunk.model_validate(entry))
    return chunks


def _validate_embedded(chunks: list[Chunk]) -> list[Chunk]:
    """Return only chunks that have a non-empty embedding vector.

    ``ft-embed`` should always populate ``embedding``, but defensively
    skip any that don't so the build step never fails the whole batch.
    """
    return [c for c in chunks if c.embedding and len(c.embedding) > 0]


@app.command()
def main(
    rebuild: bool = typer.Option(
        True,
        "--rebuild/--no-rebuild",
        help="Drop and recreate the chunks table before inserting (default: rebuild).",
    ),
) -> None:
    """Insert embedded chunks into the DuckDB vector store."""
    data_dir = settings.ft_data_dir
    embedded_dir = data_dir / "embedded"
    db_path = data_dir / "duckdb" / "fortune.duckdb"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if not embedded_dir.exists() or not any(embedded_dir.rglob("*.json")):
        console.print(f"[red]No embedded chunks at {embedded_dir}[/red]")
        raise typer.Exit(1)

    console.print(f"[bold]Loading chunks from[/bold] {embedded_dir}")
    all_chunks = _load_embedded_chunks(embedded_dir)
    chunks = _validate_embedded(all_chunks)
    skipped = len(all_chunks) - len(chunks)
    if skipped:
        console.print(f"  [yellow]Skipped {skipped} chunks with empty embeddings[/yellow]")

    if not chunks:
        console.print("[red]No chunks with embeddings to index.[/red]")
        raise typer.Exit(1)

    # Dimension comes from the first chunk (assumes all chunks share it).
    dimension = len(chunks[0].embedding or [])
    console.print(
        f"[bold]Indexing[/bold] {len(chunks)} chunks "
        f"(dim={dimension}, rebuild={rebuild}) -> {db_path}"
    )

    with VectorStore(db_path, dimension=dimension) as store:
        if rebuild:
            store.clear()
            store.ensure_schema()
        store.add_chunks(chunks)
        count = store.count()

    console.print(f"[green]Indexed {count} chunks.[/green]")

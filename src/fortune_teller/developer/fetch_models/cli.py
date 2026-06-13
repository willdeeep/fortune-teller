"""CLI entry point for fetching the embedding model snapshot.

Downloads the HuggingFace sentence-transformers snapshot into
``<FT_DATA_DIR>/models/`` for fully-offline runtime use.

Usage::

    uv run ft-fetch-models      # one-time, network-bound

The command is idempotent — re-running with the snapshot already present
is a no-op (huggingface_hub verifies the local cache).
"""

from __future__ import annotations

import sys

from huggingface_hub import snapshot_download
from rich.console import Console

from fortune_teller.application.config import settings

app = Console()


def main() -> None:
    """Download the embedding model snapshot into data/models/ for offline use."""
    target = settings.embedding_model_path
    model_name = settings.embedding_model

    app.print(f"[bold]Fetching embedding model:[/bold] {model_name}")
    app.print(f"[bold]Target directory:[/bold] {target}")

    try:
        snapshot_download(
            repo_id=model_name,
            local_dir=str(target),
        )
        app.print(f"[green]✓[/green] Model snapshot ready at {target}")
    except Exception as exc:  # pragma: no cover — network-dependent
        app.print(f"[red]✗[/red] Fetch failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()

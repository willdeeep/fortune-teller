"""Shared end-of-run reporting for the normalisation CLIs.

``ft-normalize-rw``, ``ft-normalize-thoth`` and the ``ft-normalize`` umbrella
all finish the same way: print how many cards succeeded, list any failures
with a ready-to-run ``--only`` retry command, and tell the caller whether to
exit non-zero.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.markup import escape

if TYPE_CHECKING:
    from rich.console import Console

    from fortune_teller.developer.normalize.rider_waite import NormalizeOutcome


def retry_command(base: str, *, provider: str, model: str, no_llm: bool) -> str:
    """Return *base* with the flags needed to repeat this run exactly.

    The retry must use the same backend as the failed run, so the provider and
    model are always spelled out (``--no-llm`` replaces them for dry runs).

    Args:
        base: The command to re-run, e.g. ``"uv run ft-normalize-rw"``.
        provider: The ``--provider`` value used for this run.
        model: The ``--model`` value used for this run.
        no_llm: Whether this run used ``--no-llm``.

    Returns:
        The command without ``--only``; :func:`report_outcome` appends that.
    """
    if no_llm:
        return f"{base} --no-llm"
    return f"{base} --provider {provider} --model {model}"


def report_outcome(
    console: Console,
    outcome: NormalizeOutcome,
    *,
    action: str,
    retry: str,
) -> bool:
    """Print a run summary and return whether any card failed.

    Args:
        console: Where to print.
        outcome: The normaliser's result.
        action: Phrase that precedes the card count, e.g. ``"Normalized"`` gives
            ``"Normalized 76 cards; 2 failed:"``.
        retry: Retry command from :func:`retry_command` (``--only`` is appended).

    Returns:
        ``True`` if any card failed — the caller should exit non-zero.
    """
    n_ok, n_failed = len(outcome.results), len(outcome.failed)
    if not n_failed:
        console.print(f"[green]{escape(action)} {n_ok} cards.[/green]")
        return False

    console.print(f"[yellow]{escape(action)} {n_ok} cards; {n_failed} failed:[/yellow]")
    for card_id, error in outcome.failed:
        # Errors often contain brackets (e.g. "[Errno 61]") that rich would
        # otherwise parse as markup and swallow.
        console.print(f"  [red]✗[/red] {escape(card_id)}: {escape(error)}", soft_wrap=True)
    ids = ",".join(card_id for card_id, _ in outcome.failed)
    # soft_wrap keeps the command on one logical line so it can be copied and
    # pasted even when it is wider than the terminal.
    console.print(f"Retry just these: {escape(retry)} --only {escape(ids)}", soft_wrap=True)
    return True

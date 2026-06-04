"""Load :class:`Deck` and :class:`Spread` instances from on-disk JSON.

The data pipeline (``ft-parse`` → ``ft-embed`` → ``ft-build-index``) writes
parsed card and spread JSON to ``<FT_DATA_DIR>/parsed/``::

    data/parsed/
        book-of-thoth/
            0-the-fool.json
            ...
        spreads/
            new-moon-three-card.json

The functions in this module are the inverse of the parser CLI: they read
that on-disk layout and reconstruct the typed pydantic models. They are
used by ``build_reading_service`` to bootstrap the app from a populated
``data/`` directory.

No I/O happens at import time — the directory is only touched when the
functions are called.
"""

from __future__ import annotations

from pathlib import Path

from fortune_teller.application.models.domain import Card, Deck, Spread


def load_deck(parsed_dir: Path, deck_id: str) -> Deck:
    """Load a :class:`Deck` from ``parsed_dir / <deck_id>``.

    Args:
        parsed_dir: Root of the parsed data directory
                    (e.g. ``settings.ft_data_dir / "parsed"``).
        deck_id:    Subdirectory name identifying the deck
                    (e.g. ``"book-of-thoth"``).

    Returns:
        A :class:`Deck` containing every card JSON file in the deck
        subdirectory, sorted by filename for deterministic ordering.

    Raises:
        FileNotFoundError: If the deck subdirectory does not exist.
        ValueError:         If no card JSON files are found in the subdirectory.
    """
    deck_dir = parsed_dir / deck_id
    if not deck_dir.is_dir():
        raise FileNotFoundError(f"Deck directory not found: {deck_dir}")

    card_paths = sorted(deck_dir.glob("*.json"))
    if not card_paths:
        raise ValueError(f"No card JSON files in {deck_dir}")

    cards = [Card.model_validate_json(p.read_text(encoding="utf-8")) for p in card_paths]
    return Deck(id=deck_id, name=deck_id.replace("-", " ").title(), cards=cards)


def load_spread(parsed_dir: Path, spread_id: str) -> Spread:
    """Load a :class:`Spread` from ``parsed_dir / spreads / <spread_id>.json``.

    Args:
        parsed_dir: Root of the parsed data directory.
        spread_id:  Spread slug (e.g. ``"new-moon-three-card"``).

    Returns:
        A :class:`Spread` parsed from the JSON file.

    Raises:
        FileNotFoundError: If the spread JSON does not exist.
    """
    path = parsed_dir / "spreads" / f"{spread_id}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Spread file not found: {path}")
    return Spread.model_validate_json(path.read_text(encoding="utf-8"))


def load_first_spread(parsed_dir: Path) -> Spread:
    """Load the first spread found under ``parsed_dir / spreads/``.

    A convenience for the spike, which only supports a single spread.
    Files are sorted by name for deterministic behaviour.

    Raises:
        FileNotFoundError: If the spreads directory is missing or empty.
    """
    spreads_dir = parsed_dir / "spreads"
    if not spreads_dir.is_dir():
        raise FileNotFoundError(f"Spreads directory not found: {spreads_dir}")
    paths = sorted(spreads_dir.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No spread JSON files in {spreads_dir}")
    return Spread.model_validate_json(paths[0].read_text(encoding="utf-8"))


def list_spread_ids(parsed_dir: Path) -> list[str]:
    """Return the slugs of every spread JSON under ``parsed_dir / spreads/``.

    Useful for surfacing a spread selector in a future UI iteration.
    """
    spreads_dir = parsed_dir / "spreads"
    if not spreads_dir.is_dir():
        return []
    return [p.stem for p in sorted(spreads_dir.glob("*.json"))]


__all__ = [
    "list_spread_ids",
    "load_deck",
    "load_first_spread",
    "load_spread",
]

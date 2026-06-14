"""Resolve card artwork images from the local file system.

Given a ``card_id`` and an images directory, find the corresponding image
file. Filenames are deterministic from ``card_id`` but the extension varies
(``.jpeg`` / ``.png`` / ``.webp``), so a glob resolves the exact path.

Usage::

    from fortune_teller.application.stores.images import image_path_for

    path = image_path_for("0-the-fool", Path("./data/images/book-of-thoth"))
    # => Path("./data/images/book-of-thoth/0-the-fool.jpeg") or None
"""

from __future__ import annotations

from pathlib import Path

_ALLOWED_EXTENSIONS: tuple[str, ...] = (".jpeg", ".jpg", ".png", ".webp", ".gif")


def image_path_for(card_id: str, images_dir: Path) -> Path | None:
    """Return the local artwork path for *card_id*, or ``None`` if absent.

    Searches *images_dir* for any file matching ``<card_id>.*`` with a
    recognised image extension. Returns the first match (sorted for
    determinism).

    Args:
        card_id: The card slug, e.g. ``"0-the-fool"``.
        images_dir: Directory containing deck image files.

    Returns:
        The resolved :class:`Path`, or ``None`` if no image was found.
    """
    if not images_dir.is_dir():
        return None
    matches = sorted(
        p
        for p in images_dir.iterdir()
        if p.stem == card_id and p.suffix.lower() in _ALLOWED_EXTENSIONS
    )
    return matches[0] if matches else None

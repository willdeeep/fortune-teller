"""Unit tests for the loading service.

Exercises the filesystem code paths against a ``tmp_path`` populated
by small factory helpers, with no dependency on the long-lived
``tests/fixtures/parsed/`` directory.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import HttpUrl

from fortune_teller.application.models.domain import Card, Deck, Spread
from fortune_teller.application.services.loading import (
    list_spread_ids,
    load_deck,
    load_first_spread,
    load_spread,
)


def _write_card_json(path: Path, card: Card) -> None:
    path.write_text(card.model_dump_json(indent=2), encoding="utf-8")


def _write_spread_json(path: Path, spread: Spread) -> None:
    path.write_text(spread.model_dump_json(indent=2), encoding="utf-8")


def _card(cid: str) -> Card:
    return Card(
        id=cid,
        name=cid.replace("-", " ").title(),
        arcana="major",  # type: ignore[arg-type]
        source_url=HttpUrl("https://example.test/card"),
    )


def _spread(sid: str, n: int = 3) -> Spread:
    return Spread(
        id=sid,
        name=f"Test Spread {sid}",
        positions=[
            {
                "index": i,
                "name": f"Pos {i}",
                "meaning": f"Meaning {i}.",
                "source_url": "https://example.test/spread",
            }
            for i in range(n)
        ],
    )


def _populated_parsed(
    tmp_path: Path,
    *,
    deck_id: str = "book-of-thoth",
    card_ids: list[str] | None = None,
    spreads: list[Spread] | None = None,
) -> Path:
    """Build ``<tmp>/parsed/`` with a deck subdir and optional spread JSONs."""
    parsed_root = tmp_path / "parsed"
    parsed_root.mkdir()
    deck_dir = parsed_root / deck_id
    deck_dir.mkdir()
    for cid in card_ids or ["0-the-fool", "i-the-magician"]:
        _write_card_json(deck_dir / f"{cid}.json", _card(cid))
    if spreads:
        spreads_dir = parsed_root / "spreads"
        spreads_dir.mkdir()
        for s in spreads:
            _write_spread_json(spreads_dir / f"{s.id}.json", s)
    return parsed_root


# ---------------------------------------------------------------------------
# load_deck
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLoadDeck:
    def test_returns_deck_with_all_cards(self, tmp_path: Path) -> None:
        parsed = _populated_parsed(
            tmp_path,
            card_ids=["0-the-fool", "i-the-magician", "ii-the-high-priestess"],
        )
        deck = load_deck(parsed, "book-of-thoth")
        assert isinstance(deck, Deck)
        assert deck.id == "book-of-thoth"
        assert len(deck.cards) == 3

    def test_cards_are_sorted_by_filename(self, tmp_path: Path) -> None:
        # The factory writes cards in the order given; load_deck sorts.
        parsed = _populated_parsed(
            tmp_path,
            card_ids=["z-last", "a-first", "m-middle"],
        )
        deck = load_deck(parsed, "book-of-thoth")
        ids = [c.id for c in deck.cards]
        assert ids == sorted(ids)

    def test_card_models_validate(self, tmp_path: Path) -> None:
        parsed = _populated_parsed(tmp_path, card_ids=["0-the-fool"])
        deck = load_deck(parsed, "book-of-thoth")
        assert deck.cards[0].arcana.value == "major"
        assert deck.cards[0].source_url.host == "example.test"

    def test_missing_deck_dir_raises(self, tmp_path: Path) -> None:
        (tmp_path / "parsed").mkdir()
        with pytest.raises(FileNotFoundError, match="Deck directory not found"):
            load_deck(tmp_path / "parsed", "nonexistent")

    def test_empty_deck_dir_raises(self, tmp_path: Path) -> None:
        parsed = tmp_path / "parsed"
        parsed.mkdir()
        (parsed / "empty-deck").mkdir()
        with pytest.raises(ValueError, match="No card JSON files"):
            load_deck(parsed, "empty-deck")


# ---------------------------------------------------------------------------
# load_spread
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLoadSpread:
    def test_returns_spread(self, tmp_path: Path) -> None:
        parsed = _populated_parsed(tmp_path, spreads=[_spread("new-moon", 3)])
        spread = load_spread(parsed, "new-moon")
        assert spread.id == "new-moon"
        assert len(spread.positions) == 3

    def test_missing_spread_raises(self, tmp_path: Path) -> None:
        (tmp_path / "parsed" / "spreads").mkdir(parents=True)
        with pytest.raises(FileNotFoundError, match="Spread file not found"):
            load_spread(tmp_path / "parsed", "missing")

    def test_loads_specific_spread_when_multiple(self, tmp_path: Path) -> None:
        parsed = _populated_parsed(
            tmp_path,
            spreads=[_spread("alpha", 2), _spread("beta", 5)],
        )
        spread = load_spread(parsed, "beta")
        assert spread.id == "beta"
        assert len(spread.positions) == 5


# ---------------------------------------------------------------------------
# load_first_spread
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLoadFirstSpread:
    def test_returns_first_by_filename(self, tmp_path: Path) -> None:
        parsed = _populated_parsed(
            tmp_path,
            spreads=[_spread("zebra", 2), _spread("alpha", 4)],
        )
        # Files are written as "alpha.json" and "zebra.json"; sorted
        # alphabetically "alpha" comes first.
        spread = load_first_spread(parsed)
        assert spread.id == "alpha"
        assert len(spread.positions) == 4

    def test_missing_spreads_dir_raises(self, tmp_path: Path) -> None:
        (tmp_path / "parsed").mkdir()
        with pytest.raises(FileNotFoundError, match="Spreads directory not found"):
            load_first_spread(tmp_path / "parsed")

    def test_empty_spreads_dir_raises(self, tmp_path: Path) -> None:
        (tmp_path / "parsed" / "spreads").mkdir(parents=True)
        with pytest.raises(FileNotFoundError, match="No spread JSON files"):
            load_first_spread(tmp_path / "parsed")


# ---------------------------------------------------------------------------
# list_spread_ids
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestListSpreadIds:
    def test_returns_all_spread_ids_sorted(self, tmp_path: Path) -> None:
        parsed = _populated_parsed(
            tmp_path,
            spreads=[_spread("zeta", 2), _spread("alpha", 3), _spread("mu", 4)],
        )
        assert list_spread_ids(parsed) == ["alpha", "mu", "zeta"]

    def test_returns_empty_when_no_spreads_dir(self, tmp_path: Path) -> None:
        (tmp_path / "parsed").mkdir()
        assert list_spread_ids(tmp_path / "parsed") == []

    def test_returns_empty_when_dir_is_empty(self, tmp_path: Path) -> None:
        (tmp_path / "parsed" / "spreads").mkdir(parents=True)
        assert list_spread_ids(tmp_path / "parsed") == []

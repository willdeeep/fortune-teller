"""Unit tests for the learntarot.com URL → identity map.

Tests that ``RW_CARD_MAP`` contains exactly 78 entries with correct
Rider-Waite naming (not Thoth naming).
"""

from __future__ import annotations

import pytest

from fortune_teller.application.models.domain import Arcana, Suit
from fortune_teller.developer.parse.learntarot import RW_CARD_MAP


@pytest.mark.unit
class TestRWCardMapSize:
    """RW_CARD_MAP must contain exactly 78 cards (22 major + 56 minor)."""

    def test_has_exactly_78_entries(self) -> None:
        assert len(RW_CARD_MAP) == 78

    def test_all_ids_are_unique(self) -> None:
        ids = [entry[0] for entry in RW_CARD_MAP.values()]
        assert len(ids) == len(set(ids)), "Duplicate card IDs found in RW_CARD_MAP"


@pytest.mark.unit
class TestRWCardMapMajorArcana:
    """Major arcana entries have correct Rider-Waite naming."""

    @pytest.mark.parametrize(
        ("slug", "expected_id", "expected_name", "expected_number"),
        [
            ("maj00", "the-fool", "The Fool", 0),
            ("maj01", "the-magician", "The Magician", 1),
            ("maj02", "the-high-priestess", "The High Priestess", 2),
            ("maj03", "the-empress", "The Empress", 3),
            ("maj04", "the-emperor", "The Emperor", 4),
            ("maj05", "the-hierophant", "The Hierophant", 5),
            ("maj06", "the-lovers", "The Lovers", 6),
            ("maj07", "the-chariot", "The Chariot", 7),
            ("maj08", "strength", "Strength", 8),
            ("maj09", "the-hermit", "The Hermit", 9),
            ("maj10", "wheel-of-fortune", "Wheel of Fortune", 10),
            ("maj11", "justice", "Justice", 11),
            ("maj12", "the-hanged-man", "The Hanged Man", 12),
            ("maj13", "death", "Death", 13),
            ("maj14", "temperance", "Temperance", 14),
            ("maj15", "the-devil", "The Devil", 15),
            ("maj16", "the-tower", "The Tower", 16),
            ("maj17", "the-star", "The Star", 17),
            ("maj18", "the-moon", "The Moon", 18),
            ("maj19", "the-sun", "The Sun", 19),
            ("maj20", "judgement", "Judgement", 20),
            ("maj21", "the-world", "The World", 21),
        ],
    )
    def test_major_arcana_entry(
        self,
        slug: str,
        expected_id: str,
        expected_name: str,
        expected_number: int,
    ) -> None:
        entry = RW_CARD_MAP.get(slug)
        assert entry is not None, f"Missing major arcana slug: {slug}"
        card_id, name, arcana, suit, number = entry
        assert card_id == expected_id
        assert name == expected_name
        assert arcana == Arcana.MAJOR
        assert suit is None
        assert number == expected_number

    def test_rider_waite_names_not_thoth(self) -> None:
        """Verify Rider-Waite naming where Thoth differs."""
        # RW: Strength (VIII), not Adjustment/Lust
        assert RW_CARD_MAP["maj08"][0] == "strength"
        assert RW_CARD_MAP["maj08"][1] == "Strength"
        # RW: Justice is XI, not Adjustment
        assert RW_CARD_MAP["maj11"][0] == "justice"
        assert RW_CARD_MAP["maj11"][1] == "Justice"
        # RW: Judgement, not Aeon
        assert RW_CARD_MAP["maj20"][0] == "judgement"
        assert RW_CARD_MAP["maj20"][1] == "Judgement"
        # RW: The World, not Universe
        assert RW_CARD_MAP["maj21"][0] == "the-world"
        assert RW_CARD_MAP["maj21"][1] == "The World"


@pytest.mark.unit
class TestRWCardMapMinorArcana:
    """Minor arcana entries cover all four suits with correct metadata."""

    @pytest.mark.parametrize(
        ("slug", "expected_id", "expected_name", "expected_suit", "expected_number"),
        [
            ("wa", "ace-of-wands", "Ace of Wands", Suit.WANDS, 1),
            ("w2", "two-of-wands", "Two of Wands", Suit.WANDS, 2),
            ("w10", "ten-of-wands", "Ten of Wands", Suit.WANDS, 10),
            ("wpg", "page-of-wands", "Page of Wands", Suit.WANDS, 11),
            ("wkn", "knight-of-wands", "Knight of Wands", Suit.WANDS, 12),
            ("wqn", "queen-of-wands", "Queen of Wands", Suit.WANDS, 13),
            ("wkg", "king-of-wands", "King of Wands", Suit.WANDS, 14),
            ("ca", "ace-of-cups", "Ace of Cups", Suit.CUPS, 1),
            ("c7", "seven-of-cups", "Seven of Cups", Suit.CUPS, 7),
            ("cpg", "page-of-cups", "Page of Cups", Suit.CUPS, 11),
            ("ckn", "knight-of-cups", "Knight of Cups", Suit.CUPS, 12),
            ("cqn", "queen-of-cups", "Queen of Cups", Suit.CUPS, 13),
            ("ckg", "king-of-cups", "King of Cups", Suit.CUPS, 14),
            ("sa", "ace-of-swords", "Ace of Swords", Suit.SWORDS, 1),
            ("s3", "three-of-swords", "Three of Swords", Suit.SWORDS, 3),
            ("spg", "page-of-swords", "Page of Swords", Suit.SWORDS, 11),
            ("skn", "knight-of-swords", "Knight of Swords", Suit.SWORDS, 12),
            ("sqn", "queen-of-swords", "Queen of Swords", Suit.SWORDS, 13),
            ("skg", "king-of-swords", "King of Swords", Suit.SWORDS, 14),
        ],
    )
    def test_minor_arcana_entry(
        self,
        slug: str,
        expected_id: str,
        expected_name: str,
        expected_suit: Suit,
        expected_number: int,
    ) -> None:
        entry = RW_CARD_MAP.get(slug)
        assert entry is not None, f"Missing minor arcana slug: {slug}"
        card_id, name, arcana, suit, number = entry
        assert card_id == expected_id
        assert name == expected_name
        assert arcana == Arcana.MINOR
        assert suit == expected_suit
        assert number == expected_number

    def test_pentacles_suit_used(self) -> None:
        """Pentacles prefix 'p' maps to Suit.PENTACLES, not DISKS."""
        entry = RW_CARD_MAP.get("pa")
        assert entry is not None
        assert entry[3] == Suit.PENTACLES  # suit
        assert entry[0] == "ace-of-pentacles"

    def test_pentacles_card_naming(self) -> None:
        """Pentacles cards use 'pentacles' in name, not 'disks'."""
        for slug_key, entry in RW_CARD_MAP.items():
            if slug_key.startswith("p"):
                _, name, _, suit, _ = entry
                assert suit == Suit.PENTACLES
                assert "pentacles" in name.lower()
                assert "disks" not in name.lower()

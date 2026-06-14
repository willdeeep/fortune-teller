"""Unit tests for :mod:`fortune_teller.developer.normalize.prompts`."""

from __future__ import annotations

import pytest

from fortune_teller.developer.normalize.prompts import GAPFILL_HUMAN, REBUCKET_HUMAN

# ---------------------------------------------------------------------------
# REBUCKET_HUMAN rendering
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRebucketHumanRenders:
    def test_renders_card_name(self) -> None:
        rendered = REBUCKET_HUMAN.format(
            card_name="The Fool",
            arcana="major",
            suit_info="",
            keywords="beginnings, freedom, innocence",
            actions="Take a leap, Trust the process",
            description="A young man stands at the edge of a cliff.",
        )
        assert "The Fool" in rendered

    def test_renders_keywords(self) -> None:
        rendered = REBUCKET_HUMAN.format(
            card_name="The Fool",
            arcana="major",
            suit_info="",
            keywords="beginnings, freedom, innocence",
            actions="Take a leap",
            description="A young man.",
        )
        assert "beginnings, freedom, innocence" in rendered

    def test_renders_actions(self) -> None:
        rendered = REBUCKET_HUMAN.format(
            card_name="The Fool",
            arcana="major",
            suit_info="",
            keywords="beginnings",
            actions="Take a leap, Trust the process",
            description="A young man.",
        )
        assert "Take a leap" in rendered

    def test_renders_description(self) -> None:
        rendered = REBUCKET_HUMAN.format(
            card_name="The Fool",
            arcana="major",
            suit_info="",
            keywords="beginnings",
            actions="Take a leap",
            description="A young man stands at the edge of a cliff.",
        )
        assert "edge of a cliff" in rendered

    def test_renders_suit_info_for_minor(self) -> None:
        rendered = REBUCKET_HUMAN.format(
            card_name="Ace of Wands",
            arcana="minor",
            suit_info=" of wands",
            keywords="creation, inspiration",
            actions="Create something new",
            description="A hand holds a flowering staff.",
        )
        assert " of wands" in rendered

    def test_renders_suit_info_empty_for_major(self) -> None:
        rendered = REBUCKET_HUMAN.format(
            card_name="The Fool",
            arcana="major",
            suit_info="",
            keywords="beginnings",
            actions="Take a leap",
            description="A young man.",
        )
        assert "major)" in rendered or "major" in rendered.split("Suit_info")[0]


# ---------------------------------------------------------------------------
# GAPFILL_HUMAN rendering
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGapfillHumanRenders:
    def test_renders_card_name(self) -> None:
        rendered = GAPFILL_HUMAN.format(
            card_name="The Fool",
            arcana="major",
            suit_info="",
            keywords="beginnings",
            actions="Take a leap",
            description="A young man.",
            existing_sections="light: Be spontaneous",
        )
        assert "The Fool" in rendered

    def test_renders_keywords(self) -> None:
        rendered = GAPFILL_HUMAN.format(
            card_name="The Fool",
            arcana="major",
            suit_info="",
            keywords="beginnings, freedom",
            actions="Take a leap",
            description="A young man.",
            existing_sections="light: Be spontaneous",
        )
        assert "beginnings, freedom" in rendered

    def test_renders_existing_sections(self) -> None:
        rendered = GAPFILL_HUMAN.format(
            card_name="The Fool",
            arcana="major",
            suit_info="",
            keywords="beginnings",
            actions="Take a leap",
            description="A young man.",
            existing_sections="light: Be spontaneous\nshadow: Watch your step",
        )
        assert "Be spontaneous" in rendered
        assert "Watch your step" in rendered

    def test_renders_fallback_when_empty(self) -> None:
        rendered = GAPFILL_HUMAN.format(
            card_name="The Fool",
            arcana="major",
            suit_info="",
            keywords="beginnings",
            actions="Take a leap",
            description="A young man.",
            existing_sections="(none populated)",
        )
        assert "(none populated)" in rendered

    def test_renders_suit_info_for_minor(self) -> None:
        rendered = GAPFILL_HUMAN.format(
            card_name="Three of Swords",
            arcana="minor",
            suit_info=" of swords",
            keywords="heartbreak, sorrow",
            actions="Grieve",
            description="Three swords pierce a heart.",
            existing_sections="shadow: Pain and loss",
        )
        assert " of swords" in rendered

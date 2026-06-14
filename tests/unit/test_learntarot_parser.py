"""Unit tests for the learntarot.com HTML parser.

All tests use inline HTML fixtures — no live HTTP calls or fixture files.
"""

from __future__ import annotations

import pytest

from fortune_teller.application.models.domain import Arcana, Suit
from fortune_teller.developer.parse.learntarot import parse_card_page

# ---------------------------------------------------------------------------
# Inline HTML fixtures
# ---------------------------------------------------------------------------

_MAJ00_HTML = """\
<html>
<head><title>The Fool | Learntarot</title></head>
<body>
<h2>The Fool</h2>
<p>Keywords: beginnings, innocence, spontaneity, free spirit</p>
<p>[ACTIONS]</p>
<ul>
<li>Take a leap of faith</li>
<li>Embrace the unknown</li>
<li>Trust the process</li>
</ul>
<p>[OPPOSING CARDS: Some Possibilities]</p>
<p>The Magician, The High Priestess</p>
<p>[REINFORCING CARDS: Some Possibilities]</p>
<p>The World, The Star</p>
<p>[DESCRIPTION]</p>
<p>The Fool card is the first card of the Major Arcana. It represents new
beginnings, innocence, and spontaneity. The Fool is depicted as a carefree
figure standing at the edge of a cliff, ready to step into the unknown.
This card encourages you to take a leap of faith and trust the universe.</p>
</body>
</html>"""

_C7_HTML = """\
<html>
<head><title>Seven of Cups | Learntarot</title></head>
<body>
<h2>Seven of Cups</h2>
<p>Keywords: choices, illusion, fantasy, wishful thinking</p>
<p>[ACTIONS]</p>
<ul>
<li>Consider your options carefully</li>
<li>Distinguish reality from fantasy</li>
<li>Make a conscious choice</li>
</ul>
<p>[OPPOSING CARDS: Some Possibilities]</p>
<p>The Star, Temperance</p>
<p>[REINFORCING CARDS: Some Possibilities]</p>
<p>The Moon, The Devil</p>
<p>[DESCRIPTION]</p>
<p>The Seven of Cups shows a person confronted by seven cups floating on a cloud,
each offering a different vision or temptation. The card warns against wishful
thinking and encourages clear-eyed decision-making.</p>
</body>
</html>"""

_WPG_HTML = """\
<html>
<head><title>Page of Wands | Learntarot</title></head>
<body>
<h2>Page of Wands</h2>
<p>Keywords: enthusiasm, exploration, discovery, free spirit</p>
<p>[ACTIONS]</p>
<ul>
<li>Explore new ideas</li>
<li>Start a creative project</li>
<li>Embrace your curiosity</li>
</ul>
<p>[OPPOSING CARDS: Some Possibilities]</p>
<p>Knight of Wands, King of Wands</p>
<p>[REINFORCING CARDS: Some Possibilities]</p>
<p>Ace of Wands, The Sun</p>
<p>[DESCRIPTION]</p>
<p>The Page of Wands is a youthful figure full of creative fire and adventurous
spirit. This card heralds new opportunities for growth, learning, and
exploration. The energy is fresh, enthusiastic, and unburdened by past
failures.</p>
</body>
</html>"""


# ---------------------------------------------------------------------------
# The Fool — major arcana
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestParseMajorArcana:
    """Test parsing The Fool (maj00)."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self.card = parse_card_page(_MAJ00_HTML, "maj00")

    def test_identity_from_url_map(self) -> None:
        assert self.card.id == "the-fool"
        assert self.card.name == "The Fool"
        assert self.card.arcana == Arcana.MAJOR
        assert self.card.suit is None
        assert self.card.number == 0

    def test_keywords_extracted(self) -> None:
        assert len(self.card.keywords) >= 3
        assert "beginnings" in self.card.keywords
        assert "innocence" in self.card.keywords

    def test_actions_non_empty(self) -> None:
        assert len(self.card.actions) > 0
        assert "Take a leap of faith" in self.card.actions
        assert "Embrace the unknown" in self.card.actions

    def test_opposing_names_parsed(self) -> None:
        assert len(self.card.opposing_names) > 0
        assert "The Magician" in self.card.opposing_names
        assert "The High Priestess" in self.card.opposing_names

    def test_reinforcing_names_parsed(self) -> None:
        assert len(self.card.reinforcing_names) > 0
        assert "The World" in self.card.reinforcing_names
        assert "The Star" in self.card.reinforcing_names

    def test_description_non_empty(self) -> None:
        assert len(self.card.description) > 10
        assert "leap of faith" in self.card.description

    def test_source_url_correct(self) -> None:
        assert self.card.source_url == "https://www.learntarot.com/maj00.htm"


# ---------------------------------------------------------------------------
# Seven of Cups — minor pip card
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestParsePipCard:
    """Test parsing Seven of Cups (c7)."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self.card = parse_card_page(_C7_HTML, "c7")

    def test_identity_from_url_map(self) -> None:
        assert self.card.id == "seven-of-cups"
        assert self.card.name == "Seven of Cups"
        assert self.card.arcana == Arcana.MINOR
        assert self.card.suit == Suit.CUPS
        assert self.card.number == 7

    def test_keywords_extracted(self) -> None:
        assert "choices" in self.card.keywords
        assert "illusion" in self.card.keywords

    def test_actions_non_empty(self) -> None:
        assert len(self.card.actions) > 0
        assert "Consider your options carefully" in self.card.actions

    def test_opposing_names_parsed(self) -> None:
        assert "The Star" in self.card.opposing_names
        assert "Temperance" in self.card.opposing_names

    def test_reinforcing_names_parsed(self) -> None:
        assert "The Moon" in self.card.reinforcing_names
        assert "The Devil" in self.card.reinforcing_names

    def test_description_non_empty(self) -> None:
        assert len(self.card.description) > 10
        assert "Seven of Cups" in self.card.description

    def test_source_url_correct(self) -> None:
        assert self.card.source_url == "https://www.learntarot.com/c7.htm"


# ---------------------------------------------------------------------------
# Page of Wands — court card
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestParseCourtCard:
    """Test parsing Page of Wands (wpg)."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self.card = parse_card_page(_WPG_HTML, "wpg")

    def test_identity_from_url_map(self) -> None:
        assert self.card.id == "page-of-wands"
        assert self.card.name == "Page of Wands"
        assert self.card.arcana == Arcana.MINOR
        assert self.card.suit == Suit.WANDS
        assert self.card.number == 11

    def test_keywords_extracted(self) -> None:
        assert "enthusiasm" in self.card.keywords
        assert "exploration" in self.card.keywords

    def test_actions_non_empty(self) -> None:
        assert len(self.card.actions) > 0
        assert "Explore new ideas" in self.card.actions

    def test_opposing_names_parsed(self) -> None:
        assert "Knight of Wands" in self.card.opposing_names
        assert "King of Wands" in self.card.opposing_names

    def test_reinforcing_names_parsed(self) -> None:
        assert "Ace of Wands" in self.card.reinforcing_names
        assert "The Sun" in self.card.reinforcing_names

    def test_description_non_empty(self) -> None:
        assert len(self.card.description) > 10
        assert "Page of Wands" in self.card.description

    def test_source_url_correct(self) -> None:
        assert self.card.source_url == "https://www.learntarot.com/wpg.htm"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestParseCardErrors:
    def test_unknown_slug_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown learntarot slug"):
            parse_card_page("<html></html>", "nonexistent")

    def test_missing_required_heading_raises_value_error(self) -> None:
        """Page without [ACTIONS] heading should raise ValueError."""
        html = """\
<html>
<body>
<p>Keywords: test</p>
<p>[DESCRIPTION]</p>
<p>Some description.</p>
</body>
</html>"""
        with pytest.raises(ValueError, match="missing required heading"):
            parse_card_page(html, "maj00")

    def test_missing_description_heading_raises_value_error(self) -> None:
        """Page without [DESCRIPTION] heading should raise ValueError."""
        html = """\
<html>
<body>
<p>Keywords: test</p>
<p>[ACTIONS]</p>
<p>Do something.</p>
</body>
</html>"""
        with pytest.raises(ValueError, match="missing required heading"):
            parse_card_page(html, "maj00")

    def test_no_bracketed_headings_raises_value_error(self) -> None:
        """Page with no bracketed headings at all should raise ValueError."""
        html = """\
<html>
<body>
<p>Some random content with no structure.</p>
</body>
</html>"""
        with pytest.raises(ValueError, match="No bracketed headings"):
            parse_card_page(html, "maj00")

    def test_empty_html_raises_error(self) -> None:
        with pytest.raises(ValueError):
            parse_card_page("", "maj00")

"""Unit tests for the learntarot.com HTML parser.

Fixtures mirror the *real* learntarot DOM (fake card text, real structure):
an ``<h1>`` title, a ``<ul>`` of bold keyword items, a bracketed nav menu
(``[ Actions ]`` links — which must be ignored), then uppercase section
headings anchored by ``<a name="...">`` (``ACTIONS``, ``OPPOSING CARDS: ...``,
``REINFORCING CARDS: ...``, ``DESCRIPTION``) whose card lists are
``<li><a>Name</a> - meaning`` items. No live HTTP and no copyrighted text.
"""

from __future__ import annotations

import pytest

from fortune_teller.application.models.domain import Arcana, Suit
from fortune_teller.developer.parse.learntarot import parse_card_page

# ---------------------------------------------------------------------------
# Inline HTML fixtures (structurally faithful to learntarot.com)
# ---------------------------------------------------------------------------

_MAJ00_HTML = """\
<html><head><title>THE FOOL</title></head><body>
<h1>THE FOOL</h1>
<ul>
<li><b>Beginning</b>
<li><b>Spontaneity</b>
<li><b>Faith</b>
</ul>
<img src="rbowline.gif">
<a href="#actions"><b>[ Actions ]</b></a>
<a href="#opposite"><b>[ Opposing Cards ]</b></a>
<a href="#reinforce"><b>[ Reinforcing Cards ]</b></a>
<a href="#description"><b>[ Description ]</b></a>
<a href="howcard.htm#howreversed"><b>[ Reversed? ]</b></a>
<img src="rbowline.gif">
<a name="actions"><b><a href="howcard.htm#howactions">ACTIONS</a></b></a>
<dl>
<dt>beginning a new phase
<dt>striking out on a new path
</dl>
<a name="opposite"><b><a href="#o">OPPOSING CARDS: Some Possibilities</a></b></a>
<ul>
<li><a href="maj05.htm">Hierophant</a> - following convention
<li><a href="maj13.htm">Death</a> - ending, closing down
</ul>
<a name="reinforce"><b><a href="#r">REINFORCING CARDS: Some Possibilities</a></b></a>
<ul>
<li><a href="maj17.htm">Star</a> - faith, trust
<li><a href="maj20.htm">Judgement</a> - rebirth
</ul>
<a name="description"><b><a href="howcard.htm#howdesc">DESCRIPTION</a></b></a>
<p>The Fool stands at the beginning of the journey, ready to leap into the unknown.
</body></html>"""

_C7_HTML = """\
<html><head><title>SEVEN OF CUPS</title></head><body>
<h1>SEVEN OF CUPS</h1>
<ul>
<li><b>Wishful Thinking</b>
<li><b>Options</b>
<li><b>Dissipation</b>
</ul>
<img src="rbowline.gif">
<a href="#actions"><b>[ Actions ]</b></a>
<a href="#description"><b>[ Description ]</b></a>
<img src="rbowline.gif">
<a name="actions"><b><a href="howcard.htm#howactions">ACTIONS</a></b></a>
<dl>
<dt>indulging in wishful thinking
<dt>weighing many options
</dl>
<a name="opposite"><b><a href="#o">OPPOSING CARDS: Some Possibilities</a></b></a>
<ul>
<li><a href="maj01.htm">Magician</a> - focus and commitment
<li><a href="maj04.htm">Emperor</a> - discipline, structure
</ul>
<a name="reinforce"><b><a href="#r">REINFORCING CARDS: Some Possibilities</a></b></a>
<ul>
<li><a href="maj15.htm">Devil</a> - excess, illusion
<li><a href="maj18.htm">Moon</a> - fantasy
</ul>
<a name="description"><b><a href="howcard.htm#howdesc">DESCRIPTION</a></b></a>
<p>The Seven of Cups warns against wishful thinking when faced with many options.
</body></html>"""

# Court cards on learntarot have ACTIONS + DESCRIPTION only — no opposing/reinforcing.
_WPG_HTML = """\
<html><head><title>PAGE OF WANDS</title></head><body>
<h1>PAGE OF WANDS</h1>
<ul>
<li><b>Be Creative</b>
<li><b>Be Enthusiastic</b>
</ul>
<img src="rbowline.gif">
<a href="#actions"><b>[ Actions ]</b></a>
<a href="#description"><b>[ Description ]</b></a>
<img src="rbowline.gif">
<a name="actions"><b><a href="howcard.htm#howactions">ACTIONS</a></b></a>
<dl>
<dt>taking a creative approach
<dt>exploring with enthusiasm
</dl>
<a name="description"><b><a href="howcard.htm#howdesc">DESCRIPTION</a></b></a>
<p>The Page of Wands brings opportunities for passion and creative adventure.
</body></html>"""


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
        assert self.card.keywords == ["Beginning", "Spontaneity", "Faith"]

    def test_actions_non_empty(self) -> None:
        assert "beginning a new phase" in self.card.actions
        assert "striking out on a new path" in self.card.actions

    def test_opposing_names_parsed(self) -> None:
        # Names only — the "- meaning" sub-lines are dropped, no "The" prefix.
        assert self.card.opposing_names == ["Hierophant", "Death"]

    def test_reinforcing_names_parsed(self) -> None:
        assert self.card.reinforcing_names == ["Star", "Judgement"]

    def test_nav_menu_not_treated_as_section(self) -> None:
        # The bracketed "[ Actions ]" nav links must not create sections.
        assert "[ Actions ]" not in self.card.actions

    def test_description_non_empty(self) -> None:
        assert "leap into the unknown" in self.card.description

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
        assert self.card.keywords == ["Wishful Thinking", "Options", "Dissipation"]

    def test_actions_non_empty(self) -> None:
        assert "indulging in wishful thinking" in self.card.actions

    def test_opposing_names_parsed(self) -> None:
        assert self.card.opposing_names == ["Magician", "Emperor"]

    def test_reinforcing_names_parsed(self) -> None:
        assert self.card.reinforcing_names == ["Devil", "Moon"]

    def test_description_non_empty(self) -> None:
        assert "Seven of Cups" in self.card.description

    def test_source_url_correct(self) -> None:
        assert self.card.source_url == "https://www.learntarot.com/c7.htm"


# ---------------------------------------------------------------------------
# Page of Wands — court card (no opposing/reinforcing sections)
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
        assert self.card.keywords == ["Be Creative", "Be Enthusiastic"]

    def test_actions_non_empty(self) -> None:
        assert "taking a creative approach" in self.card.actions

    def test_court_cards_have_no_opposing_or_reinforcing(self) -> None:
        # learntarot court pages omit these sections entirely.
        assert self.card.opposing_names == []
        assert self.card.reinforcing_names == []

    def test_description_non_empty(self) -> None:
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

    def test_missing_actions_heading_raises_value_error(self) -> None:
        """A page with DESCRIPTION but no ACTIONS heading must raise."""
        html = (
            "<html><body><h1>X</h1><ul><li><b>kw</b></ul>"
            "<p>DESCRIPTION</p><p>Some description.</p></body></html>"
        )
        with pytest.raises(ValueError, match="missing required heading"):
            parse_card_page(html, "maj00")

    def test_missing_description_heading_raises_value_error(self) -> None:
        """A page with ACTIONS but no DESCRIPTION heading must raise."""
        html = (
            "<html><body><h1>X</h1><ul><li><b>kw</b></ul>"
            "<p>ACTIONS</p><p>Do something.</p></body></html>"
        )
        with pytest.raises(ValueError, match="missing required heading"):
            parse_card_page(html, "maj00")

    def test_no_section_headings_raises_value_error(self) -> None:
        """A page with no recognised headings at all must raise."""
        html = "<html><body><p>Some random content with no structure.</p></body></html>"
        with pytest.raises(ValueError, match="No section headings"):
            parse_card_page(html, "maj00")

    def test_empty_html_raises_error(self) -> None:
        with pytest.raises(ValueError):
            parse_card_page("", "maj00")

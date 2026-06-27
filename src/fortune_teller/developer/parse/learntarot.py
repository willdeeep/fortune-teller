"""HTML parser for learntarot.com Rider-Waite card pages.

Page structure
~~~~~~~~~~~~~~
Each page has an ``<h1>`` title, a ``<ul>`` of bold keyword items, then a
bracketed nav menu (``[ Actions ]`` links — ignored) followed by the real
sections. Section headings are **uppercase anchor labels on their own line**
(no brackets), each preceded by an ``<a name="...">`` anchor:

    ``ACTIONS``
    ``OPPOSING CARDS: Some Possibilities``
    ``REINFORCING CARDS: Some Possibilities``
    ``DESCRIPTION``

Content for each section follows until the next heading (or the end of the
page). Court cards omit the opposing/reinforcing sections. Opposing and
reinforcing cards are ``<li><a href="<slug>.htm">Name</a>`` items — the card
identity comes from the **href slug** (see :func:`_extract_synergy_slugs`), not
the display text, so typos/markup in the link text don't matter.

URL scheme
~~~~~~~~~~
Pages live at ``https://www.learntarot.com/<slug>.htm`` where slug is a
learntarot-internal abbreviation (e.g. ``maj00``, ``c7``, ``wpg``).
"""

from __future__ import annotations

import re
from urllib.parse import urljoin

from pydantic import BaseModel, ConfigDict
from selectolax.parser import HTMLParser, Node

from fortune_teller.application.models.domain import Arcana, Suit

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RW_BASE_URL = "https://www.learntarot.com"

# Section headings are uppercase anchor labels on their own line (no brackets),
# e.g. "ACTIONS" or "OPPOSING CARDS: Some Possibilities". A separate nav menu near
# the top uses bracketed, mixed-case links ("[ Actions ]") which must NOT match —
# hence the all-caps, whole-line (MULTILINE-anchored) pattern.
_HEADING_RE = re.compile(
    r"^(ACTIONS|OPPOSING CARDS|REINFORCING CARDS|DESCRIPTION)(?::.*)?$",
    re.MULTILINE,
)

# ---------------------------------------------------------------------------
# URL → identity map
# ---------------------------------------------------------------------------

# Major arcana: slug → (id, name, arcana, suit, number)
_MAJOR_ENTRIES: list[tuple[str, str, str, int]] = [
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
]

# Number words for slug construction (pip cards)
_NUMBER_WORDS: dict[str, str] = {
    "a": "ace",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
    "10": "ten",
}

# Court rank mapping: suffix → (number, word)
_COURT_RANKS: dict[str, tuple[int, str]] = {
    "pg": (11, "page"),
    "kn": (12, "knight"),
    "qn": (13, "queen"),
    "kg": (14, "king"),
}

# Suit prefix mapping: prefix → (Suit, name)
_SUIT_PREFIXES: dict[str, tuple[Suit, str]] = {
    "w": (Suit.WANDS, "wands"),
    "c": (Suit.CUPS, "cups"),
    "s": (Suit.SWORDS, "swords"),
    "p": (Suit.PENTACLES, "pentacles"),
}


def _build_rw_card_map() -> dict[str, tuple[str, str, Arcana, Suit | None, int | None]]:
    """Build the complete 78-card URL → identity mapping.

    Returns:
        Dict mapping learntarot slug to ``(id, name, arcana, suit, number)``.
    """
    result: dict[str, tuple[str, str, Arcana, Suit | None, int | None]] = {}

    # Major arcana
    for slug, card_id, name, number in _MAJOR_ENTRIES:
        result[slug] = (card_id, name, Arcana.MAJOR, None, number)

    # Minor arcana — pips
    for prefix, (suit, suit_name) in _SUIT_PREFIXES.items():
        for rank_suffix, word in _NUMBER_WORDS.items():
            slug = f"{prefix}{rank_suffix}"
            card_id = f"{word}-of-{suit_name}"
            name = f"{word.capitalize()} of {suit_name.capitalize()}"
            number = _number_word_to_int(word)
            result[slug] = (card_id, name, Arcana.MINOR, suit, number)

        # Minor arcana — court cards
        for rank_suffix, (number, word) in _COURT_RANKS.items():
            slug = f"{prefix}{rank_suffix}"
            card_id = f"{word}-of-{suit_name}"
            name = f"{word.capitalize()} of {suit_name.capitalize()}"
            result[slug] = (card_id, name, Arcana.MINOR, suit, number)

    return result


def _number_word_to_int(word: str) -> int:
    """Convert a number word to its integer value (1-10)."""
    mapping = {
        "ace": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }
    return mapping[word]


RW_CARD_MAP: dict[str, tuple[str, str, Arcana, Suit | None, int | None]] = _build_rw_card_map()
"""Mapping of learntarot slugs to ``(id, name, arcana, suit, number)`` tuples.

Contains exactly 78 entries (22 major + 56 minor).
"""


# ---------------------------------------------------------------------------
# Synergy extraction (DOM, by href slug)
# ---------------------------------------------------------------------------


def _extract_synergy_slugs(body: Node) -> tuple[list[str], list[str]]:
    """Return ``(opposing_slugs, reinforcing_slugs)`` from a learntarot card page.

    The opposing/reinforcing lists are ``<li><a href="<slug>.htm">Name</a>``
    items under the ``<a name="opposite">`` / ``<a name="reinforce">`` section
    anchors.  The card identity lives in the **href slug** (``maj00``, ``s4``),
    not the link text — so this is immune to the display-text quirks that broke
    the old text parser (split/mangled headings, prose, missing "of", typos like
    ``"Four o Swords"``, leaked markup).  Only hrefs whose slug is a known
    :data:`RW_CARD_MAP` key are kept, which also filters out the ``howcard.htm``
    heading links.

    Court pages omit these sections, yielding two empty lists.  The Six of
    Pentacles page combines both under one heading; its shared list flows to
    *reinforcing* (opposing empty), with no junk.
    """
    opposing: list[str] = []
    reinforcing: list[str] = []
    section: str | None = None
    for anchor in body.css("a"):
        name = anchor.attributes.get("name")
        if name in {"actions", "opposite", "reinforce", "description"}:
            section = name
            continue
        if section not in {"opposite", "reinforce"}:
            continue
        href = anchor.attributes.get("href") or ""
        match = re.fullmatch(r"([a-z0-9]+)\.htm", href, re.IGNORECASE)
        if match is None:
            continue
        slug = match.group(1).lower()
        if slug not in RW_CARD_MAP:
            continue
        (opposing if section == "opposite" else reinforcing).append(slug)
    return opposing, reinforcing


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class RawCard(BaseModel):
    """A Rider-Waite card as parsed directly from learntarot.com HTML.

    This is a dev-only model used *before* the structured-section parsing
    (which is handled in a later pipeline step).  The fields mirror the
    raw structure of the learntarot.com page.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    arcana: Arcana
    suit: Suit | None
    number: int | None
    keywords: list[str]
    actions: list[str]
    opposing_slugs: list[str]
    reinforcing_slugs: list[str]
    description: str
    source_url: str
    image_url: str | None = None


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def _extract_section_text(sections: dict[str, str], heading: str) -> str:
    """Return the cleaned text for *heading*, or empty string if missing."""
    raw = sections.get(heading, "")
    return re.sub(r"\s+", " ", raw).strip()


def _extract_actions(raw: str) -> list[str]:
    """Split *raw* section text into a flat list of action phrases.

    Splits by newlines first, then by commas within each line.
    """
    actions: list[str] = []
    for _line in raw.split("\n"):
        stripped_line = _line.strip()
        if not stripped_line:
            continue
        for _part in stripped_line.split(","):
            stripped_part = _part.strip()
            if stripped_part:
                actions.append(stripped_part)
    return actions if actions else [raw.strip()] if raw.strip() else []


def _extract_image_url(body: Node, base_url: str) -> str | None:
    """Extract the full-resolution image URL from a learntarot.com card page.

    First tries to find an ``<a>`` tag whose ``href`` points to a ``bigjpgs/``
    path (the full-resolution image).  Falls back to the first ``<img>`` tag
    whose ``src`` does not match known site-chrome patterns (``rbowline.gif``,
    ``rbowline.jpg``, or any filename starting with ``rbowline``).

    Args:
        body: The selectolax ``<body>`` node of the parsed page.
        base_url: Base URL for resolving relative links.

    Returns:
        The absolute image URL, or ``None`` if no suitable image found.
    """
    # 1. Look for <a href="bigjpgs/..."> — the full-resolution link
    for link in body.css("a"):
        href = link.attributes.get("href")
        if href and re.search(r"bigjpgs/.+\.(?:jpg|jpeg|gif|png|webp)$", href, re.IGNORECASE):
            return str(urljoin(base_url, href))

    # 2. Fallback: first <img> whose src is not site chrome
    for img in body.css("img"):
        src = img.attributes.get("src")
        if not src:
            continue
        filename = src.rsplit("/", 1)[-1].lower()
        if filename in ("rbowline.gif", "rbowline.jpg") or re.match(r"rbowline", filename):
            continue
        return str(urljoin(base_url, src))

    return None


def parse_card_page(html: str, slug: str) -> RawCard:
    """Parse a learntarot.com card page into a :class:`RawCard`.

    Args:
        html: Raw HTML of the card page.
        slug: learntarot slug, e.g. ``"maj00"`` or ``"c7"``.

    Returns:
        A fully populated :class:`RawCard` instance.

    Raises:
        ValueError: If *slug* is not in :data:`RW_CARD_MAP` or
            the page is structurally unexpected (missing a required
            bracketed heading).
    """
    identity = RW_CARD_MAP.get(slug)
    if identity is None:
        raise ValueError(f"Unknown learntarot slug: {slug!r}")
    card_id, name, arcana, suit, number = identity

    tree = HTMLParser(html)
    body = tree.body
    if body is None:
        raise ValueError(f"Page for slug '{slug}' has no body element")

    # Extract body text with newline separators to preserve structure
    full_text: str = body.text(strip=True, separator="\n")
    # Normalise internal whitespace per line
    lines = [re.sub(r"\s+", " ", line).strip() for line in full_text.split("\n")]
    full_text = "\n".join(line for line in lines if line)

    # Find bracketed headings with their positions
    headings: list[tuple[int, str, str]] = []
    for match in _HEADING_RE.finditer(full_text):
        headings.append((match.start(), match.group(0), match.group(1).strip()))

    if not headings:
        raise ValueError(
            f"No section headings found in page for slug '{slug}'. "
            "Expected at least ACTIONS and DESCRIPTION."
        )

    # Keywords are the bold items of the first <ul> list (just after the <h1>
    # title). The nav menu and section bodies are not lists, so the first <ul>
    # is unambiguous.
    first_ul = body.css_first("ul")
    keywords = (
        [li.text(strip=True) for li in first_ul.css("li") if li.text(strip=True)]
        if first_ul is not None
        else []
    )

    # Split content into named sections
    sections: dict[str, str] = {}
    for i, (start_pos, heading_text, heading_name) in enumerate(headings):
        next_start = headings[i + 1][0] if i + 1 < len(headings) else len(full_text)
        content_start = start_pos + len(heading_text)
        content = full_text[content_start:next_start].strip()
        sections[heading_name] = content

    # Validate required sections
    for required in ("ACTIONS", "DESCRIPTION"):
        if required not in sections:
            raise ValueError(
                f"Page for slug '{slug}' is missing required heading [{required}]. "
                f"Found headings: {list(sections)}"
            )

    # Extract actions — process BEFORE whitespace normalisation to preserve line breaks
    actions_raw = sections.get("ACTIONS", "")
    actions = _extract_actions(actions_raw)

    # Opposing/reinforcing card identities come from the <a href="<slug>.htm">
    # links under each section anchor — the href slug is authoritative, not the
    # display text. Description is still parsed from the text section.
    opposing_slugs, reinforcing_slugs = _extract_synergy_slugs(body)
    description = _extract_section_text(sections, "DESCRIPTION")

    source_url = f"{_RW_BASE_URL}/{slug}.htm"
    image_url = _extract_image_url(body, _RW_BASE_URL)

    return RawCard(
        id=card_id,
        name=name,
        arcana=arcana,
        suit=suit,
        number=number,
        keywords=keywords,
        actions=actions,
        opposing_slugs=opposing_slugs,
        reinforcing_slugs=reinforcing_slugs,
        description=description,
        source_url=source_url,
        image_url=image_url,
    )

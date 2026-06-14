"""HTML parser for learntarot.com Rider-Waite card pages.

Page structure
~~~~~~~~~~~~~~
Each page has a title (``<h2>``), a keyword list, and then bracketed
headings that divide the content into sections:

    ``[ACTIONS]``
    ``[OPPOSING CARDS: Some Possibilities]``
    ``[REINFORCING CARDS: Some Possibilities]``
    ``[DESCRIPTION]``

Content for each section follows until the next bracketed heading (or the
end of the page).

URL scheme
~~~~~~~~~~
Pages live at ``https://www.learntarot.com/<slug>.htm`` where slug is a
learntarot-internal abbreviation (e.g. ``maj00``, ``c7``, ``wpg``).
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict
from selectolax.parser import HTMLParser

from fortune_teller.application.models.domain import Arcana, Suit

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RW_BASE_URL = "https://www.learntarot.com"

# Regex matching bracketed headings like [ACTIONS] or [OPPOSING CARDS: subtitle]
_HEADING_RE = re.compile(r"\[([A-Z\s]+)(?::[^\]]*)?\]")

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
    opposing_names: list[str]
    reinforcing_names: list[str]
    description: str
    source_url: str


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def _extract_section_text(sections: dict[str, str], heading: str) -> str:
    """Return the cleaned text for *heading*, or empty string if missing."""
    raw = sections.get(heading, "")
    return re.sub(r"\s+", " ", raw).strip()


def _split_names(text: str) -> list[str]:
    """Split a comma-separated list of card names from *text*.

    Returns:
        List of stripped, non-empty name strings.
    """
    return [n.strip() for n in text.split(",") if n.strip()]


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
            f"No bracketed headings found in page for slug '{slug}'. "
            "Expected at least [ACTIONS], [DESCRIPTION], etc."
        )

    # Extract keywords from text before the first bracketed heading
    pre_text = full_text[: headings[0][0]].strip()
    # Look for "Keywords:" or "keywords:" prefix
    kw_match = re.search(r"[Kk]eywords?:?\s*(.*)", pre_text)
    if kw_match:
        kw_text = kw_match.group(1).strip()
        keywords = [k.strip() for k in kw_text.split(",") if k.strip()]
    else:
        # Fall back to entire pre-text as keywords
        keywords = [k.strip() for k in pre_text.split(",") if k.strip()] if pre_text else []

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

    # Extract opposing / reinforcing / description with whitespace normalisation
    opposing_names = _split_names(_extract_section_text(sections, "OPPOSING CARDS"))
    reinforcing_names = _split_names(_extract_section_text(sections, "REINFORCING CARDS"))
    description = _extract_section_text(sections, "DESCRIPTION")

    source_url = f"{_RW_BASE_URL}/{slug}.htm"

    return RawCard(
        id=card_id,
        name=name,
        arcana=arcana,
        suit=suit,
        number=number,
        keywords=keywords,
        actions=actions,
        opposing_names=opposing_names,
        reinforcing_names=reinforcing_names,
        description=description,
        source_url=source_url,
    )

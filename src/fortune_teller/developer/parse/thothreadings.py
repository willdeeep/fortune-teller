"""HTML parser for thothreadings.com card and spread pages.

Page structure (cards)
~~~~~~~~~~~~~~~~~~~~~~
The ``.entry-content`` div contains a series of ``<p>`` elements.  Structured
sections are encoded as ``<p><strong>Label:</strong>text</p>``.  The overall
description is the run of ``<p>`` tags that appear *before* the first labelled
section.

The following labels are recognised (case-insensitive):

    Drive / Light / Shadow / Reversed / Keywords /
    Advice (or "Advice or Instructions") / Questions (singular or plural) /
    Suggestion (alias for Proposal) / Revelation (alias for Confirmation) /
    Affirmation

Page structure (spread)
~~~~~~~~~~~~~~~~~~~~~~~
The New Moon spread page uses ``<h3>`` tags whose text matches
``"Card N - <position name>"``.  Everything before the next ``<h3>`` (or the
end of content) is treated as the position's meaning text.

URL scheme
~~~~~~~~~~
Card and spread pages both live at ``https://thothreadings.com/<slug>/``.
"""

from __future__ import annotations

import re
import unicodedata
from typing import cast

from selectolax.parser import HTMLParser, Node

from fortune_teller.application.models.domain import (
    Arcana,
    Card,
    CardSection,
    CardSectionText,
    Spread,
    SpreadPosition,
    Suit,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BASE_URL = "https://thothreadings.com"

# Maps the bold label found on the page to our CardSection enum value.
# Keys are lower-cased and stripped; the first match wins.
_LABEL_MAP: list[tuple[str, CardSection]] = [
    ("drive", CardSection.DRIVE),
    ("light", CardSection.LIGHT),
    ("shadow", CardSection.SHADOW),
    ("reversed", CardSection.REVERSED),
    ("keywords", CardSection.KEYWORDS),
    ("keyword", CardSection.KEYWORDS),
    ("advice or", CardSection.ADVICE),
    ("advice", CardSection.ADVICE),
    ("instructions", CardSection.ADVICE),
    ("question", CardSection.QUESTION),  # singular or plural
    ("suggestion", CardSection.PROPOSAL),
    ("proposal", CardSection.PROPOSAL),
    ("revelation", CardSection.CONFIRMATION),
    ("confirmation", CardSection.CONFIRMATION),
    ("affirmation", CardSection.AFFIRMATION),
]

# Minor arcana suits keyed on substring of slug/name
_SUIT_SLUGS: dict[str, Suit] = {
    "wands": Suit.WANDS,
    "cups": Suit.CUPS,
    "swords": Suit.SWORDS,
    "disks": Suit.DISKS,
    "pentacles": Suit.DISKS,
    "coins": Suit.DISKS,
}

# Court card titles used in the Book of Thoth
_COURT_TITLES: set[str] = {"princess", "prince", "queen", "knight", "king", "page"}

# Numeric word prefixes for minor arcana pip cards
_NUMBER_WORDS: dict[str, int] = {
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

# Major arcana slug -> number (supports both blog slugs and legacy root slugs)
_MAJOR_NUMBERS: dict[str, int] = {
    # blog slugs (current)
    "0-the-fool": 0,
    "i-the-magician": 1,
    "ii-the-high-priestess": 2,
    "iii-the-empress": 3,
    "iv-the-emperor": 4,
    "v-the-hierophant": 5,
    "vi-the-lovers": 6,
    "vii-the-chariot": 7,
    "viii-adjustment": 8,
    "ix-the-hermit": 9,
    "x-the-wheel-of-fortune": 10,
    "xi-the-passion-lust": 11,
    "xii-the-hanged-man": 12,
    "xiii-the-death": 13,
    "xiv-the-art": 14,
    "xv-the-devil": 15,
    "xvi-the-tower": 16,
    "xvii-the-star": 17,
    "xviii-the-moon": 18,
    "xix-the-sun": 19,
    "xx-the-aeon": 20,
    "xxi-the-universe": 21,
    # legacy root slugs (kept for backward compat / fixture tests)
    "the-fool": 0,
    "the-magus": 1,
    "the-high-priestess": 2,
    "the-empress": 3,
    "the-emperor": 4,
    "the-hierophant": 5,
    "the-lovers": 6,
    "the-chariot": 7,
    "adjustment": 8,
    "the-hermit": 9,
    "fortune": 10,
    "lust": 11,
    "the-hanged-man": 12,
    "death": 13,
    "art": 14,
    "the-devil": 15,
    "the-tower": 16,
    "the-star": 17,
    "the-moon": 18,
    "the-sun": 19,
    "the-aeon": 20,
    "the-universe": 21,
}

# Matches leading Roman-numeral (i through xxi) or digit prefix before a hyphen
_SLUG_PREFIX_RE = re.compile(
    r"^(?:\d+|(?:i{1,3}|iv|vi{0,3}|ix|xi{0,3}|xiv|xv|xvi{0,3}|xix|xx{0,2}|xxi))-",
    re.IGNORECASE,
)

# Site-chrome image substrings to exclude from card image selection.
_CHROME_DENYLIST = ("astrologist-illustration", "Screen-Shot", "logo", "branding")

# Regex to strip WordPress size suffixes like -480x749 from image URLs.
_SIZE_SUFFIX_RE = re.compile(r"-\d+x\d+(?=\.\w+$)", re.IGNORECASE)


def _extract_image_url(content: Node) -> str | None:
    """Extract the card artwork URL from the entry-content node.

    Heuristic:
    1. Collect ``<img>`` nodes whose ``src`` is under ``/wp-content/uploads/``.
    2. Exclude known site-chrome by filename denylist.
    3. Strip ``-WxH`` size suffixes from ``src`` to get the full-resolution URL.
    4. Return the first remaining candidate, or ``None`` if no match.
    """
    candidates: list[str] = []
    for img in content.css("img"):
        src = (img.attributes.get("src") or "").strip()
        if not src or "/wp-content/uploads/" not in src:
            continue
        filename = src.rsplit("/", 1)[-1].lower()
        if any(deny in filename for deny in _CHROME_DENYLIST):
            continue
        full_url = _SIZE_SUFFIX_RE.sub("", src)
        candidates.append(full_url)
    return candidates[0] if candidates else None


# Pattern to match spread position headings
_POSITION_RE = re.compile(r"Card\s+(\d+)\s*[-\u2013]\s*(.+)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalise(text: str) -> str:
    """Normalise unicode, strip surrounding whitespace, collapse internal spaces."""
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", text).strip()


def _slug_to_name(slug: str) -> str:
    """Convert a URL slug to a human-readable card name.

    Strips leading Roman-numeral or digit prefixes used in the blog URL
    scheme (e.g. ``"0-the-fool"`` -> ``"The Fool"``,
    ``"i-the-magician"`` -> ``"The Magician"``).

    Examples::

        "0-the-fool"            -> "The Fool"
        "i-the-magician"        -> "The Magician"
        "ace-of-wands"          -> "Ace of Wands"
        "two-of-wands-dominion" -> "Two of Wands Dominion"
    """
    clean = _SLUG_PREFIX_RE.sub("", slug)
    # "of" is a stop word (lowercase) but "the" should be capitalised at
    # the start of a name (e.g. "The Fool", "The Magician").
    stop_words = {"of"}
    parts = clean.split("-")
    return " ".join(word if word in stop_words else word.capitalize() for word in parts)


def _detect_arcana_and_suit(slug: str) -> tuple[Arcana, Suit | None]:
    """Infer arcana and suit from the slug."""
    for suit_key, suit in _SUIT_SLUGS.items():
        if suit_key in slug:
            return Arcana.MINOR, suit
    return Arcana.MAJOR, None


def _detect_number(slug: str, arcana: Arcana) -> int | None:
    """Infer card number from slug."""
    if arcana == Arcana.MAJOR:
        return _MAJOR_NUMBERS.get(slug)
    parts = slug.split("-")
    for part in parts:
        if part in _NUMBER_WORDS:
            return _NUMBER_WORDS[part]
        if part in _COURT_TITLES:
            court_map = {
                "princess": 11,
                "prince": 12,
                "queen": 13,
                "knight": 14,
                "king": 14,
                "page": 11,
            }
            return court_map.get(part)
    return None


def _match_label(raw_label: str) -> CardSection | None:
    """Return the CardSection for *raw_label*, or ``None`` if unrecognised."""
    key = raw_label.lower().strip().rstrip(":")
    for prefix, section in _LABEL_MAP:
        if key.startswith(prefix):
            return section
    return None


def _get_entry_content(tree: HTMLParser) -> Node | None:
    """Return the ``.entry-content`` node, or ``None`` if not found."""
    return cast(Node | None, tree.css_first(".entry-content"))


def _extract_strong_label(para: Node) -> tuple[str, str] | None:
    """If *para* starts with a ``<strong>`` label, return ``(label, rest_text)``.

    Returns ``None`` if the paragraph has no strong prefix.
    """
    strong = para.css_first("strong")
    if strong is None:
        return None
    label = _normalise(strong.text(strip=True))
    if not label:
        return None
    full = _normalise(para.text(strip=True))
    rest = full[len(label) :].strip().lstrip(":").strip()
    return label, rest


# ---------------------------------------------------------------------------
# Card parser
# ---------------------------------------------------------------------------


def parse_card_page(html: str, slug: str) -> Card:
    """Parse a thothreadings.com card page into a :class:`Card` model.

    Args:
        html: Raw HTML of the card page.
        slug: URL slug, e.g. ``"0-the-fool"`` or ``"ace-of-wands"``.

    Returns:
        A fully populated :class:`Card` instance.

    Raises:
        ValueError: If the ``.entry-content`` element cannot be found.
    """
    tree = HTMLParser(html)
    content = _get_entry_content(tree)
    if content is None:
        raise ValueError(f"Could not find .entry-content on page for slug '{slug}'")

    arcana, suit = _detect_arcana_and_suit(slug)
    number = _detect_number(slug, arcana)
    source_url = f"{_BASE_URL}/{slug}/"

    sections: list[CardSectionText] = []
    overall_parts: list[str] = []
    seen_sections: set[CardSection] = set()
    found_first_labelled = False

    for para in content.css("p"):
        label_result = _extract_strong_label(para)
        if label_result is not None:
            label, text = label_result
            section = _match_label(label)
            if section is not None:
                found_first_labelled = True
                if text and section not in seen_sections:
                    sections.append(CardSectionText(section=section, text=text))
                    seen_sections.add(section)
                continue

        # No matching label — accumulate as overall text before first section
        if not found_first_labelled:
            text = _normalise(para.text(strip=True))
            if text:
                overall_parts.append(text)

    # Affirmation sometimes lives in an <h4> element
    if CardSection.AFFIRMATION not in seen_sections:
        for h4 in content.css("h4"):
            text = _normalise(h4.text(strip=True))
            if text.lower().startswith("affirmation"):
                rest = text[len("affirmation") :].strip().lstrip(":").strip()
                if rest:
                    sections.append(CardSectionText(section=CardSection.AFFIRMATION, text=rest))
                    seen_sections.add(CardSection.AFFIRMATION)
                    break

    overall = " ".join(overall_parts).strip()
    if overall and CardSection.OVERALL not in seen_sections:
        sections.insert(0, CardSectionText(section=CardSection.OVERALL, text=overall))

    image_url = _extract_image_url(content)

    return Card(
        id=slug,
        name=_slug_to_name(slug),
        arcana=arcana,
        suit=suit,
        number=number,
        sections=sections,
        source_url=source_url,
        image_url=image_url,
    )


# ---------------------------------------------------------------------------
# Spread parser
# ---------------------------------------------------------------------------


def parse_spread_page(html: str, spread_id: str, spread_name: str) -> Spread:
    """Parse the New Moon spread page into a :class:`Spread` model.

    Positions are extracted from ``<h3>`` elements matching
    ``"Card N - <name>"`` patterns (hyphens or en-dashes).

    Args:
        html:        Raw HTML of the spread page.
        spread_id:   Identifier for the spread (e.g. ``"new-moon-three-card"``).
        spread_name: Human-readable name.

    Returns:
        A :class:`Spread` with one :class:`SpreadPosition` per card slot.

    Raises:
        ValueError: If no position headings are found or content is missing.
    """
    tree = HTMLParser(html)
    content = _get_entry_content(tree)
    if content is None:
        raise ValueError("Could not find .entry-content on spread page.")

    source_url = f"{_BASE_URL}/spread-new-moon/"
    positions: list[SpreadPosition] = []
    seen_indices: set[int] = set()

    all_nodes = list(content.css("h3, p"))
    current_match: re.Match[str] | None = None
    meaning_parts: list[str] = []

    def _flush() -> None:
        nonlocal current_match, meaning_parts
        if current_match is None:
            return
        idx = int(current_match.group(1)) - 1  # convert to 0-based
        name = _normalise(current_match.group(2))
        meaning = " ".join(meaning_parts).strip() or name
        if idx not in seen_indices:
            positions.append(
                SpreadPosition(
                    index=idx,
                    name=name,
                    meaning=meaning,
                    source_url=source_url,
                )
            )
            seen_indices.add(idx)
        current_match = None
        meaning_parts = []

    for node in all_nodes:
        if node.tag == "h3":
            _flush()
            text = _normalise(node.text(strip=True))
            m = _POSITION_RE.match(text)
            if m:
                current_match = m
        elif node.tag == "p" and current_match is not None:
            text = _normalise(node.text(strip=True))
            if text:
                meaning_parts.append(text)

    _flush()

    if not positions:
        raise ValueError("No spread positions found in spread page HTML.")

    positions.sort(key=lambda p: p.index)

    return Spread(id=spread_id, name=spread_name, positions=positions)

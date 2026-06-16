"""Rider-Waite normalisation pipeline — :class:`RawCard` → :class:`Card`.

Pipeline stages
~~~~~~~~~~~~~~~

0. **Deterministic** (no LLM): keywords and overall description are copied
   directly from the raw card.

1. **Re-bucket** (LLM, temperature 0): the card's keywords, actions, and
   description are sorted into ``light``, ``shadow``, ``advice``, and
   ``reversed`` sections.

2. **Gap-fill** (LLM, low temperature): remaining empty sections (``drive``,
   ``question``, ``proposal``, ``confirmation``, ``affirmation``) are filled
   using the card's own material.

Each stage records provenance for every emitted section.
"""

from __future__ import annotations

import json
import re
import typing
from enum import StrEnum
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel, ConfigDict, HttpUrl

from fortune_teller.application.config import settings
from fortune_teller.application.models.domain import (
    Card,
    CardSection,
    CardSectionText,
)
from fortune_teller.developer.normalize.prompts import (
    GAPFILL_HUMAN,
    GAPFILL_SYSTEM,
    REBUCKET_HUMAN,
    REBUCKET_SYSTEM,
)
from fortune_teller.developer.parse.learntarot import RawCard, resolve_card_names

# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


class Provenance(StrEnum):
    """How a card section was produced."""

    DETERMINISTIC = "deterministic"
    REBUCKETED = "rebucketed"
    SYNTHESIZED = "synthesized"


class CardProvenance(BaseModel):
    """Tracks the origin of each section in a normalised card."""

    model_config = ConfigDict(frozen=True)

    card_id: str
    sections: dict[str, Provenance]  # section_name → provenance


# ---------------------------------------------------------------------------
# Chat model factory
# ---------------------------------------------------------------------------


def build_normalize_model(
    provider: str = "api",
    model: str = "claude-sonnet-4-6",
) -> Runnable[Any, Any]:
    """Build a chat model for normalisation.

    Args:
        provider: ``"api"`` for Claude via langchain-anthropic, ``"local"``
            for llama-server.
        model: Model identifier.

    Returns:
        A LangChain chat runnable.
    """
    if provider == "local":
        # Lazy import to avoid hard dep at package level.
        from fortune_teller.application.chains.per_card import (  # noqa: PLC0415
            build_chat_model,
        )

        return build_chat_model()

    # provider == "api" — use Claude via langchain-anthropic
    from langchain_anthropic import ChatAnthropic  # noqa: PLC0415

    return ChatAnthropic(
        model=model,
        temperature=0.0,
        anthropic_api_key=settings.anthropic_api_key,
        timeout=180,
    )


# ---------------------------------------------------------------------------
# LLM response parsing
# ---------------------------------------------------------------------------

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _parse_json_object(content: str) -> dict[str, str]:
    """Parse a JSON object from an LLM response.

    Models often wrap JSON in a Markdown code fence or add a sentence of
    preamble despite being told "JSON only". Unwrap a fence if present;
    otherwise slice from the first ``{`` to the last ``}``; then parse.

    Raises:
        ValueError: if no JSON object can be parsed (message includes a
            snippet of the offending response for debugging).
    """
    text = content.strip()
    fence = _JSON_FENCE_RE.search(text)
    if fence:
        text = fence.group(1).strip()
    elif not text.startswith("{"):
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            text = text[start : end + 1]

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"LLM did not return valid JSON ({exc}). Response began: {content[:200]!r}"
        ) from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected a JSON object from the LLM, got {type(parsed).__name__}.")
    return typing.cast(dict[str, str], parsed)


# ---------------------------------------------------------------------------
# Re-bucket stage
# ---------------------------------------------------------------------------


def _rebucket(raw_card: RawCard, llm: Runnable[Any, Any]) -> dict[str, str]:
    """Invoke the LLM to sort card text into light/shadow/advice/reversed.

    Returns:
        A dict mapping section name to text (may be empty strings).
    """
    suit_info = f" of {raw_card.suit.value}" if raw_card.suit else ""
    arcana_label = raw_card.arcana.value
    human = REBUCKET_HUMAN.format(
        card_name=raw_card.name,
        arcana=arcana_label,
        suit_info=suit_info,
        keywords=", ".join(raw_card.keywords),
        actions=", ".join(raw_card.actions),
        description=raw_card.description,
    )

    messages = [
        SystemMessage(content=REBUCKET_SYSTEM),
        HumanMessage(content=human),
    ]
    response = llm.invoke(messages)
    raw_content = response.content if isinstance(response, AIMessage) else str(response)
    content = raw_content if isinstance(raw_content, str) else str(raw_content)

    return _parse_json_object(content)


# ---------------------------------------------------------------------------
# Gap-fill stage
# ---------------------------------------------------------------------------


def _gapfill(
    raw_card: RawCard,
    existing: dict[str, str],
    llm: Runnable[Any, Any],
) -> dict[str, str]:
    """Invoke the LLM to fill missing sections.

    Args:
        raw_card: The source raw card.
        existing: Already-populated sections (may include empties).
        llm: A LangChain chat runnable.

    Returns:
        A dict mapping section name to text (may be empty strings).
    """
    suit_info = f" of {raw_card.suit.value}" if raw_card.suit else ""
    arcana_label = raw_card.arcana.value

    existing_lines = "\n".join(f"{k}: {v}" for k, v in existing.items() if v) or "(none populated)"

    human = GAPFILL_HUMAN.format(
        card_name=raw_card.name,
        arcana=arcana_label,
        suit_info=suit_info,
        keywords=", ".join(raw_card.keywords),
        actions=", ".join(raw_card.actions),
        description=raw_card.description,
        existing_sections=existing_lines,
    )

    messages = [
        SystemMessage(content=GAPFILL_SYSTEM),
        HumanMessage(content=human),
    ]
    response = llm.invoke(messages)
    raw_content = response.content if isinstance(response, AIMessage) else str(response)
    content = raw_content if isinstance(raw_content, str) else str(raw_content)

    return _parse_json_object(content)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

_SECTION_NAMES_LIGHT = ("light", "shadow", "advice", "reversed")
_SECTION_NAMES_GAP = ("drive", "question", "proposal", "confirmation", "affirmation")


def normalize_card(
    raw_card: RawCard,
    llm: Runnable[Any, Any] | None = None,
) -> tuple[Card, CardProvenance, list[str]]:
    """Normalise a :class:`RawCard` into a Thoth-shaped :class:`Card`.

    Args:
        raw_card: The raw card as parsed from learntarot.com.
        llm: Optional LangChain chat runnable. If ``None``, only the
            deterministic stage (stage 0) runs.

    Returns:
        A ``(Card, CardProvenance, unresolved_names)`` tuple.
        ``unresolved_names`` lists any reinforce/oppose card names that
        could not be resolved to IDs.
    """
    sections: dict[str, str] = {}
    provenance: dict[str, Provenance] = {}

    # ---- Stage 0: deterministic ----
    sections["keywords"] = ", ".join(raw_card.keywords)
    provenance["keywords"] = Provenance.DETERMINISTIC

    sections["overall"] = raw_card.description
    provenance["overall"] = Provenance.DETERMINISTIC

    # Resolve reinforce/oppose names → IDs (deterministic, no LLM)
    reinforcing_ids, unres_reinforce = resolve_card_names(raw_card.reinforcing_names)
    opposing_ids, unres_oppose = resolve_card_names(raw_card.opposing_names)
    unresolved_names = unres_reinforce + unres_oppose

    # ---- Stage 1: re-bucket ----
    if llm is not None:
        rebucketed = _rebucket(raw_card, llm)
        for section_name in _SECTION_NAMES_LIGHT:
            text = rebucketed.get(section_name, "")
            if text:
                sections[section_name] = text
                provenance[section_name] = Provenance.REBUCKETED

    # ---- Stage 2: gap-fill ----
    if llm is not None:
        gapfilled = _gapfill(raw_card, sections, llm)
        for section_name in _SECTION_NAMES_GAP:
            text = gapfilled.get(section_name, "")
            if text:
                sections[section_name] = text
                provenance[section_name] = Provenance.SYNTHESIZED

    # ---- Assembly ----
    card_sections: list[CardSectionText] = []
    # Order sections according to CardSection enum for determinism
    priority_order = {
        CardSection.OVERALL: 0,
        CardSection.DRIVE: 1,
        CardSection.LIGHT: 2,
        CardSection.SHADOW: 3,
        CardSection.REVERSED: 4,
        CardSection.KEYWORDS: 5,
        CardSection.ADVICE: 6,
        CardSection.QUESTION: 7,
        CardSection.PROPOSAL: 8,
        CardSection.CONFIRMATION: 9,
        CardSection.AFFIRMATION: 10,
    }

    sorted_names = sorted(sections, key=lambda n: priority_order.get(CardSection(n), 99))
    for section_name in sorted_names:
        text = sections[section_name]
        if text:
            card_sections.append(CardSectionText(section=CardSection(section_name), text=text))

    card = Card(
        id=raw_card.id,
        name=raw_card.name,
        arcana=raw_card.arcana,
        suit=raw_card.suit,
        number=raw_card.number,
        sections=card_sections,
        reinforcing_ids=reinforcing_ids,
        opposing_ids=opposing_ids,
        source_url=HttpUrl(raw_card.source_url),
        image_url=raw_card.image_url,
    )

    return card, CardProvenance(card_id=raw_card.id, sections=dict(provenance)), unresolved_names


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def generate_report(
    results: list[tuple[Card, CardProvenance]],
    *,
    unresolved: list[tuple[str, str]] | None = None,
) -> str:
    """Generate a markdown normalisation report.

    Args:
        results: List of ``(Card, CardProvenance)`` tuples from
            :func:`normalize_deck`.
        unresolved: Optional list of ``(card_id, unresolved_name)`` tuples
            for reinforce/oppose names that could not be resolved to IDs.

    Returns:
        A markdown string.
    """
    total = len(results)
    synthesized_count = 0
    empty_count = 0

    lines: list[str] = [
        "# Rider-Waite Normalisation Report",
        "",
        f"**Total cards:** {total}",
        "",
        "## Per-card provenance",
        "",
        "| Card | Sections |",
        "|------|----------|",
    ]

    for card, prov in results:
        section_flags: list[str] = []
        for section_name, p in prov.sections.items():
            if p == Provenance.SYNTHESIZED:
                synthesized_count += 1
                section_flags.append(f"{section_name} 🤖")
            else:
                section_flags.append(section_name)
        # Track empty sections — any CardSection enum value not in prov.sections
        for cs in CardSection:
            if cs.value not in prov.sections:
                empty_count += 1
                section_flags.append(f"{cs.value} ⚠️")

        section_str = ", ".join(section_flags) if section_flags else "(none)"
        lines.append(f"| {card.name} | {section_str} |")

    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- **Synthesised sections (🤖):** {synthesized_count}",
            f"- **Empty sections (⚠️):** {empty_count}",
        ]
    )

    if unresolved:
        lines.extend(
            [
                "",
                "## Unresolved card names",
                "",
                "The following reinforce/oppose names could not be resolved to card IDs:",
                "",
                "| Card | Unresolved name |",
                "|------|-----------------|",
            ]
        )
        for card_id, name in unresolved:
            lines.append(f"| {card_id} | {name} |")

    lines.extend(
        [
            "",
            "## Re-run",
            "",
            "To re-run individual cards:",
            "",
            "```bash",
            "uv run ft-normalize-rw --only <card-id>[,<card-id>,...]",
            "```",
            "",
            "To re-run with local llama-server:",
            "",
            "```bash",
            "uv run ft-normalize-rw --provider local",
            "```",
        ]
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Deck-level entry point
# ---------------------------------------------------------------------------


def normalize_deck(
    raw_dir: Path,
    out_dir: Path,
    *,
    llm: Runnable[Any, Any] | None = None,
    only: set[str] | None = None,
) -> list[tuple[Card, CardProvenance]]:
    """Normalise all cards in a raw rider-waite directory.

    Args:
        raw_dir: Path to ``data/raw/rider-waite/``.
        out_dir: Path to ``data/parsed/rider-waite/``.
        llm: Optional LLM runnable. If ``None``, only deterministic stage runs.
        only: Optional set of card IDs to process (for re-running individuals).

    Returns:
        List of ``(Card, CardProvenance)`` tuples for every processed card.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    provenance_dir = out_dir / ".norm"
    provenance_dir.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(raw_dir.glob("*.json"))
    results: list[tuple[Card, CardProvenance]] = []

    failed: list[tuple[str, str]] = []
    all_unresolved: list[tuple[str, str]] = []  # (card_id, unresolved_name)

    for raw_path in raw_files:
        raw_data = json.loads(raw_path.read_text(encoding="utf-8"))
        raw_card = RawCard.model_validate(raw_data)

        if only is not None and raw_card.id not in only:
            continue

        # One bad LLM response shouldn't sink the whole 78-card batch — record
        # the failure and carry on so the rest still get written.
        try:
            card, prov, card_unresolved = normalize_card(raw_card, llm=llm)
        except Exception as exc:  # batch tool must be resilient: skip one bad card
            failed.append((raw_card.id, str(exc)))
            print(f"  ERROR normalising {raw_card.id}: {exc}")
            continue
        results.append((card, prov))
        all_unresolved.extend((raw_card.id, name) for name in card_unresolved)

        # Write Card JSON
        card_path = out_dir / f"{raw_card.id}.json"
        card_path.write_text(card.model_dump_json(indent=2), encoding="utf-8")

        # Write provenance sidecar
        prov_path = provenance_dir / f"{raw_card.id}.json"
        prov_path.write_text(prov.model_dump_json(indent=2), encoding="utf-8")

    if failed:
        ids = ",".join(cid for cid, _ in failed)
        print(f"\n{len(failed)} card(s) failed; re-run with: ft-normalize-rw --only {ids}")

    # Write deck metadata
    meta = {
        "id": "rider-waite",
        "name": "Rider-Waite-Smith",
        "card_count": len(results),
        "card_ids": [c.id for c, _ in results],
    }
    meta_path = out_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Write normalisation report
    report = generate_report(results, unresolved=all_unresolved)
    report_path = out_dir / "_normalization_report.md"
    report_path.write_text(report, encoding="utf-8")

    return results

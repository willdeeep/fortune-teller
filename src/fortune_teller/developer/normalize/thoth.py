"""Thoth reinforce/oppose LLM synthesis.

The Book of Thoth deck has no reinforce/oppose data in its HTML source.
This module uses an LLM to synthesize reinforcing and opposing card IDs
for each Thoth card, based on the card's own sections and the full deck
card list.
"""

from __future__ import annotations

import logging
import typing
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable

from fortune_teller.application.models.domain import Card, Deck
from fortune_teller.application.services.loading import load_deck
from fortune_teller.developer.normalize.prompts import SYNERGY_HUMAN, SYNERGY_SYSTEM
from fortune_teller.developer.normalize.rider_waite import (
    CardProvenance,
    Provenance,
    _parse_json_object,
)

logger = logging.getLogger(__name__)

_MAX_SYNERGY_IDS = 5


def _build_deck_card_list(deck: Deck) -> str:
    """Build a compact card list for the LLM prompt.

    Format: ``id — name`` per line.
    """
    return "\n".join(f"{c.id} — {c.name}" for c in deck.cards)


def _build_sections_text(card: Card) -> str:
    """Build a compact sections summary for the LLM prompt."""
    lines: list[str] = []
    for s in card.sections:
        lines.append(f"{s.section.value}: {s.text}")
    return "\n".join(lines)


def _validate_synergy_ids(
    ids: list[str],
    deck: Deck,
    card_id: str,
    *,
    max_ids: int = _MAX_SYNERGY_IDS,
) -> list[str]:
    """Validate and clean LLM-returned synergy IDs.

    - Remove self-references (card listing itself).
    - Remove IDs not in the deck.
    - Truncate to max_ids.
    """
    valid_ids = {c.id for c in deck.cards}
    cleaned = [i for i in ids if i != card_id and i in valid_ids]
    return cleaned[:max_ids]


def _parse_synergy_response(content: str) -> dict[str, Any]:
    """Parse a synergy JSON response from the LLM.

    Wraps :func:`_parse_json_object` but returns the parsed dict with
    relaxed value typing (values may be ``list[str]`` rather than ``str``).
    """
    return typing.cast(dict[str, Any], _parse_json_object(content))


def synthesize_card_synergies(
    card: Card,
    deck: Deck,
    llm: Runnable[Any, Any],
) -> tuple[list[str], list[str]]:
    """Synthesize reinforcing and opposing IDs for a single Thoth card.

    Args:
        card: The card to synthesize synergies for.
        deck: The full deck (used for validation and prompt context).
        llm: A LangChain chat runnable.

    Returns:
        A ``(reinforcing_ids, opposing_ids)`` tuple of validated card ID lists.
    """
    suit_info = f" of {card.suit.value}" if card.suit else ""
    arcana_label = card.arcana.value

    human = SYNERGY_HUMAN.format(
        card_name=card.name,
        arcana=arcana_label,
        suit_info=suit_info,
        sections_text=_build_sections_text(card),
        deck_card_list=_build_deck_card_list(deck),
    )

    messages = [
        SystemMessage(content=SYNERGY_SYSTEM),
        HumanMessage(content=human),
    ]
    response = llm.invoke(messages)
    raw_content = response.content if isinstance(response, AIMessage) else str(response)
    content = raw_content if isinstance(raw_content, str) else str(raw_content)

    parsed = _parse_synergy_response(content)

    reinforcing_ids = _validate_synergy_ids(
        parsed.get("reinforcing_ids", []),
        deck,
        card.id,
    )
    opposing_ids = _validate_synergy_ids(
        parsed.get("opposing_ids", []),
        deck,
        card.id,
    )

    return reinforcing_ids, opposing_ids


def synthesize_deck_synergies(
    parsed_dir: Path,
    *,
    llm: Runnable[Any, Any] | None = None,
    only: set[str] | None = None,
    deck_id: str = "book-of-thoth",
) -> list[tuple[Card, CardProvenance]]:
    """Synthesize reinforce/oppose IDs for all (or selected) Thoth cards.

    Reads existing parsed cards from ``parsed_dir / deck_id``, synthesizes
    synergy IDs via LLM, and writes updated Card JSON + provenance sidecars.

    Args:
        parsed_dir: Path to the parsed data directory (e.g. ``data/parsed``).
        llm: Optional LLM runnable. If ``None``, only deterministic stage runs
            (which means no synergy synthesis — IDs stay empty).
        only: Optional set of card IDs to process.
        deck_id: Deck identifier (default ``"book-of-thoth"``).

    Returns:
        List of ``(Card, CardProvenance)`` tuples for every processed card.
    """
    deck = load_deck(parsed_dir, deck_id)
    provenance_dir = parsed_dir / deck_id / ".norm"
    provenance_dir.mkdir(parents=True, exist_ok=True)

    results: list[tuple[Card, CardProvenance]] = []
    failed: list[tuple[str, str]] = []

    for card in deck.cards:
        if only is not None and card.id not in only:
            continue

        if llm is not None:
            try:
                reinforcing_ids, opposing_ids = synthesize_card_synergies(card, deck, llm)
            except Exception as exc:
                logger.warning("Failed to synthesize synergies for %s: %s", card.id, exc)
                failed.append((card.id, str(exc)))
                # Keep the card as-is (empty synergy IDs)
                reinforcing_ids = card.reinforcing_ids
                opposing_ids = card.opposing_ids
        else:
            # No LLM — keep existing IDs (empty for Thoth)
            reinforcing_ids = card.reinforcing_ids
            opposing_ids = card.opposing_ids

        updated_card = card.model_copy(
            update={
                "reinforcing_ids": reinforcing_ids,
                "opposing_ids": opposing_ids,
            }
        )

        is_synthesized = Provenance.SYNTHESIZED if reinforcing_ids else Provenance.DETERMINISTIC
        is_opposed = Provenance.SYNTHESIZED if opposing_ids else Provenance.DETERMINISTIC
        provenance = CardProvenance(
            card_id=card.id,
            sections={
                "reinforcing_ids": is_synthesized,
                "opposing_ids": is_opposed,
            },
        )
        results.append((updated_card, provenance))

        # Write updated Card JSON
        card_path = parsed_dir / deck_id / f"{card.id}.json"
        card_path.write_text(updated_card.model_dump_json(indent=2), encoding="utf-8")

        # Write provenance sidecar
        prov_path = provenance_dir / f"{card.id}.json"
        prov_path.write_text(provenance.model_dump_json(indent=2), encoding="utf-8")

    if failed:
        ids = ",".join(cid for cid, _ in failed)
        logger.warning(
            "%d card(s) failed synergy synthesis; re-run with: --only %s",
            len(failed),
            ids,
        )

    # Write synergy report
    report = _generate_synergy_report(results, deck_id=deck_id)
    report_path = parsed_dir / deck_id / "_synergy_report.md"
    report_path.write_text(report, encoding="utf-8")

    return results


def _generate_synergy_report(
    results: list[tuple[Card, CardProvenance]],
    *,
    deck_id: str = "book-of-thoth",
) -> str:
    """Generate a markdown synergy synthesis report.

    Args:
        results: List of ``(Card, CardProvenance)`` tuples from
            :func:`synthesize_deck_synergies`.
        deck_id: Deck identifier for the report header.

    Returns:
        A markdown string.
    """
    total = len(results)
    synthesized_reinforce = sum(1 for card, _ in results if card.reinforcing_ids)
    synthesized_oppose = sum(1 for card, _ in results if card.opposing_ids)

    lines: list[str] = [
        f"# {deck_id.replace('-', ' ').title()} Synergy Synthesis Report",
        "",
        f"**Total cards:** {total}",
        f"**Cards with reinforcing IDs:** {synthesized_reinforce}",
        f"**Cards with opposing IDs:** {synthesized_oppose}",
        "",
        "## Per-card provenance",
        "",
        "| Card | Reinforcing | Opposing |",
        "|------|-------------|----------|",
    ]

    for card, prov in results:
        rein_count = len(card.reinforcing_ids)
        opp_count = len(card.opposing_ids)
        rein_prov = prov.sections.get("reinforcing_ids", Provenance.DETERMINISTIC)
        opp_prov = prov.sections.get("opposing_ids", Provenance.DETERMINISTIC)
        rein_label = f"{rein_count} 🤖" if rein_prov == Provenance.SYNTHESIZED else f"{rein_count}"
        opp_label = f"{opp_count} 🤖" if opp_prov == Provenance.SYNTHESIZED else f"{opp_count}"
        lines.append(f"| {card.name} | {rein_label} | {opp_label} |")

    lines.extend(
        [
            "",
            "## Re-run",
            "",
            "To re-run individual cards:",
            "",
            "```bash",
            "uv run ft-normalize-thoth --only <card-id>[,<card-id>,...]",
            "```",
            "",
            "To re-run with local llama-server:",
            "",
            "```bash",
            "uv run ft-normalize-thoth --provider local",
            "```",
        ]
    )

    return "\n".join(lines)

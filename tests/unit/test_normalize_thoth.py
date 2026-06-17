"""Unit tests for :mod:`fortune_teller.developer.normalize.thoth`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from pydantic import HttpUrl

from fortune_teller.application.models.domain import (
    Arcana,
    Card,
    CardSection,
    CardSectionText,
    Deck,
)
from fortune_teller.developer.normalize.rider_waite import CardProvenance, Provenance
from fortune_teller.developer.normalize.thoth import (
    _build_deck_card_list,
    _build_sections_text,
    _parse_synergy_response,
    _validate_synergy_ids,
    synthesize_card_synergies,
    synthesize_deck_synergies,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_COMMON_URL = "https://example.com/"


def _make_card(
    card_id: str,
    name: str,
    arcana: Arcana = Arcana.MAJOR,
    sections: list[CardSectionText] | None = None,
) -> Card:
    return Card(
        id=card_id,
        name=name,
        arcana=arcana,
        sections=sections or [],
        source_url=HttpUrl(_COMMON_URL),
    )


def _make_deck(
    cards: list[Card] | None = None,
) -> Deck:
    return Deck(
        id="test-deck",
        name="Test Deck",
        cards=cards or [],
    )


def _stub_synergy_llm(response: dict[str, list[str]]) -> RunnableLambda:
    """Return a stub LLM that produces the given synergy JSON response."""
    return RunnableLambda(lambda _: AIMessage(content=json.dumps(response)))


def _write_card_json(deck_dir: Path, card: Card) -> None:
    """Write a single card JSON file into the deck directory."""
    path = deck_dir / f"{card.id}.json"
    path.write_text(card.model_dump_json(indent=2), encoding="utf-8")


def _write_meta_json(deck_dir: Path, **overrides: str) -> None:
    """Write a minimal meta.json into the deck directory."""
    meta = {"id": "book-of-thoth", "name": "Book of Thoth", **overrides}
    path = deck_dir / "meta.json"
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# _build_deck_card_list
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildDeckCardList:
    """Verify the deck card list format for the LLM prompt."""

    def test_format_id_name_per_line(self) -> None:
        deck = _make_deck(
            cards=[
                _make_card("the-fool", "The Fool"),
                _make_card("the-magician", "The Magician"),
            ]
        )
        result = _build_deck_card_list(deck)
        assert result == "the-fool — The Fool\nthe-magician — The Magician"

    def test_single_card(self) -> None:
        deck = _make_deck(cards=[_make_card("the-fool", "The Fool")])
        result = _build_deck_card_list(deck)
        assert result == "the-fool — The Fool"

    def test_empty_deck(self) -> None:
        deck = _make_deck(cards=[])
        result = _build_deck_card_list(deck)
        assert result == ""


# ---------------------------------------------------------------------------
# _build_sections_text
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildSectionsText:
    """Verify the sections text format for the LLM prompt."""

    def test_sections_rendered_as_section_value_text(self) -> None:
        card = _make_card(
            "the-fool",
            "The Fool",
            sections=[
                CardSectionText(section=CardSection.LIGHT, text="Bright chance."),
                CardSectionText(section=CardSection.SHADOW, text="Blind leap."),
            ],
        )
        result = _build_sections_text(card)
        assert result == "light: Bright chance.\nshadow: Blind leap."

    def test_empty_sections(self) -> None:
        card = _make_card("the-fool", "The Fool", sections=[])
        result = _build_sections_text(card)
        assert result == ""

    def test_single_section(self) -> None:
        card = _make_card(
            "the-fool",
            "The Fool",
            sections=[
                CardSectionText(section=CardSection.KEYWORDS, text="beginnings, freedom"),
            ],
        )
        result = _build_sections_text(card)
        assert result == "keywords: beginnings, freedom"


# ---------------------------------------------------------------------------
# _validate_synergy_ids
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestValidateSynergyIds:
    """Validate synergy ID cleaning logic."""

    def test_valid_ids_kept(self) -> None:
        deck = _make_deck(
            cards=[
                _make_card("the-fool", "The Fool"),
                _make_card("the-magician", "The Magician"),
                _make_card("the-empress", "The Empress"),
            ]
        )
        result = _validate_synergy_ids(["the-magician", "the-empress"], deck, "the-fool")
        assert result == ["the-magician", "the-empress"]

    def test_self_reference_removed(self) -> None:
        deck = _make_deck(
            cards=[
                _make_card("the-fool", "The Fool"),
                _make_card("the-magician", "The Magician"),
            ]
        )
        result = _validate_synergy_ids(["the-fool", "the-magician"], deck, "the-fool")
        assert result == ["the-magician"]

    def test_ids_not_in_deck_removed(self) -> None:
        deck = _make_deck(
            cards=[
                _make_card("the-fool", "The Fool"),
                _make_card("the-magician", "The Magician"),
            ]
        )
        result = _validate_synergy_ids(["the-magician", "nonexistent-card"], deck, "the-fool")
        assert result == ["the-magician"]

    def test_max_ids_truncation(self) -> None:
        deck = _make_deck(cards=[_make_card(f"card-{i}", f"Card {i}") for i in range(10)])
        many_ids = [f"card-{i}" for i in range(1, 10)]  # 9 valid IDs
        result = _validate_synergy_ids(many_ids, deck, "card-0", max_ids=3)
        assert len(result) == 3
        assert result == ["card-1", "card-2", "card-3"]

    def test_empty_list_stays_empty(self) -> None:
        deck = _make_deck(cards=[_make_card("the-fool", "The Fool")])
        result = _validate_synergy_ids([], deck, "the-fool")
        assert result == []

    def test_all_invalid_returns_empty(self) -> None:
        deck = _make_deck(cards=[_make_card("the-fool", "The Fool")])
        result = _validate_synergy_ids(["nope", "also-nope"], deck, "the-fool")
        assert result == []


# ---------------------------------------------------------------------------
# _parse_synergy_response
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestParseSynergyResponse:
    """Verify the synergy response parser extracts the right shape."""

    def test_parses_valid_synergy_json(self) -> None:
        content = '{"reinforcing_ids": ["a", "b"], "opposing_ids": ["c"]}'
        result = _parse_synergy_response(content)
        assert result["reinforcing_ids"] == ["a", "b"]
        assert result["opposing_ids"] == ["c"]

    def test_handles_missing_keys(self) -> None:
        content = '{"reinforcing_ids": ["a"]}'
        result = _parse_synergy_response(content)
        assert result.get("reinforcing_ids") == ["a"]
        assert result.get("opposing_ids") is None

    def test_handles_markdown_fence(self) -> None:
        content = '```json\n{"reinforcing_ids": ["x"], "opposing_ids": []}\n```'
        result = _parse_synergy_response(content)
        assert result["reinforcing_ids"] == ["x"]
        assert result["opposing_ids"] == []


# ---------------------------------------------------------------------------
# synthesize_card_synergies
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSynthesizeCardSynergies:
    """Test synergy synthesis for a single card with a stub LLM."""

    def test_valid_json_returns_correct_ids(self) -> None:
        deck = _make_deck(
            cards=[
                _make_card("the-fool", "The Fool"),
                _make_card("the-magician", "The Magician"),
                _make_card("the-empress", "The Empress"),
                _make_card("the-tower", "The Tower"),
            ]
        )
        card = _make_card("the-fool", "The Fool")
        stub_llm = _stub_synergy_llm(
            {
                "reinforcing_ids": ["the-magician", "the-empress"],
                "opposing_ids": ["the-tower"],
            }
        )
        reinforcing, opposing = synthesize_card_synergies(card, deck, stub_llm)
        assert reinforcing == ["the-magician", "the-empress"]
        assert opposing == ["the-tower"]

    def test_ids_not_in_deck_filtered_out(self) -> None:
        deck = _make_deck(
            cards=[
                _make_card("the-fool", "The Fool"),
                _make_card("the-magician", "The Magician"),
            ]
        )
        card = _make_card("the-fool", "The Fool")
        stub_llm = _stub_synergy_llm(
            {
                "reinforcing_ids": ["the-magician", "nonexistent"],
                "opposing_ids": [],
            }
        )
        reinforcing, opposing = synthesize_card_synergies(card, deck, stub_llm)
        assert reinforcing == ["the-magician"]
        assert opposing == []

    def test_self_reference_removed(self) -> None:
        deck = _make_deck(
            cards=[
                _make_card("the-fool", "The Fool"),
                _make_card("the-magician", "The Magician"),
            ]
        )
        card = _make_card("the-fool", "The Fool")
        stub_llm = _stub_synergy_llm(
            {
                "reinforcing_ids": ["the-fool", "the-magician"],
                "opposing_ids": [],
            }
        )
        reinforcing, opposing = synthesize_card_synergies(card, deck, stub_llm)
        assert reinforcing == ["the-magician"]
        assert opposing == []

    def test_too_many_ids_truncated_to_max(self) -> None:
        deck = _make_deck(cards=[_make_card(f"card-{i}", f"Card {i}") for i in range(10)])
        card = _make_card("card-0", "Card 0")
        many_ids = [f"card-{i}" for i in range(1, 9)]  # 8 valid IDs
        stub_llm = _stub_synergy_llm(
            {
                "reinforcing_ids": many_ids,
                "opposing_ids": [],
            }
        )
        reinforcing, _ = synthesize_card_synergies(card, deck, stub_llm)
        assert len(reinforcing) == 5  # default _MAX_SYNERGY_IDS
        assert reinforcing == [f"card-{i}" for i in range(1, 6)]

    def test_empty_lists_returned(self) -> None:
        deck = _make_deck(
            cards=[
                _make_card("the-fool", "The Fool"),
                _make_card("the-magician", "The Magician"),
            ]
        )
        card = _make_card("the-fool", "The Fool")
        stub_llm = _stub_synergy_llm({"reinforcing_ids": [], "opposing_ids": []})
        reinforcing, opposing = synthesize_card_synergies(card, deck, stub_llm)
        assert reinforcing == []
        assert opposing == []


# ---------------------------------------------------------------------------
# synthesize_deck_synergies
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSynthesizeDeckSynergies:
    """Test batch deck-level synergy synthesis."""

    def _populate_deck_dir(self, parsed_dir: Path) -> list[Card]:
        """Create a minimal deck directory with meta.json and card files.

        Returns the list of cards written.
        """
        deck_dir = parsed_dir / "book-of-thoth"
        deck_dir.mkdir(parents=True, exist_ok=True)
        _write_meta_json(deck_dir)

        cards = [
            _make_card("the-fool", "The Fool"),
            _make_card("the-magician", "The Magician"),
            _make_card("the-empress", "The Empress"),
            _make_card("the-tower", "The Tower"),
        ]
        for card in cards:
            _write_card_json(deck_dir, card)
        return cards

    def test_only_filter_processes_selected_cards(self, tmp_path: Path) -> None:
        self._populate_deck_dir(tmp_path / "parsed")
        stub_llm = _stub_synergy_llm(
            {
                "reinforcing_ids": ["the-magician"],
                "opposing_ids": ["the-tower"],
            }
        )
        results = synthesize_deck_synergies(
            tmp_path / "parsed",
            llm=stub_llm,
            only={"the-fool"},
        )
        assert len(results) == 1
        card, _ = results[0]
        assert card.id == "the-fool"
        assert card.reinforcing_ids == ["the-magician"]
        assert card.opposing_ids == ["the-tower"]

    def test_no_llm_keeps_existing_ids(self, tmp_path: Path) -> None:
        self._populate_deck_dir(tmp_path / "parsed")
        results = synthesize_deck_synergies(
            tmp_path / "parsed",
            llm=None,
            only={"the-fool"},
        )
        assert len(results) == 1
        card, _ = results[0]
        assert card.id == "the-fool"
        assert card.reinforcing_ids == []
        assert card.opposing_ids == []

    def test_writes_updated_card_json_to_disk(self, tmp_path: Path) -> None:
        self._populate_deck_dir(tmp_path / "parsed")
        stub_llm = _stub_synergy_llm(
            {
                "reinforcing_ids": ["the-magician"],
                "opposing_ids": ["the-tower"],
            }
        )
        synthesize_deck_synergies(
            tmp_path / "parsed",
            llm=stub_llm,
            only={"the-fool"},
        )

        card_path = tmp_path / "parsed" / "book-of-thoth" / "the-fool.json"
        assert card_path.is_file()
        written_card = Card.model_validate_json(card_path.read_text(encoding="utf-8"))
        assert written_card.id == "the-fool"
        assert written_card.reinforcing_ids == ["the-magician"]
        assert written_card.opposing_ids == ["the-tower"]

    def test_writes_provenance_sidecar_to_disk(self, tmp_path: Path) -> None:
        self._populate_deck_dir(tmp_path / "parsed")
        stub_llm = _stub_synergy_llm(
            {
                "reinforcing_ids": ["the-magician"],
                "opposing_ids": ["the-tower"],
            }
        )
        synthesize_deck_synergies(
            tmp_path / "parsed",
            llm=stub_llm,
            only={"the-fool"},
        )

        prov_path = tmp_path / "parsed" / "book-of-thoth" / ".norm" / "the-fool.json"
        assert prov_path.is_file()
        prov_data = json.loads(prov_path.read_text(encoding="utf-8"))
        assert prov_data["card_id"] == "the-fool"
        assert "sections" in prov_data

    def test_provenance_tracks_synthesized(self, tmp_path: Path) -> None:
        self._populate_deck_dir(tmp_path / "parsed")
        stub_llm = _stub_synergy_llm(
            {
                "reinforcing_ids": ["the-magician"],
                "opposing_ids": [],
            }
        )
        results = synthesize_deck_synergies(
            tmp_path / "parsed",
            llm=stub_llm,
            only={"the-fool"},
        )
        _, prov = results[0]
        assert prov.sections["reinforcing_ids"] == Provenance.SYNTHESIZED
        assert prov.sections["opposing_ids"] == Provenance.DETERMINISTIC

    def test_llm_none_provenance_is_deterministic(self, tmp_path: Path) -> None:
        self._populate_deck_dir(tmp_path / "parsed")
        results = synthesize_deck_synergies(
            tmp_path / "parsed",
            llm=None,
            only={"the-fool"},
        )
        _, prov = results[0]
        assert prov.sections["reinforcing_ids"] == Provenance.DETERMINISTIC
        assert prov.sections["opposing_ids"] == Provenance.DETERMINISTIC

    def test_result_shape(self, tmp_path: Path) -> None:
        self._populate_deck_dir(tmp_path / "parsed")
        stub_llm = _stub_synergy_llm(
            {
                "reinforcing_ids": ["the-magician"],
                "opposing_ids": ["the-tower"],
            }
        )
        results = synthesize_deck_synergies(
            tmp_path / "parsed",
            llm=stub_llm,
            only={"the-fool"},
        )
        assert len(results) == 1
        card, prov = results[0]
        assert isinstance(card, Card)
        assert isinstance(prov, CardProvenance)

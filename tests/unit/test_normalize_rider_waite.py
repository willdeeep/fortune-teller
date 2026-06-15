"""Unit tests for :mod:`fortune_teller.developer.normalize.rider_waite`."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage
from pydantic import HttpUrl

from fortune_teller.application.models.domain import (
    Arcana,
    Card,
    CardSection,
    CardSectionText,
    Suit,
)
from fortune_teller.developer.normalize.rider_waite import (
    CardProvenance,
    Provenance,
    _parse_json_object,
    _rebucket,
    build_normalize_model,
    generate_report,
    normalize_card,
    normalize_deck,
)
from fortune_teller.developer.parse.learntarot import RawCard

# ---------------------------------------------------------------------------
# Stub LLM
# ---------------------------------------------------------------------------


class _StubLLM:
    """Duck-typed LangChain runnable that returns a fixed JSON string."""

    def __init__(self, response: str) -> None:
        self._response = response

    def invoke(self, messages: object) -> AIMessage:  # noqa: ARG002
        return AIMessage(content=self._response)


@pytest.mark.unit
class TestParseJsonObject:
    """The extractor must survive the ways models wrap/garnish JSON."""

    def test_bare_json(self) -> None:
        assert _parse_json_object('{"light": "x"}') == {"light": "x"}

    def test_fenced_json_with_lang(self) -> None:
        assert _parse_json_object('```json\n{"light": "x"}\n```') == {"light": "x"}

    def test_fenced_json_without_lang(self) -> None:
        assert _parse_json_object('```\n{"light": "x"}\n```') == {"light": "x"}

    def test_preamble_then_json(self) -> None:
        assert _parse_json_object('Here is the JSON:\n{"light": "x"}') == {"light": "x"}

    def test_unparseable_raises(self) -> None:
        with pytest.raises(ValueError, match="did not return valid JSON"):
            _parse_json_object("sorry, I can't do that")

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="did not return valid JSON"):
            _parse_json_object("")

    def test_non_object_json_raises(self) -> None:
        with pytest.raises(ValueError, match="Expected a JSON object"):
            _parse_json_object("[1, 2, 3]")


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_raw_card(
    card_id: str = "the-fool",
    name: str = "The Fool",
    arcana: Arcana = Arcana.MAJOR,
    suit: Suit | None = None,
    number: int | None = 0,
    keywords: list[str] | None = None,
    actions: list[str] | None = None,
    description: str = "A young man stands at the edge of a cliff.",
    image_url: str | None = None,
) -> RawCard:
    return RawCard(
        id=card_id,
        name=name,
        arcana=arcana,
        suit=suit,
        number=number,
        keywords=keywords or ["beginnings", "freedom"],
        actions=actions or ["Take a leap", "Trust the process"],
        opposing_names=[],
        reinforcing_names=[],
        description=description,
        image_url=image_url,
        source_url=f"https://www.learntarot.com/{card_id}.htm",
    )


def _make_raw_minor_card(
    card_id: str = "ace-of-wands",
    name: str = "Ace of Wands",
    suit: Suit = Suit.WANDS,
    number: int = 1,
    image_url: str | None = None,
) -> RawCard:
    return RawCard(
        id=card_id,
        name=name,
        arcana=Arcana.MINOR,
        suit=suit,
        number=number,
        keywords=["creation", "inspiration"],
        actions=["Create", "Inspire"],
        opposing_names=[],
        reinforcing_names=[],
        description="A hand holds a flowering staff.",
        image_url=image_url,
        source_url=f"https://www.learntarot.com/{card_id}.htm",
    )


# ---------------------------------------------------------------------------
# Stage 0: deterministic (no LLM)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDeterministicStage:
    def test_keywords_section_has_deterministic_provenance(self) -> None:
        raw = _make_raw_card()
        card, prov = normalize_card(raw, llm=None)
        assert prov.sections["keywords"] == Provenance.DETERMINISTIC
        assert card.section_text(CardSection.KEYWORDS) == "beginnings, freedom"

    def test_overall_section_has_deterministic_provenance(self) -> None:
        raw = _make_raw_card()
        card, prov = normalize_card(raw, llm=None)
        assert prov.sections["overall"] == Provenance.DETERMINISTIC
        assert "edge of a cliff" in card.section_text(CardSection.OVERALL)

    def test_other_sections_are_absent(self) -> None:
        raw = _make_raw_card()
        _, prov = normalize_card(raw, llm=None)
        # Only keywords and overall should be present
        assert prov.sections.keys() == {"keywords", "overall"}

    def test_card_identity_fields_match_raw_card(self) -> None:
        raw = _make_raw_card()
        card, _ = normalize_card(raw, llm=None)
        assert card.id == raw.id
        assert card.name == raw.name
        assert card.arcana == raw.arcana
        assert card.suit == raw.suit
        assert card.number == raw.number

    def test_minor_card_suit_is_preserved(self) -> None:
        raw = _make_raw_minor_card()
        card, _ = normalize_card(raw, llm=None)
        assert card.suit == Suit.WANDS
        assert card.arcana == Arcana.MINOR

    def test_deterministic_stage_produces_byte_identical_results(self) -> None:
        raw = _make_raw_card()
        card1, prov1 = normalize_card(raw, llm=None)
        card2, prov2 = normalize_card(raw, llm=None)
        assert card1.model_dump_json() == card2.model_dump_json()
        assert prov1.model_dump_json() == prov2.model_dump_json()


# ---------------------------------------------------------------------------
# Image URL carry-through
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestImageUrlCarryThrough:
    """``image_url`` on ``RawCard`` must be preserved on the output ``Card``."""

    def test_image_url_carried_to_card(self) -> None:
        raw = _make_raw_card(image_url="https://example.com/big.jpg")
        card, _ = normalize_card(raw, llm=None)
        assert card.image_url == "https://example.com/big.jpg"

    def test_image_url_none_stays_none(self) -> None:
        raw = _make_raw_card()
        assert raw.image_url is None
        card, _ = normalize_card(raw, llm=None)
        assert card.image_url is None

    def test_image_url_preserved_with_llm(self) -> None:
        raw = _make_raw_card(image_url="https://example.com/art.jpg")
        llm = _StubLLM(
            json.dumps(
                {
                    "light": "Bright opportunities ahead.",
                    "shadow": "Risk of naivety and overconfidence.",
                    "advice": "Trust your instincts but stay aware.",
                    "reversed": "Fear of the unknown holds you back.",
                }
            )
        )
        card, _ = normalize_card(raw, llm=llm)
        assert card.image_url == "https://example.com/art.jpg"


# ---------------------------------------------------------------------------
# Stage 1: re-bucket (LLM)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRebucketStage:
    REBUCKET_RESPONSE = json.dumps(
        {
            "light": "Bright opportunities ahead.",
            "shadow": "Risk of naivety and overconfidence.",
            "advice": "Trust your instincts but stay aware.",
            "reversed": "Fear of the unknown holds you back.",
        }
    )

    def test_sections_have_rebucketed_provenance(self) -> None:
        raw = _make_raw_card()
        llm = _StubLLM(self.REBUCKET_RESPONSE)
        _, prov = normalize_card(raw, llm=llm)
        for section in ("light", "shadow", "advice", "reversed"):
            assert prov.sections[section] == Provenance.REBUCKETED

    def test_rebucketed_text_is_present_in_card(self) -> None:
        raw = _make_raw_card()
        llm = _StubLLM(self.REBUCKET_RESPONSE)
        card, _ = normalize_card(raw, llm=llm)
        assert card.section_text(CardSection.LIGHT) == "Bright opportunities ahead."
        assert card.section_text(CardSection.SHADOW) == "Risk of naivety and overconfidence."

    def test_keywords_and_overall_retain_deterministic_provenance(self) -> None:
        raw = _make_raw_card()
        llm = _StubLLM(self.REBUCKET_RESPONSE)
        _, prov = normalize_card(raw, llm=llm)
        assert prov.sections["keywords"] == Provenance.DETERMINISTIC
        assert prov.sections["overall"] == Provenance.DETERMINISTIC

    def test_empty_rebucketed_sections_are_skipped(self) -> None:
        response = json.dumps(
            {
                "light": "Bright side.",
                "shadow": "",
                "advice": "",
                "reversed": "",
            }
        )
        raw = _make_raw_card()
        llm = _StubLLM(response)
        _, prov = normalize_card(raw, llm=llm)
        assert prov.sections.get("light") == Provenance.REBUCKETED
        assert "shadow" not in prov.sections


# ---------------------------------------------------------------------------
# Stage 2: gap-fill (LLM)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestGapfillStage:
    REBUCKET_RESPONSE = json.dumps(
        {
            "light": "Bright opportunities.",
            "shadow": "Risk of naivety.",
            "advice": "Trust your instincts.",
            "reversed": "Holding back.",
        }
    )
    GAPFILL_RESPONSE = json.dumps(
        {
            "drive": "Inner drive to explore the unknown.",
            "question": "What am I ready to begin?",
            "proposal": "",
            "confirmation": "",
            "affirmation": "",
        }
    )

    def test_gapfilled_sections_have_synthesized_provenance(self) -> None:
        raw = _make_raw_card()
        llm = _StubLLM(self.REBUCKET_RESPONSE)
        normalize_card(raw, llm=llm)

        # Swap to gap-fill stub
        llm2 = _StubLLM(self.GAPFILL_RESPONSE)
        _, prov2 = normalize_card(raw, llm=llm2)
        if "drive" in prov2.sections:
            assert prov2.sections["drive"] == Provenance.SYNTHESIZED
        if "question" in prov2.sections:
            assert prov2.sections["question"] == Provenance.SYNTHESIZED

    def test_empty_gapfill_sections_are_absent(self) -> None:
        raw = _make_raw_card()
        llm = _StubLLM(self.GAPFILL_RESPONSE)
        _, prov = normalize_card(raw, llm=llm)
        assert "proposal" not in prov.sections
        assert "confirmation" not in prov.sections
        assert "affirmation" not in prov.sections

    def test_previously_filled_sections_retain_provenance(self) -> None:
        raw = _make_raw_card()
        rebucket_response = self.REBUCKET_RESPONSE
        gapfill_response = self.GAPFILL_RESPONSE

        # The same LLM is used for both stages — return rebucket response first,
        # then gapfill
        class _TwoStageLLM:
            def __init__(self) -> None:
                self._call_count = 0

            def invoke(self, messages: object) -> AIMessage:  # noqa: ARG002
                self._call_count += 1
                if self._call_count == 1:
                    return AIMessage(content=rebucket_response)
                return AIMessage(content=gapfill_response)

        _, prov = normalize_card(raw, llm=_TwoStageLLM())
        assert prov.sections["keywords"] == Provenance.DETERMINISTIC
        assert prov.sections["overall"] == Provenance.DETERMINISTIC
        assert prov.sections["light"] == Provenance.REBUCKETED
        assert prov.sections["shadow"] == Provenance.REBUCKETED
        assert prov.sections["drive"] == Provenance.SYNTHESIZED


# ---------------------------------------------------------------------------
# Assembly validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAssembly:
    def test_major_arcana_card_has_suit_none(self) -> None:
        raw = _make_raw_card()  # major arcana, suit=None
        card, _ = normalize_card(raw, llm=None)
        assert card.arcana == Arcana.MAJOR
        assert card.suit is None
        # Validates against domain model validator
        card.model_dump()

    def test_minor_arcana_card_validates_with_suit(self) -> None:
        raw = _make_raw_minor_card()
        card, _ = normalize_card(raw, llm=None)
        assert card.suit == Suit.WANDS
        card.model_dump()

    def test_sections_ordered_consistently(self) -> None:
        raw = _make_raw_card()
        llm = _StubLLM(
            json.dumps(
                {
                    "light": "Bright.",
                    "shadow": "Dark.",
                    "advice": "Do this.",
                    "reversed": "Blocked.",
                }
            )
        )
        card, _ = normalize_card(raw, llm=llm)
        section_names = [s.section.value for s in card.sections]
        # overall should come before keywords, light before shadow, etc.
        assert section_names.index("overall") < section_names.index("keywords")
        assert section_names.index("light") < section_names.index("shadow")


# ---------------------------------------------------------------------------
# Provenance sidecar shape
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestProvenance:
    def test_every_emitted_section_has_provenance(self) -> None:
        raw = _make_raw_card()
        llm = _StubLLM(
            json.dumps(
                {
                    "light": "Bright.",
                    "shadow": "",
                    "advice": "",
                    "reversed": "Blocked.",
                }
            )
        )
        card, prov = normalize_card(raw, llm=llm)
        for section in card.sections:
            assert section.section.value in prov.sections

    def test_provenance_sidecar_serialises(self) -> None:
        raw = _make_raw_card()
        _, prov = normalize_card(raw, llm=None)
        serialised = prov.model_dump_json()
        assert prov.card_id in serialised
        assert Provenance.DETERMINISTIC.value in serialised

    def test_provenance_is_frozen(self) -> None:
        raw = _make_raw_card()
        _, prov = normalize_card(raw, llm=None)
        assert isinstance(prov, CardProvenance)
        # Verify it's truly frozen (pydantic ConfigDict(frozen=True))
        with pytest.raises((TypeError, ValueError)):
            prov.card_id = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestReport:
    def test_generate_report_contains_summary_header(self) -> None:
        raw = _make_raw_card()
        card, prov = normalize_card(raw, llm=None)
        report = generate_report([(card, prov)])
        assert "# Rider-Waite Normalisation Report" in report
        assert "Total cards:" in report

    def test_generate_report_contains_per_card_provenance(self) -> None:
        raw = _make_raw_card()
        card, prov = normalize_card(raw, llm=None)
        report = generate_report([(card, prov)])
        assert card.name in report

    def test_generate_report_shows_section_name(self) -> None:
        raw = _make_raw_card()
        llm = _StubLLM(
            json.dumps(
                {
                    "light": "Bright.",
                    "shadow": "Dark.",
                    "advice": "Do this.",
                    "reversed": "Blocked.",
                }
            )
        )
        card, prov = normalize_card(raw, llm=llm)
        report = generate_report([(card, prov)])
        assert "light" in report
        assert "keywords" in report
        assert "overall" in report

    def test_generate_report_contains_synth_emoji(self) -> None:
        # Create a card with a synthesized section
        raw = _make_raw_card()
        card = Card(
            id=raw.id,
            name=raw.name,
            arcana=raw.arcana,
            suit=raw.suit,
            number=raw.number,
            sections=[
                CardSectionText(section=CardSection.LIGHT, text="Bright."),
                CardSectionText(
                    section=CardSection.DRIVE,
                    text="Inner drive.",  # synthesised
                ),
            ],
            source_url=HttpUrl(raw.source_url),
        )
        prov = CardProvenance(
            card_id=raw.id,
            sections={
                "light": Provenance.REBUCKETED,
                "drive": Provenance.SYNTHESIZED,
            },
        )
        report = generate_report([(card, prov)])
        assert "🤖" in report

    def test_generate_report_contains_empty_flag(self) -> None:
        raw = _make_raw_card()
        card, prov = normalize_card(raw, llm=None)
        report = generate_report([(card, prov)])
        # Keywords and overall are populated, but other sections should be empty
        assert "⚠️" in report


# ---------------------------------------------------------------------------
# Provider factory
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildNormalizeModel:
    def test_local_provider_builds_without_anthropic(self) -> None:
        """``provider='local'`` should use ChatOpenAI (no anthropic import)."""
        with patch("fortune_teller.application.chains.per_card.build_chat_model") as mock_build:
            result = build_normalize_model(provider="local")
            mock_build.assert_called_once()
            assert result == mock_build.return_value

    def test_api_provider_constructs_chat_anthropic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``provider='api'`` should construct ChatAnthropic with the configured key.

        Pin the key to a fixed value so the test is independent of the
        developer's ``.env`` (and never echoes a real secret on failure).
        """
        monkeypatch.setattr(
            "fortune_teller.developer.normalize.rider_waite.settings.anthropic_api_key",
            "test-key",
        )
        with patch("langchain_anthropic.ChatAnthropic") as mock_chat:
            result = build_normalize_model(provider="api", model="claude-sonnet-4-6")
            mock_chat.assert_called_once_with(
                model="claude-sonnet-4-6",
                temperature=0.0,
                anthropic_api_key="test-key",
                timeout=180,
            )
            assert result == mock_chat.return_value


# ---------------------------------------------------------------------------
# normalize_deck with tmp_path
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNormalizeDeck:
    def test_normalize_deck_writes_output_files(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "parsed"
        raw_dir.mkdir(parents=True)

        raw = _make_raw_card()
        raw_path = raw_dir / f"{raw.id}.json"
        raw_path.write_text(raw.model_dump_json(), encoding="utf-8")

        results = normalize_deck(raw_dir, out_dir, llm=None)
        assert len(results) == 1

        # Check output files
        card_path = out_dir / f"{raw.id}.json"
        assert card_path.is_file()

        prov_path = out_dir / ".norm" / f"{raw.id}.json"
        assert prov_path.is_file()

        meta_path = out_dir / "meta.json"
        assert meta_path.is_file()

        report_path = out_dir / "_normalization_report.md"
        assert report_path.is_file()

    def test_normalize_deck_only_filters(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "parsed"
        raw_dir.mkdir(parents=True)

        raw1 = _make_raw_card(card_id="the-fool", name="The Fool")
        raw2 = _make_raw_card(card_id="the-magician", name="The Magician")

        raw_dir.joinpath("the-fool.json").write_text(raw1.model_dump_json(), encoding="utf-8")
        raw_dir.joinpath("the-magician.json").write_text(raw2.model_dump_json(), encoding="utf-8")

        results = normalize_deck(raw_dir, out_dir, llm=None, only={"the-fool"})
        assert len(results) == 1
        assert results[0][0].id == "the-fool"

    def test_normalize_deck_parses_raw_card_correctly(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "parsed"
        raw_dir.mkdir(parents=True)

        raw = _make_raw_card()
        raw_path = raw_dir / f"{raw.id}.json"
        raw_path.write_text(raw.model_dump_json(), encoding="utf-8")

        results = normalize_deck(raw_dir, out_dir, llm=None)
        card, prov = results[0]
        assert isinstance(card, Card)
        assert isinstance(prov, CardProvenance)

    def test_normalize_deck_provenance_sidecar_shape(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "raw"
        out_dir = tmp_path / "parsed"
        raw_dir.mkdir(parents=True)

        raw = _make_raw_card()
        raw_path = raw_dir / f"{raw.id}.json"
        raw_path.write_text(raw.model_dump_json(), encoding="utf-8")

        normalize_deck(raw_dir, out_dir, llm=None)
        prov_path = out_dir / ".norm" / f"{raw.id}.json"
        prov_data = json.loads(prov_path.read_text(encoding="utf-8"))
        assert prov_data["card_id"] == raw.id
        assert "sections" in prov_data


# ---------------------------------------------------------------------------
# _rebucket helper detail test
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRebucketHelper:
    def test_rebucket_returns_parsed_json(self) -> None:
        raw = _make_raw_card()
        llm = _StubLLM(
            json.dumps(
                {
                    "light": "Be spontaneous.",
                    "shadow": "Watch your step.",
                    "advice": "Trust.",
                    "reversed": "Hold back.",
                }
            )
        )
        result = _rebucket(raw, llm)
        assert isinstance(result, dict)
        assert result["light"] == "Be spontaneous."
        assert result["shadow"] == "Watch your step."

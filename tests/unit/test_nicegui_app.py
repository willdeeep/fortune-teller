"""Unit tests for the NiceGUI UI module.

Tests the framework-agnostic formatting functions and the
NiceGUI-specific wiring (build_app, detail dialog, history section).
NiceGUI's ``User`` fixture drives the browser-like interaction tests.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import ClassVar
from uuid import UUID

import pytest
from pydantic import HttpUrl

import fortune_teller.application.ui.nicegui_app as nicegui_app_module
from fortune_teller.application.models.domain import (
    Arcana,
    Card,
    CardInterpretation,
    CardSection,
    CardSectionText,
    Chunk,
    ChunkType,
    DealtCard,
    Deck,
    Orientation,
    Reading,
    ReadingListItem,
    Spread,
    SpreadPosition,
    Suit,
)
from fortune_teller.application.services.reading import ReadingHandle, ReadingService
from fortune_teller.application.stores.vector import VectorStore
from fortune_teller.application.ui.nicegui_app import (
    _format_card_detail,
    _format_card_text,
    _format_position_info,
    _format_reading_detail,
    _history_rows,
    build_app,
    run_reading_generator,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


_SOURCE = "https://example.test/card"
_SPREAD_URL = "https://example.test/spread"


def _card(cid: str) -> Card:
    return Card(
        id=cid,
        name=cid.replace("-", " ").title(),
        arcana=Arcana.MAJOR,
        source_url=HttpUrl(_SOURCE),
    )


def _make_deck(size: int = 5) -> Deck:
    return Deck(
        id="test-deck",
        name="Test Deck",
        cards=[_card(f"card-{i:02d}") for i in range(size)],
    )


def _make_spread(position_count: int = 3) -> Spread:
    return Spread(
        id="test-spread",
        name="Test Spread",
        positions=[
            SpreadPosition(
                index=i,
                name=f"Position {i}",
                meaning=f"Meaning of position {i}.",
                source_url=HttpUrl(_SPREAD_URL),
            )
            for i in range(position_count)
        ],
    )


class _StubChain:
    """Chain that returns a canned string; records inputs."""

    inputs_received: ClassVar[list[dict[str, str]] | None] = None

    def __init__(self, *, reply: str = "Stub.") -> None:
        self.reply = reply
        self.invocations = 0
        if _StubChain.inputs_received is None:
            _StubChain.inputs_received = []

    def invoke(self, inputs: dict[str, str]) -> str:
        self.invocations += 1
        _StubChain.inputs_received.append(dict(inputs))
        return f"{self.reply}#{self.invocations}"


class _StubReadingService:
    """In-process double for :class:`ReadingService`."""

    def __init__(self, deck: Deck, spread: Spread) -> None:
        self._deck = deck
        self._spread = spread
        self._per_card_chain = _StubChain(reply="CARD")
        self._summary_chain = _StubChain(reply="SUMMARY")
        self._vector_store = None
        self._embedder = None
        self._counter = 0

    def start(self, seed: int | None = None) -> ReadingHandle:  # noqa: ARG002
        from fortune_teller.application.services.deck import (  # noqa: PLC0415
            DeckSession,
        )

        session = DeckSession(self._deck, rng=__import__("random").Random(0))
        return ReadingHandle(
            deck_session=session,
            deck_id=self._deck.id,
            spread=self._spread,
        )

    def deal_next(self, handle: ReadingHandle) -> CardInterpretation:
        position_index = len(handle.dealt)
        if position_index >= len(self._spread.positions):
            raise RuntimeError("All positions filled.")
        dealt = handle.deck_session.deal_one(position_index)
        position = self._spread.position_by_index(position_index)
        card = self._deck.card_by_id(dealt.card_id)
        self._counter += 1
        text = self._per_card_chain.invoke(
            {
                "card_name": card.name,
                "orientation": dealt.orientation.value,
                "position_name": position.name,
                "position_meaning": position.meaning,
            }
        )
        interp = CardInterpretation(
            dealt=dealt,
            card_name=card.name,
            position_name=position.name,
            text=text,
        )
        handle.dealt.append(dealt)
        handle.interpretations.append(interp)
        return interp

    def finalize(self, handle: ReadingHandle) -> Reading:
        summary = self._summary_chain.invoke(
            {"spread_name": self._spread.name, "card_summaries": "x"}
        )
        return Reading(
            deck_id=handle.deck_id,
            spread_id=self._spread.id,
            dealt=list(handle.dealt),
            per_card=list(handle.interpretations),
            summary=summary,
        )


class _StubHistoryStore:
    """In-memory HistoryStore double for UI tests."""

    def __init__(self) -> None:
        self.saved: list[Reading] = []

    def save(self, reading: Reading) -> None:
        self.saved.append(reading)

    def list_recent(self, limit: int = 50) -> list[ReadingListItem]:
        return [
            ReadingListItem(
                id=r.id,
                deck_id=r.deck_id,
                spread_id=r.spread_id,
                card_names=[i.card_name for i in r.per_card],
                summary_preview=r.summary[:120] if len(r.summary) > 120 else r.summary,
                created_at=r.created_at,
            )
            for r in reversed(self.saved[-limit:])
        ]

    def get(self, reading_id: UUID) -> Reading | None:
        for r in self.saved:
            if r.id == reading_id:
                return r
        return None


@pytest.fixture
def stub_service() -> _StubReadingService:
    _StubChain.inputs_received = []
    return _StubReadingService(deck=_make_deck(5), spread=_make_spread(3))


# ---------------------------------------------------------------------------
# _format_card_text (framework-agnostic — unchanged from Gradio tests)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFormatCardText:
    def test_upright_uses_up_arrow(self) -> None:
        out = _format_card_text("The Fool", "upright", "Beginnings.")
        assert "The Fool" in out
        assert "▲ UPRIGHT" in out
        assert "Beginnings." in out

    def test_reversed_uses_down_arrow(self) -> None:
        out = _format_card_text("The Fool", "reversed", "Holding back.")
        assert "The Fool" in out
        assert "▼ REVERSED" in out
        assert "Holding back." in out

    def test_three_lines_then_body(self) -> None:
        out = _format_card_text("The Fool", "upright", "Body text.")
        lines = out.split("\n")
        assert lines[0] == "The Fool"
        assert lines[1] == "▲ UPRIGHT"
        assert lines[2] == ""
        assert lines[3] == "Body text."

    def test_position_meaning_included_when_provided(self) -> None:
        out = _format_card_text("The Fool", "upright", "text", "Past", "What was set in motion")
        assert "*Past: What was set in motion*" in out

    def test_position_meaning_omitted_when_none(self) -> None:
        out = _format_card_text("The Fool", "upright", "text")
        assert "*" not in out


# ---------------------------------------------------------------------------
# _format_card_detail (framework-agnostic)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFormatCardDetail:
    def test_renders_card_name_and_arcana(self) -> None:
        card = Card(
            id="0-the-fool",
            name="The Fool",
            arcana=Arcana.MAJOR,
            source_url=HttpUrl("https://example.test/the-fool"),
        )
        result = _format_card_detail(card)
        assert "## The Fool" in result
        assert "*Major Arcana*" in result

    def test_renders_minor_arcana_with_suit(self) -> None:
        card = Card(
            id="ace-of-wands",
            name="Ace of Wands",
            arcana=Arcana.MINOR,
            suit=Suit.WANDS,
            source_url=HttpUrl("https://example.test/ace-of-wands"),
        )
        result = _format_card_detail(card)
        assert "## Ace of Wands" in result
        assert "*Wands*" in result

    def test_renders_sections(self) -> None:
        card = Card(
            id="0-the-fool",
            name="The Fool",
            arcana=Arcana.MAJOR,
            sections=[
                CardSectionText(section=CardSection.DRIVE, text="Pure potential."),
                CardSectionText(section=CardSection.LIGHT, text="Spontaneity."),
            ],
            source_url=HttpUrl("https://example.test/the-fool"),
        )
        result = _format_card_detail(card)
        assert "**Drive:** Pure potential." in result
        assert "**Light:** Spontaneity." in result

    def test_shows_fallback_when_no_sections(self) -> None:
        card = Card(
            id="0-the-fool",
            name="The Fool",
            arcana=Arcana.MAJOR,
            source_url=HttpUrl("https://example.test/the-fool"),
        )
        result = _format_card_detail(card)
        assert "No structured data" in result

    def test_includes_source_link(self) -> None:
        card = Card(
            id="0-the-fool",
            name="The Fool",
            arcana=Arcana.MAJOR,
            source_url=HttpUrl("https://example.test/the-fool"),
        )
        result = _format_card_detail(card)
        assert "[View source" in result
        assert "https://example.test/the-fool" in result

    def test_includes_image_when_provided(self) -> None:
        card = Card(
            id="0-the-fool",
            name="The Fool",
            arcana=Arcana.MAJOR,
            source_url=HttpUrl("https://example.test/the-fool"),
        )
        result = _format_card_detail(card, image_path="/data/images/the-fool.jpeg")
        assert "![The Fool]" in result
        assert "/data/images/the-fool.jpeg" in result

    def test_omits_image_when_none(self) -> None:
        card = Card(
            id="0-the-fool",
            name="The Fool",
            arcana=Arcana.MAJOR,
            source_url=HttpUrl("https://example.test/the-fool"),
        )
        result = _format_card_detail(card, image_path=None)
        assert "![" not in result

    def test_includes_number_for_major_arcana(self) -> None:
        card = Card(
            id="0-the-fool",
            name="The Fool",
            arcana=Arcana.MAJOR,
            number=0,
            source_url=HttpUrl("https://example.test/the-fool"),
        )
        result = _format_card_detail(card)
        assert "·  0" in result


# ---------------------------------------------------------------------------
# _format_position_info (framework-agnostic)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFormatPositionInfo:
    def test_renders_name_and_meaning(self) -> None:
        result = _format_position_info(
            "Past", "What was set in motion.", "https://example.test/spread"
        )
        assert "**Past:** What was set in motion." in result

    def test_includes_source_link(self) -> None:
        result = _format_position_info("Present", "Current energy.", "https://example.test/spread")
        assert "[Source ↗](https://example.test/spread)" in result


# ---------------------------------------------------------------------------
# run_reading_generator (framework-agnostic)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRunReadingGenerator:
    def test_yields_one_snapshot_per_card_plus_one_for_summary(
        self,
        stub_service: _StubReadingService,
    ) -> None:
        snapshots = list(run_reading_generator(stub_service))
        assert len(snapshots) == 4

    def test_each_snapshot_has_one_more_panel_filled(
        self,
        stub_service: _StubReadingService,
    ) -> None:
        snapshots = list(run_reading_generator(stub_service))
        n = 3
        s0 = snapshots[0]
        assert s0[0] is None and s0[1] is None and s0[2] is None
        assert s0[n] != ""
        assert s0[n + 1] == "" and s0[n + 2] == ""
        assert s0[-1] == ""
        s1 = snapshots[1]
        assert s1[n] != "" and s1[n + 1] != ""
        assert s1[n + 2] == ""
        s2 = snapshots[2]
        assert s2[n] != "" and s2[n + 1] != "" and s2[n + 2] != ""
        assert s2[-1] == ""
        s3 = snapshots[3]
        assert s3[n] != "" and s3[n + 1] != "" and s3[n + 2] != ""
        assert s3[-1] != ""

    def test_panels_carry_card_name_and_orientation(
        self,
        stub_service: _StubReadingService,
    ) -> None:
        snapshots = list(run_reading_generator(stub_service))
        n = 3
        final = snapshots[-1]
        panel_start = n
        for i in range(n):
            panel = final[panel_start + i]
            assert "▲ UPRIGHT" in panel or "▼ REVERSED" in panel

    def test_summary_includes_text_from_summary_chain(
        self,
        stub_service: _StubReadingService,
    ) -> None:
        snapshots = list(run_reading_generator(stub_service))
        assert "SUMMARY#1" in snapshots[-1][-1]

    def test_each_deal_uses_a_different_card(
        self,
        stub_service: _StubReadingService,
    ) -> None:
        snapshots = list(run_reading_generator(stub_service))
        n = 3
        panel_start = n
        assert stub_service._per_card_chain.invocations == 3
        newest_panel_per_snapshot = [
            s[panel_start + len([p for p in s[panel_start : panel_start + n] if p]) - 1]
            for s in snapshots[:n]
        ]
        card_names = [p.split("\n")[0] for p in newest_panel_per_snapshot]
        assert len(set(card_names)) == 3

    def test_image_slot_paths_resolved_from_images_dir(
        self,
        stub_service: _StubReadingService,
        tmp_path: Path,
    ) -> None:
        deck_dir = tmp_path / "test-deck"
        deck_dir.mkdir()
        (deck_dir / "card-00.png").write_bytes(b"fake")
        (deck_dir / "card-01.png").write_bytes(b"fake")
        (deck_dir / "card-02.png").write_bytes(b"fake")

        snapshots = list(run_reading_generator(stub_service, images_dir=deck_dir))
        for snapshot in snapshots:
            first_image = snapshot[0]
            if first_image is not None:
                assert str(deck_dir) in first_image

    def test_image_slots_are_none_when_no_images_dir(
        self,
        stub_service: _StubReadingService,
    ) -> None:
        snapshots = list(run_reading_generator(stub_service))
        for snapshot in snapshots:
            assert snapshot[0] is None
            assert snapshot[1] is None
            assert snapshot[2] is None


# ---------------------------------------------------------------------------
# ReadingService.deck_id (framework-agnostic)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestReadingServiceDeckId:
    def test_deck_id_exposed_as_property(self) -> None:
        deck = _make_deck(3)
        spread = _make_spread(3)
        svc = ReadingService(
            deck=deck,
            spread=spread,
            per_card_chain=_StubChain(),
            summary_chain=_StubChain(),
        )
        assert svc.deck_id == "test-deck"


# ---------------------------------------------------------------------------
# RAG wiring (framework-agnostic)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRagWiringInService:
    def test_vector_store_and_embedder_produce_extended_context(self) -> None:
        _StubChain.inputs_received = []
        card = _card("0-the-fool")
        card_with_sections = card.model_copy(
            update={
                "sections": [
                    CardSectionText(
                        section=CardSection.OVERALL,
                        text="Pure potential.",
                    ),
                ],
            },
        )
        deck = Deck(id="test-deck", name="Test Deck", cards=[card_with_sections])
        spread = _make_spread(1)
        chain = _StubChain(reply="INTERP")

        with VectorStore(":memory:", dimension=4) as store:
            chunk = Chunk(
                chunk_type=ChunkType.CARD_SECTION,
                deck_id="test-deck",
                card_id="0-the-fool",
                card_name="The Fool",
                section=CardSection.OVERALL,
                source_url="https://example.test/the-fool",
                text="Pure potential, unformed.",
                embedding=[0.0, 0.0, 0.0, 0.0],
            )
            store.add_chunks([chunk])

            class _StubEmbedder:
                def embed_query(self, _text: str) -> list[float]:
                    return [0.0, 0.0, 0.0, 0.0]

            svc = ReadingService(
                deck=deck,
                spread=spread,
                per_card_chain=chain,
                summary_chain=_StubChain(),
                vector_store=store,
                embedder=_StubEmbedder(),
            )
            handle = svc.start(seed=0)
            interp = svc.deal_next(handle)

        assert _StubChain.inputs_received is not None
        ctx = _StubChain.inputs_received[0]
        assert "retrieved_card_sections" in ctx
        assert "Pure potential" in ctx["retrieved_card_sections"]
        assert "retrieved_position_text" in ctx
        assert "card_name" in ctx
        assert "position_name" in ctx
        assert "position_meaning" in ctx
        assert "INTERP" in interp.text


# ---------------------------------------------------------------------------
# History helpers (framework-agnostic)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFormatReadingDetail:
    def test_format_reading_detail_finds_reading(self) -> None:
        history = _StubHistoryStore()
        reading = Reading(
            deck_id="test-deck",
            spread_id="test-spread",
            dealt=[
                DealtCard(card_id="the-fool", orientation=Orientation.UPRIGHT, position_index=0)
            ],
            per_card=[
                CardInterpretation(
                    dealt=DealtCard(
                        card_id="the-fool", orientation=Orientation.UPRIGHT, position_index=0
                    ),
                    card_name="The Fool",
                    position_name="Past",
                    text="New beginnings.",
                ),
            ],
            summary="A short summary.",
        )
        history.save(reading)
        result = _format_reading_detail(str(reading.id), history)
        assert "The Fool" in result
        assert "A short summary." in result

    def test_format_reading_detail_returns_empty_for_missing(self) -> None:
        history = _StubHistoryStore()
        result = _format_reading_detail(str(uuid.uuid4()), history)
        assert result == ""


# ---------------------------------------------------------------------------
# _history_rows (NiceGUI-specific helper)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHistoryRows:
    def test_converts_load_history_list_rows_to_dicts(self) -> None:
        history = _StubHistoryStore()
        reading = Reading(
            deck_id="test-deck",
            spread_id="test-spread",
            dealt=[
                DealtCard(card_id="the-fool", orientation=Orientation.UPRIGHT, position_index=0)
            ],
            per_card=[
                CardInterpretation(
                    dealt=DealtCard(
                        card_id="the-fool",
                        orientation=Orientation.UPRIGHT,
                        position_index=0,
                    ),
                    card_name="The Fool",
                    position_name="Past",
                    text="New beginnings.",
                ),
            ],
            summary="A short summary.",
        )
        history.save(reading)
        rows = _history_rows(history)
        assert len(rows) == 1
        assert rows[0]["id"] == str(reading.id)
        assert rows[0]["spread"] == "test-spread"
        assert "The Fool" in rows[0]["cards"]


# ---------------------------------------------------------------------------
# build_app — NiceGUI wiring
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildApp:
    def test_build_app_sets_module_state(self, stub_service: _StubReadingService) -> None:
        build_app(stub_service)
        assert nicegui_app_module._service is stub_service
        assert nicegui_app_module._cards_by_id is not None
        assert len(nicegui_app_module._cards_by_id) == 5

    def test_build_app_with_history_sets_module_state(
        self, stub_service: _StubReadingService
    ) -> None:
        history = _StubHistoryStore()
        build_app(stub_service, history_store=history)
        assert nicegui_app_module._history_store is history

    def test_build_app_with_images_dir_registers_static_mount(
        self,
        stub_service: _StubReadingService,
        tmp_path: Path,
    ) -> None:
        deck_dir = tmp_path / "test-deck"
        deck_dir.mkdir()
        build_app(stub_service, images_dir=deck_dir)
        assert nicegui_app_module._images_dir == deck_dir

    def test_build_app_without_images_dir(self, stub_service: _StubReadingService) -> None:
        build_app(stub_service, images_dir=None)
        assert nicegui_app_module._images_dir is None

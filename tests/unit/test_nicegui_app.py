"""Unit tests for the NiceGUI UI module.

Covers three layers:

- the framework-agnostic formatters;
- the ``build_app`` dependency wiring (module-state + static-mount); and
- the page/interaction layer (``reading_page``, ``_run_reading``,
  ``_show_detail``, the card-detail ``ui.dialog``, and the history section),
  driven headlessly by the ``nicegui.testing`` ``User`` fixture against the
  stub app in ``tests/nicegui_main.py``.

The ``main()`` entry point is covered separately with its heavy dependencies
mocked.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import ClassVar
from uuid import UUID

import pytest
from nicegui import ui
from nicegui.element_filter import ElementFilter
from nicegui.testing import User
from pydantic import HttpUrl

import fortune_teller.application.services.reading as reading_mod
import fortune_teller.application.stores.sqlite as sqlite_mod
import fortune_teller.application.ui.nicegui_app as nicegui_app_module
from fortune_teller.application.config import settings
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
    _format_list_item,
    _format_position_info,
    _format_reading_detail,
    _history_rows,
    build_app,
    rotation_style,
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

    @property
    def deck_id(self) -> str:
        return self._deck.id

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
# _format_list_item (framework-agnostic)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFormatListItem:
    def test_numbers_and_names_the_position_and_card(self) -> None:
        out = _format_list_item(1, "Past", "The Fool", "upright", "Beginnings.")
        assert out.startswith("**1. Past**")
        assert "The Fool" in out
        assert "▲ UPRIGHT" in out
        assert "Beginnings." in out

    def test_reversed_uses_down_arrow_and_label(self) -> None:
        out = _format_list_item(2, "Present", "The Tower", "reversed", "Upheaval.")
        assert "**2. Present**" in out
        assert "▼ REVERSED" in out
        assert "Upheaval." in out

    def test_text_is_separated_from_header_by_blank_line(self) -> None:
        out = _format_list_item(3, "Future", "The Star", "upright", "Hope.")
        assert "\n\nHope." in out


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

    def test_build_app_sets_current_deck_id(self, stub_service: _StubReadingService) -> None:
        build_app(stub_service)
        assert nicegui_app_module._current_deck_id == stub_service.deck_id

    def test_build_app_with_deck_options(self, stub_service: _StubReadingService) -> None:
        build_app(stub_service, deck_options=[("test-deck", "Test Deck")])
        assert nicegui_app_module._deck_options == [("test-deck", "Test Deck")]

    def test_build_app_without_deck_options(self, stub_service: _StubReadingService) -> None:
        build_app(stub_service)
        assert nicegui_app_module._deck_options == []


# ---------------------------------------------------------------------------
# NiceGUI page / interaction tests (nicegui.testing User simulation)
#
# These drive the real reading_page / _run_reading / _show_detail / history
# wiring against the in-process stub app defined in tests/nicegui_main.py.
# ---------------------------------------------------------------------------

_MAIN = "tests/nicegui_main.py"
_GRID_MAIN = "tests/nicegui_grid_main.py"


@pytest.mark.unit
@pytest.mark.nicegui_main_file(_MAIN)
class TestReadingPage:
    async def test_page_renders_title_deck_and_spread(self, user: User) -> None:
        await user.open("/")
        await user.should_see("Fortune Teller")
        await user.should_see("Test Spread")
        await user.should_see("Test Deck")

    async def test_position_titles_render(self, user: User) -> None:
        await user.open("/")
        await user.should_see("Position 0")
        await user.should_see("Position 2")

    async def test_new_reading_button_present(self, user: User) -> None:
        await user.open("/")
        await user.should_see("New Reading")

    async def test_new_reading_deals_cards_and_summary(self, user: User) -> None:
        await user.open("/")
        user.find("New Reading").click()
        # The stub deck names cards "Card 00".."Card 04"; a 3-card spread deals
        # three of them, each panel carrying an orientation arrow + summary.
        await user.should_see("UPRIGHT")
        await user.should_see("Summary")

    async def test_dealing_flips_card_backs_to_faces(self, user: User) -> None:
        await user.open("/")
        # Before dealing: every card back is visible, every face image hidden.
        with user.client:
            backs = [e for e in ElementFilter(kind=ui.element) if "ft-card-back" in e._classes]
            faces = list(ElementFilter(kind=ui.image))
        assert backs, "expected card-back elements before dealing"
        assert all("hidden" not in b._classes for b in backs)
        assert all("hidden" in f._classes for f in faces)

        user.find("New Reading").click()
        await user.should_see("Summary")

        # After dealing: every back is hidden and every face image is shown.
        with user.client:
            backs = [e for e in ElementFilter(kind=ui.element) if "ft-card-back" in e._classes]
            faces = list(ElementFilter(kind=ui.image))
        assert all("hidden" in b._classes for b in backs)
        assert all("hidden" not in f._classes for f in faces)


@pytest.mark.unit
@pytest.mark.nicegui_main_file(_MAIN)
class TestDetailDialog:
    async def test_detail_button_before_reading_shows_placeholder(self, user: User) -> None:
        await user.open("/")
        user.find("Details · Position 0").click()
        await user.should_see("No card dealt")

    async def test_detail_dialog_shows_card_detail_after_reading(self, user: User) -> None:
        await user.open("/")
        user.find("New Reading").click()
        await user.should_see("Summary")
        user.find("Details · Position 0").click()
        # Stub cards have no sections, so the detail renders the fallback plus
        # the source-attribution link.
        await user.should_see("No structured data")
        await user.should_see("View source")


@pytest.mark.unit
@pytest.mark.nicegui_main_file(_MAIN)
class TestHistorySection:
    async def test_history_section_renders_seeded_reading(self, user: User) -> None:
        await user.open("/")
        await user.should_see("History")
        # Quasar renders table cells client-side, so assert the table element is
        # present rather than its row text.
        await user.should_see(kind=ui.table)

    async def test_refresh_button_does_not_error(self, user: User) -> None:
        await user.open("/")
        user.find("Refresh").click()
        await user.should_see("History")


# ---------------------------------------------------------------------------
# main() — console-script entry point (heavy deps mocked)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMain:
    def test_main_builds_app_and_runs(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        stub_service: _StubReadingService,
    ) -> None:
        class _FakeStore:
            opened = False
            closed = False

            def __init__(self, _path: object) -> None:
                pass

            def open(self) -> None:
                _FakeStore.opened = True

            def close(self) -> None:
                _FakeStore.closed = True

        monkeypatch.setattr(sqlite_mod, "SQLiteStore", _FakeStore)
        monkeypatch.setattr(
            reading_mod,
            "build_reading_service",
            lambda *_args, **_kwargs: stub_service,
        )
        monkeypatch.setattr(settings, "images_dir", tmp_path)

        ran: dict[str, object] = {}
        monkeypatch.setattr(nicegui_app_module.ui, "run", lambda **kwargs: ran.update(kwargs))

        nicegui_app_module.main()

        assert _FakeStore.opened is True
        assert ran["port"] == 7860
        assert ran["title"] == "Fortune Teller"
        assert nicegui_app_module._service is stub_service


# ---------------------------------------------------------------------------
# _format_card_detail — reinforcing/opposing synergy references (plan 0024)
# ---------------------------------------------------------------------------


def _synergy_card() -> Card:
    """Card with reinforcing/opposing IDs for synergy-render tests."""
    return Card(
        id="the-fool",
        name="The Fool",
        arcana=Arcana.MAJOR,
        reinforcing_ids=["the-magician", "the-high-priestess"],
        opposing_ids=["the-tower"],
        source_url=HttpUrl("https://example.test/the-fool"),
    )


def _ref_card(card_id: str, name: str) -> Card:
    return Card(
        id=card_id,
        name=name,
        arcana=Arcana.MAJOR,
        source_url=HttpUrl(f"https://example.test/{card_id}"),
    )


@pytest.mark.unit
class TestFormatCardDetailSynergy:
    def test_renders_reinforcing_names_when_cards_by_id_provided(self) -> None:
        card = _synergy_card()
        ref_deck: dict[str, Card] = {
            card.id: card,
            "the-magician": _ref_card("the-magician", "The Magician"),
            "the-high-priestess": _ref_card("the-high-priestess", "The High Priestess"),
        }
        result = _format_card_detail(card, cards_by_id=ref_deck)
        assert "**Reinforcing:** The Magician, The High Priestess" in result

    def test_renders_opposing_names_when_cards_by_id_provided(self) -> None:
        card = Card(
            id="the-fool",
            name="The Fool",
            arcana=Arcana.MAJOR,
            opposing_ids=["the-tower"],
            source_url=HttpUrl("https://example.test/the-fool"),
        )
        ref_deck: dict[str, Card] = {
            card.id: card,
            "the-tower": _ref_card("the-tower", "The Tower"),
        }
        result = _format_card_detail(card, cards_by_id=ref_deck)
        assert "**Opposing:** The Tower" in result

    def test_omits_synergy_refs_when_cards_by_id_none(self) -> None:
        card = _synergy_card()
        result = _format_card_detail(card)
        assert "Reinforcing:" not in result
        assert "Opposing:" not in result

    def test_omits_synergy_refs_when_ids_do_not_resolve(self) -> None:
        card = _synergy_card()
        # cards_by_id only contains the card itself — reinforcing/opposing IDs
        # cannot resolve, so the lines should be omitted.
        result = _format_card_detail(card, cards_by_id={card.id: card})
        assert "Reinforcing:" not in result
        assert "Opposing:" not in result

    def test_omits_synergy_refs_when_card_has_none(self) -> None:
        card = _ref_card("the-fool", "The Fool")
        other = _ref_card("the-magician", "The Magician")
        result = _format_card_detail(card, cards_by_id={card.id: card, other.id: other})
        assert "Reinforcing:" not in result
        assert "Opposing:" not in result

    def test_renders_both_reinforcing_and_opposing_together(self) -> None:
        card = _synergy_card()
        ref_deck: dict[str, Card] = {
            card.id: card,
            "the-magician": _ref_card("the-magician", "The Magician"),
            "the-high-priestess": _ref_card("the-high-priestess", "The High Priestess"),
            "the-tower": _ref_card("the-tower", "The Tower"),
        }
        result = _format_card_detail(card, cards_by_id=ref_deck)
        assert "**Reinforcing:** The Magician, The High Priestess" in result
        assert "**Opposing:** The Tower" in result
        # Reinforcing block comes before the source link.
        assert result.index("**Reinforcing:**") < result.index("[View source ↗]")
        assert result.index("**Opposing:**") < result.index("[View source ↗]")


# ---------------------------------------------------------------------------
# _show_position_meaning — NiceGUI User interaction (plan 0024)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.nicegui_main_file(_MAIN)
class TestPositionMeaningDialog:
    """Clicking a position title opens the meaning popover (row layout)."""

    async def test_click_position_title_opens_meaning(self, user: User) -> None:
        await user.open("/")
        # The row layout renders the title as a ui.label ("Position 0")
        # alongside the per-position "📋 Position 0" detail button.  Filter by
        # kind to ensure we click the title label, not the button.
        user.find(kind=ui.label, content="Position 0").click()
        await user.should_see("Meaning of position 0.")
        await user.should_see("Source")

    async def test_meaning_dialog_has_close_button(self, user: User) -> None:
        await user.open("/")
        user.find(kind=ui.label, content="Position 1").click()
        await user.should_see("Meaning of position 1.")
        user.find("Close").click()

    async def test_click_title_works_after_dealing(self, user: User) -> None:
        await user.open("/")
        user.find("New Reading").click()
        await user.should_see("Summary")
        user.find(kind=ui.label, content="Position 2").click()
        await user.should_see("Meaning of position 2.")


# ---------------------------------------------------------------------------
# Grid-layout position-meaning dialog (plan 0024 + 0030)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.nicegui_main_file(_GRID_MAIN)
class TestPositionMeaningGridDialog:
    """Clicking a position title in the grid layout opens the meaning popover."""

    async def test_grid_position_title_opens_meaning(self, user: User) -> None:
        await user.open("/")
        user.find(kind=ui.label, content="Center").click()
        await user.should_see("The centre.")
        await user.should_see("Source")

    async def test_grid_title_click_does_not_open_card_detail(self, user: User) -> None:
        """Clicking the title should NOT also fire the cell's card-detail dialog.

        The browser-side ``stopPropagation`` keeps the two handlers from
        stepping on each other; the headless ``User`` simulation dispatches
        only to the clicked element's listeners, so only the title handler
        fires here.
        """
        await user.open("/")
        user.find(kind=ui.label, content="Crossing").click()
        await user.should_see("The crossing card.")
        # No card has been dealt yet, so the card-detail dialog would show
        # the "No card dealt" placeholder — assert it does NOT appear.
        await user.should_not_see("No card dealt")


# ---------------------------------------------------------------------------
# rotation_style — pure helper (plan 0025)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRotationStyle:
    def test_reversed_returns_180_transform(self) -> None:
        assert rotation_style(Orientation.REVERSED) == "transform: rotate(180deg);"

    def test_upright_returns_empty_string(self) -> None:
        assert rotation_style(Orientation.UPRIGHT) == ""


# ---------------------------------------------------------------------------
# Reversed card rotation — NiceGUI User interaction (plan 0025)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.nicegui_main_file(_MAIN)
class TestReversedCardRotation:
    """After a reading, reversed card images get a 180° CSS transform;

    upright images do not.  The stub uses ``random.Random(0)`` which deals
    cards 0-1 upright and card 2 reversed (3-position spread).
    """

    async def test_reversed_image_has_rotation_style(self, user: User) -> None:
        await user.open("/")
        user.find("New Reading").click()
        await user.should_see("Summary")
        with user.client:
            images = list(ElementFilter(kind=ui.image))
        transforms = [img.style.get("transform", "") for img in images]
        # Card 2 is reversed → one image should have the 180° transform.
        assert any("180deg" in t for t in transforms if t)

    async def test_upright_image_has_no_rotation_style(self, user: User) -> None:
        await user.open("/")
        user.find("New Reading").click()
        await user.should_see("Summary")
        with user.client:
            images = list(ElementFilter(kind=ui.image))
        transforms = [img.style.get("transform", "") for img in images]
        # Cards 0-1 are upright → at least one image has no transform.
        assert any(not t or "180deg" not in t for t in transforms)

    async def test_reading_clears_pre_existing_rotation_on_upright_slots(self, user: User) -> None:
        """The per-card clear path pops a pre-existing transform from upright slots.

        Plant a 180° transform on *every* (hidden) image before dealing, then run a
        reading: only the reversed slot should keep a transform — the upright slots
        must be cleared by the ``else`` branch in ``_run_reading``.

        Planting *before the first reading* is deliberate: the summary text is empty
        until a reading completes, so ``should_see("Summary")`` reliably waits for
        it. After a reading the summary persists, so a second click cannot be
        awaited that way (an earlier version of this test asserted against stale
        first-reading state and so never exercised clearing at all).
        """
        await user.open("/")
        with user.client:
            for img in ElementFilter(kind=ui.image):
                img.style("transform: rotate(180deg);")
        user.find("New Reading").click()
        await user.should_see("Summary")
        with user.client:
            transforms = [img.style.get("transform", "") for img in ElementFilter(kind=ui.image)]
        # The stub deals exactly one reversed card (3-position spread, seed 0): its
        # image keeps the 180° transform; the upright slots were cleared.
        assert sum("180deg" in t for t in transforms) == 1


# ---------------------------------------------------------------------------
# Deck selector (plan 0023)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.nicegui_main_file(_GRID_MAIN)
class TestDeckSelector:
    """Deck selector presence and deck switching (grid harness has two decks)."""

    async def test_deck_selector_present_in_grid_harness(self, user: User) -> None:
        await user.open("/")
        await user.should_see("Deck")

    async def test_default_deck_name_rendered(self, user: User) -> None:
        await user.open("/")
        # The grid harness defaults to "test-deck" → "Test Deck".
        await user.should_see("Test Deck")

    async def test_switching_deck_updates_active_deck(self, user: User) -> None:
        await user.open("/")
        await user.should_see("Test Deck")  # default deck in the title
        # Pick the other deck → on_value_change resolves the alt-deck service,
        # updates _current_deck_id / _cards_by_id, and rebuilds with a new title.
        selects = list(user.find(ui.select).elements)
        deck_select = next(s for s in selects if s.props.get("label") == "Deck")
        deck_select.set_value("alt-deck")
        await user.should_see("Alt Deck")
        assert nicegui_app_module._current_deck_id == "alt-deck"


@pytest.mark.unit
@pytest.mark.nicegui_main_file(_MAIN)
class TestNoDeckSelectorBackcompat:
    """Single-deck harness (no deck_options) should not render a deck selector."""

    async def test_no_deck_selector(self, user: User) -> None:
        await user.open("/")
        # The main harness has neither deck_options nor spread_options,
        # so no ui.select elements are rendered at all.
        await user.should_not_see(kind=ui.select)

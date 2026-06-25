"""Unit tests for the complex spreads feature (plan 0030).

Tests:
- ``list_spreads`` loader function.
- Celtic Cross spread JSON validation.
- Layout helpers (``_has_grid_layout``, ``_grid_dimensions``).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from nicegui import ui
from nicegui.testing import User
from pydantic import HttpUrl

import fortune_teller.application.ui.nicegui_app as nicegui_app_module
from fortune_teller.application.models.domain import Spread, SpreadPosition
from fortune_teller.application.services.loading import list_spread_ids, list_spreads
from fortune_teller.application.ui.nicegui_app import (
    _CARD_H,
    _CARD_W,
    _COLUMN_GAP,
    _card_box_style,
    _effective_dimensions,
    _effective_grid,
    _grid_container_style,
    _grid_dimensions,
    _has_grid_layout,
    _resolve_service,
    build_app,
)
from tests.unit.test_nicegui_app import _make_deck, _make_spread, _StubReadingService

_SPREAD_URL = "https://example.test/spread"


def _make_position(
    index: int,
    name: str = "Past",
    row: int | None = None,
    col: int | None = None,
    rotation: int = 0,
) -> SpreadPosition:
    return SpreadPosition(
        index=index,
        name=name,
        meaning=f"Meaning of {name}.",
        source_url=HttpUrl(_SPREAD_URL),
        row=row,
        col=col,
        rotation=rotation,
    )


# ---------------------------------------------------------------------------
# list_spreads / list_spread_ids
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestListSpreads:
    def test_returns_empty_for_missing_dir(self, tmp_path: Path) -> None:
        assert list_spreads(tmp_path / "nonexistent") == []

    def test_returns_empty_for_no_spreads(self, tmp_path: Path) -> None:
        spreads_dir = tmp_path / "spreads"
        spreads_dir.mkdir()
        assert list_spreads(tmp_path) == []

    def test_finds_spread_files(self, tmp_path: Path) -> None:
        spreads_dir = tmp_path / "spreads"
        spreads_dir.mkdir()
        spread = Spread(
            id="test-spread",
            name="Test Spread",
            positions=[_make_position(0)],
        )
        (spreads_dir / "test-spread.json").write_text(spread.model_dump_json())
        result = list_spreads(tmp_path)
        assert result == [("test-spread", "Test Spread")]

    def test_multiple_spreads_sorted(self, tmp_path: Path) -> None:
        spreads_dir = tmp_path / "spreads"
        spreads_dir.mkdir()
        for sid, name in [("bravo", "Bravo"), ("alpha", "Alpha")]:
            spread = Spread(id=sid, name=name, positions=[_make_position(0)])
            (spreads_dir / f"{sid}.json").write_text(spread.model_dump_json())
        result = list_spreads(tmp_path)
        assert result == [("alpha", "Alpha"), ("bravo", "Bravo")]

    def test_list_spread_ids_consistent_with_list_spreads(self, tmp_path: Path) -> None:
        spreads_dir = tmp_path / "spreads"
        spreads_dir.mkdir()
        spread = Spread(
            id="test-spread",
            name="Test Spread",
            positions=[_make_position(0)],
        )
        (spreads_dir / "test-spread.json").write_text(spread.model_dump_json())
        ids = list_spread_ids(tmp_path)
        tuples = list_spreads(tmp_path)
        assert [t[0] for t in tuples] == ids


# ---------------------------------------------------------------------------
# Celtic Cross spread JSON
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCelticCrossSpread:
    @pytest.fixture
    def celtic_cross_path(self) -> Path:
        return (
            Path(__file__).parent.parent.parent
            / "data"
            / "parsed"
            / "spreads"
            / "celtic-cross.json"
        )

    def test_celtic_cross_json_exists(self, celtic_cross_path: Path) -> None:
        assert celtic_cross_path.is_file(), f"Celtic Cross JSON not found: {celtic_cross_path}"

    def test_celtic_cross_validates_as_spread(self, celtic_cross_path: Path) -> None:
        spread = Spread.model_validate_json(celtic_cross_path.read_text())
        assert spread.id == "celtic-cross"
        assert spread.name == "Celtic Cross Spread"

    def test_celtic_cross_has_10_positions(self, celtic_cross_path: Path) -> None:
        spread = Spread.model_validate_json(celtic_cross_path.read_text())
        assert len(spread.positions) == 10
        indices = sorted(p.index for p in spread.positions)
        assert indices == list(range(10))

    def test_celtic_cross_all_positions_have_grid_coords(self, celtic_cross_path: Path) -> None:
        spread = Spread.model_validate_json(celtic_cross_path.read_text())
        for pos in spread.positions:
            assert pos.row is not None, f"Position {pos.index} has no row"
            assert pos.col is not None, f"Position {pos.index} has no col"

    def test_celtic_cross_has_exactly_one_90_degree_rotation(self, celtic_cross_path: Path) -> None:
        spread = Spread.model_validate_json(celtic_cross_path.read_text())
        rotated = [p for p in spread.positions if p.rotation == 90]
        assert len(rotated) == 1, f"Expected exactly one rotated position, got {len(rotated)}"
        assert rotated[0].name == "Challenge"

    def test_celtic_cross_positions_share_cell_for_crossing(self, celtic_cross_path: Path) -> None:
        spread = Spread.model_validate_json(celtic_cross_path.read_text())
        pos0 = spread.positions[0]  # Present
        pos1 = spread.positions[1]  # Challenge (crossing)
        assert pos0.row == pos1.row
        assert pos0.col == pos1.col


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLayoutHelpers:
    def test_has_grid_layout_true_when_all_present(self) -> None:
        positions = [_make_position(0, row=0, col=0), _make_position(1, row=1, col=1)]
        assert _has_grid_layout(positions)

    def test_has_grid_layout_false_when_any_absent(self) -> None:
        positions = [_make_position(0, row=0, col=0), _make_position(1)]
        assert not _has_grid_layout(positions)

    def test_has_grid_layout_false_when_all_absent(self) -> None:
        positions = [_make_position(0), _make_position(1)]
        assert not _has_grid_layout(positions)

    def test_grid_dimensions(self) -> None:
        positions = [
            _make_position(0, row=0, col=0),
            _make_position(1, row=2, col=3),
        ]
        rows, cols = _grid_dimensions(positions)
        assert rows == 3
        assert cols == 4

    def test_grid_dimensions_single_cell(self) -> None:
        positions = [_make_position(0, row=0, col=0)]
        rows, cols = _grid_dimensions(positions)
        assert rows == 1
        assert cols == 1

    def test_effective_grid_identity_for_grid_spread(self) -> None:
        positions = [
            _make_position(0, row=0, col=0),
            _make_position(1, row=0, col=0, rotation=90),
            _make_position(2, row=1, col=2),
        ]
        assert _effective_grid(positions) == [(0, 0), (0, 0), (1, 2)]

    def test_effective_grid_linear_when_no_coords(self) -> None:
        positions = [_make_position(0), _make_position(1), _make_position(2)]
        assert _effective_grid(positions) == [(0, 0), (0, 1), (0, 2)]

    def test_effective_dimensions_spans_coords(self) -> None:
        assert _effective_dimensions([(0, 0), (0, 0), (1, 2)]) == (2, 3)

    def test_effective_dimensions_single_linear_row(self) -> None:
        assert _effective_dimensions([(0, 0), (0, 1), (0, 2)]) == (1, 3)


# ---------------------------------------------------------------------------
# Layout style helpers
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestLayoutStyles:
    def test_column_gap_clears_rotated_overhang(self) -> None:
        # A 90° card overhangs its cell by (H - W)/2 each side; the gap must be
        # at least that so a crossing card never touches its neighbours.
        assert _COLUMN_GAP >= (_CARD_H - _CARD_W) // 2

    def test_grid_container_style_sets_fixed_track_sizes(self) -> None:
        style = _grid_container_style(rows=4, cols=5)
        assert "display:grid" in style
        assert f"repeat(5,{_CARD_W}px)" in style
        assert f"repeat(4,{_CARD_H}px)" in style
        assert f"column-gap:{_COLUMN_GAP}px" in style

    def test_card_box_style_places_cell_and_centres(self) -> None:
        style = _card_box_style(row=2, col=1, z=3, rotation=0)
        # CSS grid lines are 1-based, so row/col are offset by 1.
        assert "grid-row:3" in style
        assert "grid-column:2" in style
        assert "place-self:center" in style
        assert "z-index:3" in style
        assert f"width:{_CARD_W}px" in style
        assert f"height:{_CARD_H}px" in style
        assert "rotate" not in style

    def test_card_box_style_rotates_when_rotation_set(self) -> None:
        style = _card_box_style(row=0, col=0, z=1, rotation=90)
        assert "transform:rotate(90deg)" in style


# ---------------------------------------------------------------------------
# _resolve_service (spread → service resolution + caching)
# ---------------------------------------------------------------------------


def _grid_service() -> _StubReadingService:
    spread = Spread(
        id="grid-spread",
        name="Grid",
        positions=[_make_position(0, row=0, col=0), _make_position(1, row=0, col=1)],
    )
    return _StubReadingService(deck=_make_deck(6), spread=spread)


@pytest.mark.unit
class TestResolveService:
    def test_returns_default_when_ids_match(self) -> None:
        svc = _grid_service()
        build_app(svc)
        assert _resolve_service(svc.deck_id, "grid-spread") is svc

    def test_uses_factory_for_other_spread(self) -> None:
        other = _StubReadingService(deck=_make_deck(6), spread=_make_spread(3))
        build_app(_grid_service(), service_factory=lambda _did, _sid: other)
        assert _resolve_service("test-deck", "test-spread") is other

    def test_caches_factory_result(self) -> None:
        calls: list[tuple[str, str]] = []

        def factory(deck_id: str, spread_id: str) -> _StubReadingService:
            calls.append((deck_id, spread_id))
            return _StubReadingService(deck=_make_deck(6), spread=_make_spread(3))

        build_app(_grid_service(), service_factory=factory)  # type: ignore[arg-type]
        first = _resolve_service("test-deck", "test-spread")
        second = _resolve_service("test-deck", "test-spread")
        assert first is second
        assert calls == [("test-deck", "test-spread")]

    def test_falls_back_to_default_without_factory(self) -> None:
        default = _grid_service()
        build_app(default)
        assert _resolve_service("test-deck", "unknown-spread") is default

    def test_raises_when_no_service(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(nicegui_app_module, "_service", None)
        monkeypatch.setattr(nicegui_app_module, "_service_factory", None)
        monkeypatch.setattr(nicegui_app_module, "_service_cache", {})
        with pytest.raises(RuntimeError):
            _resolve_service("any-deck", "anything")

    def test_caches_per_deck_for_same_spread(self) -> None:
        """The cache key is (deck_id, spread_id): same spread, different decks
        resolve to distinct services, each built once."""
        calls: list[tuple[str, str]] = []

        def factory(deck_id: str, spread_id: str) -> _StubReadingService:
            calls.append((deck_id, spread_id))
            return _StubReadingService(deck=_make_deck(6), spread=_make_spread(3))

        build_app(_grid_service(), service_factory=factory)  # type: ignore[arg-type]
        deck_a = _resolve_service("deck-a", "grid-spread")
        deck_b = _resolve_service("deck-b", "grid-spread")
        assert deck_a is not deck_b
        # Each (deck, spread) pair built exactly once; repeat hits the cache.
        assert _resolve_service("deck-a", "grid-spread") is deck_a
        assert calls == [("deck-a", "grid-spread"), ("deck-b", "grid-spread")]


# ---------------------------------------------------------------------------
# Grid layout + spread selector (nicegui.testing User simulation)
# ---------------------------------------------------------------------------

_GRID_MAIN = "tests/nicegui_grid_main.py"


@pytest.mark.unit
@pytest.mark.nicegui_main_file(_GRID_MAIN)
class TestGridLayoutUI:
    async def test_grid_renders_all_position_labels(self, user: User) -> None:
        await user.open("/")
        for name in ("Center", "Crossing", "Right", "Below"):
            await user.should_see(name)

    async def test_spread_selector_present(self, user: User) -> None:
        await user.open("/")
        await user.should_see(kind=ui.select)

    async def test_new_reading_deals_in_grid(self, user: User) -> None:
        await user.open("/")
        user.find("New Reading").click()
        await user.should_see("UPRIGHT")
        await user.should_see("Summary")

    async def test_switching_to_row_spread_rebuilds_layout(self, user: User) -> None:
        await user.open("/")
        await user.should_see("Center")  # grid layout active
        # Picking the other spread fires on_value_change → rebuild.
        # With deck_options there are two selects; find the one labelled "Spread".
        selects = list(user.find(ui.select).elements)
        spread_select = next(s for s in selects if s.props.get("label") == "Spread")
        spread_select.set_value("test-spread")
        # The unified renderer drops the old per-position 📋 button; the
        # numbered list now exposes a "Details · <name>" button instead.
        await user.should_see("Details · Position 0")
        await user.should_see("Test Spread")  # title updated to the new spread

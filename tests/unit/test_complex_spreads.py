"""Unit tests for the complex spreads feature (plan 0030).

Tests:
- ``list_spreads`` loader function.
- Celtic Cross spread JSON validation.
- Layout helpers (``_has_grid_layout``, ``_grid_dimensions``).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import HttpUrl

from fortune_teller.application.models.domain import Spread, SpreadPosition
from fortune_teller.application.services.loading import list_spread_ids, list_spreads
from fortune_teller.application.ui.nicegui_app import _grid_dimensions, _has_grid_layout

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

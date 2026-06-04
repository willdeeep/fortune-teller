"""Unit tests for ``ft-build-index`` CLI.

Exercises the CLI via :func:`typer.testing.CliRunner` against a
``tmp_path``-isolated data directory. Reads embedded JSON written by
``ft-embed`` (or directly here) and asserts that the DuckDB file is
populated correctly.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from fortune_teller.application.config import Settings
from fortune_teller.application.models.domain import (
    CardSection,
    Chunk,
    ChunkType,
)
from fortune_teller.application.stores.vector import VectorStore
from fortune_teller.developer.build_index import cli as build_cli
from fortune_teller.developer.build_index.cli import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_data_dir(tmp_path: Path, monkeypatch) -> Settings:
    """Point ``settings.ft_data_dir`` at a fresh ``tmp_path`` directory."""
    data_dir = tmp_path / "data"
    settings = Settings(ft_data_dir=data_dir)
    monkeypatch.setattr(build_cli, "settings", settings)
    return settings


def _write_embedded(
    embedded_dir: Path,
    *,
    subdir: str,
    source: str,
    chunks: list[dict[str, Any]],
) -> None:
    """Write an embedded chunks JSON file under *embedded_dir*."""
    out_dir = embedded_dir / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{source}.json").write_text(json.dumps(chunks), encoding="utf-8")


def _make_chunk_dict(
    *,
    text: str,
    embedding: list[float],
    chunk_type: str = "card_section",
    card_id: str | None = "the-fool",
    section: str | None = "drive",
    spread_id: str | None = None,
    position_index: int | None = None,
) -> dict[str, Any]:
    return {
        "id": str(uuid.uuid4()),
        "chunk_type": chunk_type,
        "deck_id": "book-of-thoth" if chunk_type == "card_section" else None,
        "card_id": card_id,
        "card_name": "The Fool" if card_id else None,
        "section": section,
        "spread_id": spread_id,
        "position_index": position_index,
        "source_url": "https://example.com/",
        "text": text,
        "embedding": embedding,
    }


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildIndexCliPaths:
    def test_creates_duckdb_file(self, isolated_data_dir: Settings) -> None:
        embedded = isolated_data_dir.ft_data_dir / "embedded"
        _write_embedded(
            embedded,
            subdir="book-of-thoth",
            source="the-fool",
            chunks=[_make_chunk_dict(text="x", embedding=[0.1, 0.2, 0.3, 0.4])],
        )

        result = runner.invoke(app, [])
        assert result.exit_code == 0, result.output

        db_path = isolated_data_dir.ft_data_dir / "duckdb" / "fortune.duckdb"
        assert db_path.exists()

    def test_no_embedded_dir_exits_nonzero(self, isolated_data_dir: Settings) -> None:  # noqa: ARG002
        result = runner.invoke(app, [])
        assert result.exit_code != 0
        assert "no embedded chunks" in result.output.lower()

    def test_embedded_dir_empty_exits_nonzero(self, isolated_data_dir: Settings) -> None:
        # Create an empty embedded/ directory.
        (isolated_data_dir.ft_data_dir / "embedded").mkdir(parents=True)
        result = runner.invoke(app, [])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildIndexCliIngestion:
    def test_indexes_all_chunks(self, isolated_data_dir: Settings) -> None:
        embedded = isolated_data_dir.ft_data_dir / "embedded"
        _write_embedded(
            embedded,
            subdir="book-of-thoth",
            source="the-fool",
            chunks=[
                _make_chunk_dict(text=f"chunk-{i}", embedding=[0.1 * (i + 1)] * 4) for i in range(5)
            ],
        )

        result = runner.invoke(app, [])
        assert result.exit_code == 0, result.output
        assert "Indexed 5 chunks" in result.output

    def test_skips_chunks_with_empty_embedding(self, isolated_data_dir: Settings) -> None:
        embedded = isolated_data_dir.ft_data_dir / "embedded"
        _write_embedded(
            embedded,
            subdir="book-of-thoth",
            source="the-fool",
            chunks=[
                _make_chunk_dict(text="good", embedding=[0.1, 0.2, 0.3, 0.4]),
                _make_chunk_dict(text="empty", embedding=[]),
            ],
        )

        result = runner.invoke(app, [])
        assert result.exit_code == 0, result.output
        # 1 indexed, 1 skipped
        assert "Indexed 1 chunks" in result.output
        assert "Skipped 1 chunks" in result.output

    def test_processes_multiple_subdirs(self, isolated_data_dir: Settings) -> None:
        embedded = isolated_data_dir.ft_data_dir / "embedded"
        _write_embedded(
            embedded,
            subdir="book-of-thoth",
            source="the-fool",
            chunks=[_make_chunk_dict(text="a", embedding=[0.1, 0.2, 0.3, 0.4])],
        )
        _write_embedded(
            embedded,
            subdir="spreads",
            source="new-moon-three-card",
            chunks=[
                _make_chunk_dict(
                    text="position-0",
                    embedding=[0.5, 0.6, 0.7, 0.8],
                    chunk_type="spread_position",
                    card_id=None,
                    section=None,
                    spread_id="new-moon-three-card",
                    position_index=0,
                ),
            ],
        )

        result = runner.invoke(app, [])
        assert result.exit_code == 0, result.output
        assert "Indexed 2 chunks" in result.output


# ---------------------------------------------------------------------------
# Rebuild semantics
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildIndexCliRebuild:
    def test_default_rebuild_clears_existing(self, isolated_data_dir: Settings) -> None:
        embedded = isolated_data_dir.ft_data_dir / "embedded"
        _write_embedded(
            embedded,
            subdir="book-of-thoth",
            source="the-fool",
            chunks=[
                _make_chunk_dict(text=f"a{i}", embedding=[0.1 * (i + 1)] * 4) for i in range(3)
            ],
        )

        # First build: 3 chunks
        result = runner.invoke(app, [])
        assert result.exit_code == 0, result.output
        assert "Indexed 3 chunks" in result.output

        # Replace the source with fewer chunks and rebuild.
        _write_embedded(
            embedded,
            subdir="book-of-thoth",
            source="the-fool",
            chunks=[
                _make_chunk_dict(text="only", embedding=[0.9, 0.9, 0.9, 0.9]),
            ],
        )
        result = runner.invoke(app, [])
        assert result.exit_code == 0, result.output
        # Default --rebuild replaces, so the count is 1, not 4.
        assert "Indexed 1 chunks" in result.output

    def test_no_rebuild_appends(self, isolated_data_dir: Settings) -> None:
        embedded = isolated_data_dir.ft_data_dir / "embedded"
        _write_embedded(
            embedded,
            subdir="book-of-thoth",
            source="the-fool",
            chunks=[
                _make_chunk_dict(text="a", embedding=[0.1, 0.2, 0.3, 0.4]),
            ],
        )

        # Initial build
        result = runner.invoke(app, [])
        assert result.exit_code == 0, result.output

        # Append a new card
        _write_embedded(
            embedded,
            subdir="book-of-thoth",
            source="the-magician",
            chunks=[
                _make_chunk_dict(
                    text="b",
                    embedding=[0.5, 0.6, 0.7, 0.8],
                    card_id="the-magician",
                    section="light",
                ),
            ],
        )
        result = runner.invoke(app, ["--no-rebuild"])
        assert result.exit_code == 0, result.output
        # Without rebuild, 1 + 1 = 2 chunks total
        assert "Indexed 2 chunks" in result.output


# ---------------------------------------------------------------------------
# Integration smoke: search after build
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildIndexCliEndToEnd:
    def test_chunks_are_searchable_after_build(self, isolated_data_dir: Settings) -> None:
        """End-to-end: build an index, then verify a SearchHit comes back."""
        embedded = isolated_data_dir.ft_data_dir / "embedded"
        # Two orthogonal chunks so similarity is easy to verify.
        _write_embedded(
            embedded,
            subdir="book-of-thoth",
            source="the-fool",
            chunks=[
                _make_chunk_dict(
                    text="drive-x",
                    embedding=[1.0, 0.0, 0.0, 0.0],
                    section="drive",
                ),
                _make_chunk_dict(
                    text="light-x",
                    embedding=[0.0, 1.0, 0.0, 0.0],
                    section="light",
                ),
            ],
        )

        result = runner.invoke(app, [])
        assert result.exit_code == 0, result.output

        db_path = isolated_data_dir.ft_data_dir / "duckdb" / "fortune.duckdb"
        with VectorStore(db_path, dimension=4) as store:
            assert store.count() == 2
            hits = store.search([1.0, 0.0, 0.0, 0.0], k=1)
            assert hits[0].chunk.text == "drive-x"
            assert hits[0].score == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildIndexCliHelpers:
    def test_load_embedded_chunks_handles_nested_dirs(self, tmp_path: Path) -> None:
        """_load_embedded_chunks must find JSON under any nesting depth."""
        _write_embedded(
            tmp_path,
            subdir="deck-a",
            source="c1",
            chunks=[_make_chunk_dict(text="a", embedding=[0.1, 0.2, 0.3, 0.4])],
        )
        _write_embedded(
            tmp_path,
            subdir="spreads",
            source="spread-x",
            chunks=[
                _make_chunk_dict(
                    text="b",
                    embedding=[0.5, 0.6, 0.7, 0.8],
                    chunk_type="spread_position",
                    card_id=None,
                    section=None,
                    spread_id="spread-x",
                    position_index=0,
                ),
            ],
        )

        chunks = build_cli._load_embedded_chunks(tmp_path)
        assert len(chunks) == 2

    def test_validate_embedded_filters_empty(self) -> None:
        good = Chunk(
            chunk_type=ChunkType.CARD_SECTION,
            deck_id="d",
            card_id="c",
            section=CardSection.DRIVE,
            source_url="https://example.com/",
            text="good",
            embedding=[0.1, 0.2, 0.3, 0.4],
        )
        empty = Chunk(
            chunk_type=ChunkType.CARD_SECTION,
            deck_id="d",
            card_id="c",
            section=CardSection.DRIVE,
            source_url="https://example.com/",
            text="empty",
            embedding=[],
        )
        none_emb = Chunk(
            chunk_type=ChunkType.CARD_SECTION,
            deck_id="d",
            card_id="c",
            section=CardSection.DRIVE,
            source_url="https://example.com/",
            text="none",
            embedding=None,
        )
        validated = build_cli._validate_embedded([good, empty, none_emb])
        assert len(validated) == 1
        assert validated[0].text == "good"

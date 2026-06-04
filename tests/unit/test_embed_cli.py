"""Unit tests for ``ft-embed`` CLI.

The CLI is exercised through :func:`typer.testing.CliRunner` and uses the
``stub_embedder_factory`` fixture to avoid loading any model weights.
All filesystem mutations live under ``tmp_path``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

import pytest
from typer.testing import CliRunner

from fortune_teller.application.config import Settings
from fortune_teller.application.models.domain import (
    CardSection,
    Chunk,
    ChunkType,
)
from fortune_teller.developer.embed import cli as embed_cli
from fortune_teller.developer.embed.cli import app

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


runner = CliRunner()


def _write_card(deck_dir: Path, slug: str, **overrides: Any) -> None:
    """Write a minimal Card JSON under *deck_dir* with the given overrides."""
    payload: dict[str, Any] = {
        "id": slug,
        "name": slug.replace("-", " ").title(),
        "arcana": "major",
        "suit": None,
        "number": 0,
        "sections": [
            {"section": "drive", "text": f"Drive text for {slug}."},
            {"section": "light", "text": f"Light text for {slug}."},
        ],
        "source_url": f"https://thothreadings.com/blog/{slug}/",
    }
    payload.update(overrides)
    deck_dir.mkdir(parents=True, exist_ok=True)
    (deck_dir / f"{slug}.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_spread(spreads_dir: Path, spread_id: str) -> None:
    payload = {
        "id": spread_id,
        "name": spread_id.replace("-", " ").title(),
        "positions": [
            {
                "index": 0,
                "name": "Past",
                "meaning": "What has been.",
                "source_url": "https://thothreadings.com/spread-new-moon/",
            },
            {
                "index": 1,
                "name": "Present",
                "meaning": "What is.",
                "source_url": "https://thothreadings.com/spread-new-moon/",
            },
        ],
    }
    spreads_dir.mkdir(parents=True, exist_ok=True)
    (spreads_dir / f"{spread_id}.json").write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def isolated_data_dir(tmp_path: Path, monkeypatch, stub_embedder_factory) -> Settings:
    """Point ``settings.ft_data_dir`` at a fresh ``tmp_path`` and patch the
    CLI to use the stub embedder.

    Returns the Settings instance with the patched path so tests can
    read the resolved path off of it.
    """
    data_dir = tmp_path / "data"
    parsed_dir = data_dir / "parsed"
    parsed_dir.mkdir(parents=True)

    settings = Settings(ft_data_dir=data_dir)
    monkeypatch.setattr(embed_cli, "settings", settings)
    # Make the CLI use a stubbed embedder so no model is loaded.
    monkeypatch.setattr(embed_cli, "Embedder", lambda: stub_embedder_factory(dim=4))
    return settings


# ---------------------------------------------------------------------------
# Card processing
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEmbedCliCards:
    def test_writes_one_json_per_card(self, isolated_data_dir: Settings) -> None:
        deck_dir = isolated_data_dir.ft_data_dir / "parsed" / "book-of-thoth"
        _write_card(deck_dir, "0-the-fool")
        _write_card(deck_dir, "i-the-magician")

        result = runner.invoke(app, [])
        assert result.exit_code == 0, result.output

        out_deck = isolated_data_dir.ft_data_dir / "embedded" / "book-of-thoth"
        assert (out_deck / "0-the-fool.json").exists()
        assert (out_deck / "i-the-magician.json").exists()

    def test_output_chunks_have_embeddings(self, isolated_data_dir: Settings) -> None:
        deck_dir = isolated_data_dir.ft_data_dir / "parsed" / "book-of-thoth"
        _write_card(deck_dir, "0-the-fool")

        runner.invoke(app, [])

        embedded_path = (
            isolated_data_dir.ft_data_dir / "embedded" / "book-of-thoth" / "0-the-fool.json"
        )
        chunks = json.loads(embedded_path.read_text())
        assert len(chunks) == 2  # two sections per card in _write_card
        for c in chunks:
            assert c["embedding"] is not None
            assert len(c["embedding"]) == 4
            assert c["chunk_type"] == "card_section"
            assert c["card_id"] == "0-the-fool"
            assert c["deck_id"] == "book-of-thoth"

    def test_deck_filter_limits_to_one_deck(self, isolated_data_dir: Settings) -> None:
        a = isolated_data_dir.ft_data_dir / "parsed" / "deck-a"
        b = isolated_data_dir.ft_data_dir / "parsed" / "deck-b"
        _write_card(a, "0-the-fool")
        _write_card(b, "i-the-magician")

        result = runner.invoke(app, ["--deck", "deck-a"])
        assert result.exit_code == 0, result.output

        out_a = isolated_data_dir.ft_data_dir / "embedded" / "deck-a"
        out_b = isolated_data_dir.ft_data_dir / "embedded" / "deck-b"
        assert (out_a / "0-the-fool.json").exists()
        assert not (out_b / "i-the-magician.json").exists()

    def test_unknown_deck_exits_nonzero(self, isolated_data_dir: Settings) -> None:  # noqa: ARG002
        result = runner.invoke(app, ["--deck", "nope"])
        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "deck" in result.output.lower()

    def test_empty_parsed_dir_exits_nonzero(self, isolated_data_dir: Settings) -> None:  # noqa: ARG002
        result = runner.invoke(app, [])
        assert result.exit_code != 0
        assert "no parsed data" in result.output.lower()


# ---------------------------------------------------------------------------
# Spread processing
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEmbedCliSpreads:
    def test_writes_spread_json(self, isolated_data_dir: Settings) -> None:
        spreads_dir = isolated_data_dir.ft_data_dir / "parsed" / "spreads"
        _write_spread(spreads_dir, "new-moon-three-card")

        result = runner.invoke(app, [])
        assert result.exit_code == 0, result.output

        out = isolated_data_dir.ft_data_dir / "embedded" / "spreads" / "new-moon-three-card.json"
        assert out.exists()

    def test_spread_chunks_have_correct_metadata(self, isolated_data_dir: Settings) -> None:
        spreads_dir = isolated_data_dir.ft_data_dir / "parsed" / "spreads"
        _write_spread(spreads_dir, "new-moon-three-card")

        runner.invoke(app, [])

        spread_path = (
            isolated_data_dir.ft_data_dir / "embedded" / "spreads" / "new-moon-three-card.json"
        )
        chunks = json.loads(spread_path.read_text())
        assert len(chunks) == 2
        for c in chunks:
            assert c["chunk_type"] == "spread_position"
            assert c["spread_id"] == "new-moon-three-card"
            assert c["embedding"] is not None
            assert len(c["embedding"]) == 4

    def test_no_spreads_dir_is_not_an_error(self, isolated_data_dir: Settings) -> None:
        """A deck with cards but no spreads/ subdirectory must succeed."""
        deck_dir = isolated_data_dir.ft_data_dir / "parsed" / "book-of-thoth"
        _write_card(deck_dir, "0-the-fool")
        result = runner.invoke(app, [])
        assert result.exit_code == 0, result.output
        # No "spreads" subdir under embedded/ since there were no spreads.
        assert not (isolated_data_dir.ft_data_dir / "embedded" / "spreads").exists()


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEmbedCliErrors:
    def test_malformed_card_skipped_with_error_count(self, isolated_data_dir: Settings) -> None:
        deck_dir = isolated_data_dir.ft_data_dir / "parsed" / "book-of-thoth"
        deck_dir.mkdir(parents=True)
        # Write a card missing the required "id" field
        (deck_dir / "bad.json").write_text('{"name": "bad"}', encoding="utf-8")
        # And a good one
        _write_card(deck_dir, "0-the-fool")

        result = runner.invoke(app, [])
        # 1 error -> exit 1
        assert result.exit_code == 1
        assert "1 errors" in result.output or "ERROR" in result.output

    def test_summary_line_in_output(self, isolated_data_dir: Settings) -> None:
        deck_dir = isolated_data_dir.ft_data_dir / "parsed" / "book-of-thoth"
        _write_card(deck_dir, "0-the-fool")
        _write_card(deck_dir, "i-the-magician")
        spreads_dir = isolated_data_dir.ft_data_dir / "parsed" / "spreads"
        _write_spread(spreads_dir, "new-moon-three-card")

        result = runner.invoke(app, [])
        assert "2 cards" in result.output
        assert "1 spreads" in result.output
        assert "Done" in result.output


# ---------------------------------------------------------------------------
# _write_chunks smoke (module-level helper)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestWriteChunksHelper:
    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        out_path = tmp_path / "nested" / "sub" / "out.json"
        chunks = [
            Chunk(
                chunk_type=ChunkType.CARD_SECTION,
                deck_id="d",
                card_id="c",
                section=CardSection.DRIVE,
                source_url="https://example.com/",
                text="hello",
                embedding=[0.0, 0.0],
            )
        ]
        embed_cli._write_chunks(out_path, chunks)
        assert out_path.exists()
        loaded = json.loads(out_path.read_text())
        assert loaded[0]["text"] == "hello"

    def test_json_is_an_array(self, tmp_path: Path) -> None:
        chunks = [
            Chunk(
                chunk_type=ChunkType.CARD_SECTION,
                deck_id="d",
                card_id="c",
                section=CardSection.DRIVE,
                source_url="https://example.com/",
                text="a",
            ),
            Chunk(
                chunk_type=ChunkType.CARD_SECTION,
                deck_id="d",
                card_id="c",
                section=CardSection.LIGHT,
                source_url="https://example.com/",
                text="b",
            ),
        ]
        out = tmp_path / "x.json"
        embed_cli._write_chunks(out, chunks)
        loaded = json.loads(out.read_text())
        assert isinstance(loaded, list)
        assert len(loaded) == 2


# Use the ClassVar import to satisfy ruff (re-export guard)
_ = ClassVar

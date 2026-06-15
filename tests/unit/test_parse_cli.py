"""Unit tests for the ``ft-parse`` CLI source selection.

With no cached HTML present, each source is *skipped* (not an error), so
these tests verify which sources the CLI visits — default visits all — without
needing HTML fixtures or network.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from fortune_teller.application.config import Settings
from fortune_teller.developer.parse import cli as parse_cli
from fortune_teller.developer.parse.cli import app

runner = CliRunner()


@pytest.fixture
def isolated_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    settings = Settings(ft_data_dir=tmp_path / "data")
    monkeypatch.setattr(parse_cli, "settings", settings)
    return settings


@pytest.mark.unit
class TestParseSourceSelection:
    def test_default_parses_all_sources(self, isolated_data_dir: Settings) -> None:  # noqa: ARG002
        result = runner.invoke(app, [])
        assert result.exit_code == 0, result.output
        assert "thothreadings" in result.output
        assert "learntarot" in result.output

    def test_single_source_only(self, isolated_data_dir: Settings) -> None:  # noqa: ARG002
        result = runner.invoke(app, ["--source", "learntarot"])
        assert result.exit_code == 0, result.output
        assert "learntarot" in result.output
        assert "thothreadings" not in result.output

    def test_unknown_source_errors(self, isolated_data_dir: Settings) -> None:  # noqa: ARG002
        result = runner.invoke(app, ["--source", "bogus"])
        assert result.exit_code != 0
        assert "Unknown source" in result.output

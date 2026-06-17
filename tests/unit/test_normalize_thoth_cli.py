"""Unit tests for :mod:`fortune_teller.developer.normalize.thoth_cli`.

Covers the ``ft-normalize-thoth`` CLI entry point.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from fortune_teller.developer.normalize.thoth_cli import app


@pytest.mark.unit
class TestNormalizeThothCli:
    """CLI invocation tests for ``ft-normalize-thoth``."""

    def test_no_llm_dry_run(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``--no-llm`` should call ``synthesize_deck_synergies`` with ``llm=None``
        and *not* call ``build_normalize_model``.
        """
        (tmp_path / "parsed" / "book-of-thoth").mkdir(parents=True)
        monkeypatch.setattr(
            "fortune_teller.developer.normalize.thoth_cli.settings.ft_data_dir",
            tmp_path,
        )

        synthesize_mock = MagicMock(return_value=[])
        monkeypatch.setattr(
            "fortune_teller.developer.normalize.thoth_cli.synthesize_deck_synergies",
            synthesize_mock,
        )
        # build_normalize_model should NOT be called in dry-run mode
        build_mock = MagicMock()
        monkeypatch.setattr(
            "fortune_teller.developer.normalize.thoth_cli.build_normalize_model",
            build_mock,
        )

        runner = CliRunner()
        result = runner.invoke(app, ["--no-llm"])

        assert result.exit_code == 0
        synthesize_mock.assert_called_once_with(
            tmp_path / "parsed",
            llm=None,
            only=None,
        )
        build_mock.assert_not_called()

    def test_with_provider(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``--provider api`` should build the LLM and pass it through to the
        synthesizer.
        """
        (tmp_path / "parsed" / "book-of-thoth").mkdir(parents=True)
        monkeypatch.setattr(
            "fortune_teller.developer.normalize.thoth_cli.settings.ft_data_dir",
            tmp_path,
        )

        build_mock = MagicMock(return_value="fake-llm")
        monkeypatch.setattr(
            "fortune_teller.developer.normalize.thoth_cli.build_normalize_model",
            build_mock,
        )
        synthesize_mock = MagicMock(return_value=[])
        monkeypatch.setattr(
            "fortune_teller.developer.normalize.thoth_cli.synthesize_deck_synergies",
            synthesize_mock,
        )

        runner = CliRunner()
        result = runner.invoke(app, ["--provider", "api"])

        assert result.exit_code == 0
        build_mock.assert_called_once_with("api", "claude-sonnet-4-6")
        synthesize_mock.assert_called_once_with(
            tmp_path / "parsed",
            llm="fake-llm",
            only=None,
        )

    def test_only_flag(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``--only the-fool,the-magician`` should parse the comma-separated
        list into a set and pass it as ``only``.
        """
        (tmp_path / "parsed" / "book-of-thoth").mkdir(parents=True)
        monkeypatch.setattr(
            "fortune_teller.developer.normalize.thoth_cli.settings.ft_data_dir",
            tmp_path,
        )

        build_mock = MagicMock(return_value="fake-llm")
        monkeypatch.setattr(
            "fortune_teller.developer.normalize.thoth_cli.build_normalize_model",
            build_mock,
        )
        synthesize_mock = MagicMock(return_value=[])
        monkeypatch.setattr(
            "fortune_teller.developer.normalize.thoth_cli.synthesize_deck_synergies",
            synthesize_mock,
        )

        runner = CliRunner()
        result = runner.invoke(app, ["--only", "the-fool,the-magician"])

        assert result.exit_code == 0
        synthesize_mock.assert_called_once_with(
            tmp_path / "parsed",
            llm="fake-llm",
            only={"the-fool", "the-magician"},
        )

    def test_missing_parsed_dir(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When ``parsed/book-of-thoth`` does not exist, the CLI should exit
        with code 1.
        """
        monkeypatch.setattr(
            "fortune_teller.developer.normalize.thoth_cli.settings.ft_data_dir",
            tmp_path,
        )
        # Intentionally *not* creating parsed/book-of-thoth

        runner = CliRunner()
        result = runner.invoke(app, [])

        assert result.exit_code == 1

"""Unit tests for :mod:`fortune_teller.developer.normalize.cli`.

Covers the ``ft-normalize-rw`` CLI entry point.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from fortune_teller.developer.normalize.cli import app
from fortune_teller.developer.normalize.rider_waite import NormalizeOutcome


@pytest.mark.unit
class TestNormalizeRwCli:
    """CLI invocation tests for ``ft-normalize-rw``."""

    def test_no_llm_dry_run(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``--no-llm`` should call ``normalize_deck`` with ``llm=None`` and
        *not* build an LLM.
        """
        (tmp_path / "raw" / "rider-waite").mkdir(parents=True)
        monkeypatch.setattr(
            "fortune_teller.developer.normalize.cli.settings.ft_data_dir",
            tmp_path,
        )

        normalize_mock = MagicMock(return_value=NormalizeOutcome())
        monkeypatch.setattr(
            "fortune_teller.developer.normalize.cli.normalize_deck",
            normalize_mock,
        )
        build_mock = MagicMock()
        monkeypatch.setattr(
            "fortune_teller.developer.normalize.cli.build_normalize_model",
            build_mock,
        )

        runner = CliRunner()
        result = runner.invoke(app, ["--no-llm"])

        assert result.exit_code == 0
        normalize_mock.assert_called_once_with(
            tmp_path / "raw" / "rider-waite",
            tmp_path / "parsed" / "rider-waite",
            llm=None,
            only=None,
        )
        build_mock.assert_not_called()

    def test_with_provider(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``--provider api`` should build the LLM and pass it to ``normalize_deck``."""
        (tmp_path / "raw" / "rider-waite").mkdir(parents=True)
        monkeypatch.setattr(
            "fortune_teller.developer.normalize.cli.settings.ft_data_dir",
            tmp_path,
        )

        build_mock = MagicMock(return_value="fake-llm")
        monkeypatch.setattr(
            "fortune_teller.developer.normalize.cli.build_normalize_model",
            build_mock,
        )
        normalize_mock = MagicMock(return_value=NormalizeOutcome())
        monkeypatch.setattr(
            "fortune_teller.developer.normalize.cli.normalize_deck",
            normalize_mock,
        )

        runner = CliRunner()
        result = runner.invoke(app, ["--provider", "api"])

        assert result.exit_code == 0
        build_mock.assert_called_once_with("api", "claude-sonnet-4-6")
        normalize_mock.assert_called_once_with(
            tmp_path / "raw" / "rider-waite",
            tmp_path / "parsed" / "rider-waite",
            llm="fake-llm",
            only=None,
        )

    def test_only_flag(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``--only the-fool,seven-of-cups`` should parse into a set passed as ``only``."""
        (tmp_path / "raw" / "rider-waite").mkdir(parents=True)
        monkeypatch.setattr(
            "fortune_teller.developer.normalize.cli.settings.ft_data_dir",
            tmp_path,
        )

        build_mock = MagicMock(return_value="fake-llm")
        monkeypatch.setattr(
            "fortune_teller.developer.normalize.cli.build_normalize_model",
            build_mock,
        )
        normalize_mock = MagicMock(return_value=NormalizeOutcome())
        monkeypatch.setattr(
            "fortune_teller.developer.normalize.cli.normalize_deck",
            normalize_mock,
        )

        runner = CliRunner()
        result = runner.invoke(app, ["--only", "the-fool,seven-of-cups"])

        assert result.exit_code == 0
        normalize_mock.assert_called_once_with(
            tmp_path / "raw" / "rider-waite",
            tmp_path / "parsed" / "rider-waite",
            llm="fake-llm",
            only={"the-fool", "seven-of-cups"},
        )

    def test_missing_raw_dir(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When ``raw/rider-waite`` does not exist, the CLI should exit with code 1."""
        monkeypatch.setattr(
            "fortune_teller.developer.normalize.cli.settings.ft_data_dir",
            tmp_path,
        )
        # Intentionally *not* creating raw/rider-waite

        runner = CliRunner()
        result = runner.invoke(app, [])

        assert result.exit_code == 1


@pytest.mark.unit
class TestFailureReporting:
    """Partial failures: every card is attempted, then the CLI exits 1 (plan: exit codes)."""

    def _invoke(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        outcome: NormalizeOutcome,
        args: list[str],
    ):
        (tmp_path / "raw/rider-waite").mkdir(parents=True)
        monkeypatch.setattr("fortune_teller.developer.normalize.cli.settings.ft_data_dir", tmp_path)
        monkeypatch.setattr(
            "fortune_teller.developer.normalize.cli.normalize_deck", MagicMock(return_value=outcome)
        )
        monkeypatch.setattr(
            "fortune_teller.developer.normalize.cli.build_normalize_model",
            MagicMock(return_value="fake-llm"),
        )
        return CliRunner().invoke(app, args)

    def test_failures_exit_1_with_retry_command(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        outcome = NormalizeOutcome(failed=[("the-tower", "boom"), ("death", "[Errno 61] refused")])
        result = self._invoke(tmp_path, monkeypatch, outcome, ["--provider", "local"])

        assert result.exit_code == 1
        assert "0 cards; 2 failed" in result.output
        assert "[Errno 61] refused" in result.output  # rich markup escaped, not swallowed
        assert (
            "uv run ft-normalize-rw --provider local --model claude-sonnet-4-6"
            " --only the-tower,death" in result.output
        )

    def test_no_failures_exit_0(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        result = self._invoke(tmp_path, monkeypatch, NormalizeOutcome(), [])

        assert result.exit_code == 0
        assert "failed" not in result.output

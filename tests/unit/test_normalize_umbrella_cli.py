"""Unit tests for :mod:`fortune_teller.developer.normalize.umbrella_cli`.

Covers the ``ft-normalize`` umbrella CLI entry point.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from fortune_teller.developer.normalize.rider_waite import NormalizeOutcome
from fortune_teller.developer.normalize.umbrella_cli import app


@pytest.mark.unit
class TestNormalizeUmbrellaCli:
    """CLI invocation tests for ``ft-normalize``."""

    # ------------------------------------------------------------------
    # Deck selection: --deck all / rw / thoth
    # ------------------------------------------------------------------

    def test_deck_all(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``--deck all`` (the default) should call both
        ``normalize_deck`` and ``synthesize_deck_synergies``.
        """
        (tmp_path / "raw" / "rider-waite").mkdir(parents=True)
        (tmp_path / "parsed" / "book-of-thoth").mkdir(parents=True)
        monkeypatch.setattr(
            "fortune_teller.developer.normalize.umbrella_cli.settings.ft_data_dir",
            tmp_path,
        )

        normalize_mock = MagicMock(return_value=NormalizeOutcome())
        monkeypatch.setattr(
            "fortune_teller.developer.normalize.umbrella_cli.normalize_deck",
            normalize_mock,
        )
        synthesize_mock = MagicMock(return_value=NormalizeOutcome())
        monkeypatch.setattr(
            "fortune_teller.developer.normalize.umbrella_cli.synthesize_deck_synergies",
            synthesize_mock,
        )
        build_mock = MagicMock(return_value="fake-llm")
        monkeypatch.setattr(
            "fortune_teller.developer.normalize.umbrella_cli.build_normalize_model",
            build_mock,
        )

        runner = CliRunner()
        result = runner.invoke(app, [])  # default --deck all

        assert result.exit_code == 0
        normalize_mock.assert_called_once()
        synthesize_mock.assert_called_once()

    def test_deck_rw(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``--deck rw`` should call only ``normalize_deck``."""
        (tmp_path / "raw" / "rider-waite").mkdir(parents=True)
        monkeypatch.setattr(
            "fortune_teller.developer.normalize.umbrella_cli.settings.ft_data_dir",
            tmp_path,
        )

        normalize_mock = MagicMock(return_value=NormalizeOutcome())
        monkeypatch.setattr(
            "fortune_teller.developer.normalize.umbrella_cli.normalize_deck",
            normalize_mock,
        )
        synthesize_mock = MagicMock(return_value=NormalizeOutcome())
        monkeypatch.setattr(
            "fortune_teller.developer.normalize.umbrella_cli.synthesize_deck_synergies",
            synthesize_mock,
        )
        build_mock = MagicMock(return_value="fake-llm")
        monkeypatch.setattr(
            "fortune_teller.developer.normalize.umbrella_cli.build_normalize_model",
            build_mock,
        )

        runner = CliRunner()
        result = runner.invoke(app, ["--deck", "rw"])

        assert result.exit_code == 0
        normalize_mock.assert_called_once()
        synthesize_mock.assert_not_called()

    def test_deck_thoth(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``--deck thoth`` should call only ``synthesize_deck_synergies``."""
        (tmp_path / "parsed" / "book-of-thoth").mkdir(parents=True)
        monkeypatch.setattr(
            "fortune_teller.developer.normalize.umbrella_cli.settings.ft_data_dir",
            tmp_path,
        )

        normalize_mock = MagicMock(return_value=NormalizeOutcome())
        monkeypatch.setattr(
            "fortune_teller.developer.normalize.umbrella_cli.normalize_deck",
            normalize_mock,
        )
        synthesize_mock = MagicMock(return_value=NormalizeOutcome())
        monkeypatch.setattr(
            "fortune_teller.developer.normalize.umbrella_cli.synthesize_deck_synergies",
            synthesize_mock,
        )
        build_mock = MagicMock(return_value="fake-llm")
        monkeypatch.setattr(
            "fortune_teller.developer.normalize.umbrella_cli.build_normalize_model",
            build_mock,
        )

        runner = CliRunner()
        result = runner.invoke(app, ["--deck", "thoth"])

        assert result.exit_code == 0
        normalize_mock.assert_not_called()
        synthesize_mock.assert_called_once()

    # ------------------------------------------------------------------
    # --no-llm flag
    # ------------------------------------------------------------------

    def test_no_llm_flag(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``--no-llm`` should pass ``llm=None`` to both normalizers and
        skip calling ``build_normalize_model``.
        """
        (tmp_path / "raw" / "rider-waite").mkdir(parents=True)
        (tmp_path / "parsed" / "book-of-thoth").mkdir(parents=True)
        monkeypatch.setattr(
            "fortune_teller.developer.normalize.umbrella_cli.settings.ft_data_dir",
            tmp_path,
        )

        normalize_mock = MagicMock(return_value=NormalizeOutcome())
        monkeypatch.setattr(
            "fortune_teller.developer.normalize.umbrella_cli.normalize_deck",
            normalize_mock,
        )
        synthesize_mock = MagicMock(return_value=NormalizeOutcome())
        monkeypatch.setattr(
            "fortune_teller.developer.normalize.umbrella_cli.synthesize_deck_synergies",
            synthesize_mock,
        )
        build_mock = MagicMock()
        monkeypatch.setattr(
            "fortune_teller.developer.normalize.umbrella_cli.build_normalize_model",
            build_mock,
        )

        runner = CliRunner()
        result = runner.invoke(app, ["--no-llm"])

        assert result.exit_code == 0
        normalize_mock.assert_called_once()
        synthesize_mock.assert_called_once()
        # Both normalizers should receive llm=None
        assert normalize_mock.call_args.kwargs.get("llm") is None
        assert synthesize_mock.call_args.kwargs.get("llm") is None
        build_mock.assert_not_called()

    # ------------------------------------------------------------------
    # Missing directory errors
    # ------------------------------------------------------------------

    def test_missing_rw_dir(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When the RW raw directory is missing and ``--deck rw``, exit
        with code 1.
        """
        monkeypatch.setattr(
            "fortune_teller.developer.normalize.umbrella_cli.settings.ft_data_dir",
            tmp_path,
        )
        # Intentionally *not* creating raw/rider-waite

        runner = CliRunner()
        result = runner.invoke(app, ["--deck", "rw"])

        assert result.exit_code == 1

    def test_missing_thoth_dir(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When the Thoth parsed directory is missing and ``--deck thoth``,
        exit with code 1.
        """
        monkeypatch.setattr(
            "fortune_teller.developer.normalize.umbrella_cli.settings.ft_data_dir",
            tmp_path,
        )
        # Intentionally *not* creating parsed/book-of-thoth

        runner = CliRunner()
        result = runner.invoke(app, ["--deck", "thoth"])

        assert result.exit_code == 1


@pytest.mark.unit
class TestUmbrellaFailureReporting:
    """Both decks run to completion; the umbrella exits 1 if either had failures."""

    def _invoke(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        rw: NormalizeOutcome,
        thoth: NormalizeOutcome,
    ):
        (tmp_path / "raw" / "rider-waite").mkdir(parents=True)
        (tmp_path / "parsed" / "book-of-thoth").mkdir(parents=True)
        monkeypatch.setattr(
            "fortune_teller.developer.normalize.umbrella_cli.settings.ft_data_dir", tmp_path
        )
        rw_mock = MagicMock(return_value=rw)
        thoth_mock = MagicMock(return_value=thoth)
        monkeypatch.setattr(
            "fortune_teller.developer.normalize.umbrella_cli.normalize_deck", rw_mock
        )
        monkeypatch.setattr(
            "fortune_teller.developer.normalize.umbrella_cli.synthesize_deck_synergies", thoth_mock
        )
        monkeypatch.setattr(
            "fortune_teller.developer.normalize.umbrella_cli.build_normalize_model",
            MagicMock(return_value="fake-llm"),
        )
        result = CliRunner().invoke(app, [])
        return result, rw_mock, thoth_mock

    def test_rw_failure_still_runs_thoth_then_exits_1(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result, rw_mock, thoth_mock = self._invoke(
            tmp_path,
            monkeypatch,
            NormalizeOutcome(failed=[("the-fool", "boom")]),
            NormalizeOutcome(),
        )

        assert result.exit_code == 1
        rw_mock.assert_called_once()
        thoth_mock.assert_called_once()
        assert "--deck rw --provider api --model claude-sonnet-4-6 --only the-fool" in result.output

    def test_thoth_failure_exits_1(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        result, _, _ = self._invoke(
            tmp_path, monkeypatch, NormalizeOutcome(), NormalizeOutcome(failed=[("death", "boom")])
        )

        assert result.exit_code == 1
        assert "--deck thoth --provider api --model claude-sonnet-4-6 --only death" in result.output

    def test_no_failures_exit_0(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        result, _, _ = self._invoke(tmp_path, monkeypatch, NormalizeOutcome(), NormalizeOutcome())

        assert result.exit_code == 0

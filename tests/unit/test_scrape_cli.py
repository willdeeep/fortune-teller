"""Unit tests for the ``ft-scrape`` CLI source selection.

Uses ``--dry-run`` so no network or filesystem writes occur — the seed
files on disk are read and their slugs listed. Verifies that the default
(no ``--source``) scrapes *all* configured sources.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from fortune_teller.developer.scrape.cli import app

runner = CliRunner()


@pytest.mark.unit
class TestScrapeSourceSelection:
    def test_default_scrapes_all_sources(self) -> None:
        """Bare `ft-scrape --dry-run` processes both Thoth and Rider-Waite seeds."""
        result = runner.invoke(app, ["--dry-run"])
        assert result.exit_code == 0
        assert "book_of_thoth.txt" in result.output
        assert "rider_waite.txt" in result.output

    def test_single_source_limits_to_one(self) -> None:
        """`--source learntarot` processes only the Rider-Waite seeds."""
        result = runner.invoke(app, ["--source", "learntarot", "--dry-run"])
        assert result.exit_code == 0
        assert "rider_waite.txt" in result.output
        assert "book_of_thoth.txt" not in result.output

    def test_unknown_source_errors(self) -> None:
        result = runner.invoke(app, ["--source", "bogus", "--dry-run"])
        assert result.exit_code != 0
        assert "Unknown source" in result.output

    def test_seeds_with_all_is_rejected(self) -> None:
        """`--seeds` is ambiguous across multiple sources, so it requires one."""
        result = runner.invoke(app, ["--seeds", "whatever.txt", "--dry-run"])
        assert result.exit_code != 0
        assert "single --source" in result.output

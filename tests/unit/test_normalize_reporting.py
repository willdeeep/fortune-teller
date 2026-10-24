"""Unit tests for :mod:`fortune_teller.developer.normalize.reporting`."""

from __future__ import annotations

import pytest
from pydantic import HttpUrl
from rich.console import Console

from fortune_teller.application.models.domain import Arcana, Card
from fortune_teller.developer.normalize.reporting import report_outcome, retry_command
from fortune_teller.developer.normalize.rider_waite import (
    CardProvenance,
    NormalizeOutcome,
)


def _result(card_id: str) -> tuple[Card, CardProvenance]:
    card = Card(
        id=card_id, name=card_id, arcana=Arcana.MAJOR, source_url=HttpUrl("https://example.com/")
    )
    return card, CardProvenance(card_id=card_id, sections={})


def _console() -> Console:
    return Console(record=True, width=200, color_system=None)


@pytest.mark.unit
class TestRetryCommand:
    def test_repeats_provider_and_model(self) -> None:
        cmd = retry_command("uv run ft-normalize-rw", provider="local", model="m1", no_llm=False)
        assert cmd == "uv run ft-normalize-rw --provider local --model m1"

    def test_no_llm_replaces_provider_flags(self) -> None:
        cmd = retry_command("uv run ft-normalize-thoth", provider="api", model="m1", no_llm=True)
        assert cmd == "uv run ft-normalize-thoth --no-llm"


@pytest.mark.unit
class TestReportOutcome:
    def test_success_returns_false_and_prints_count(self) -> None:
        console = _console()
        outcome = NormalizeOutcome(results=[_result("a"), _result("b")])

        failed = report_outcome(console, outcome, action="Normalized", retry="uv run x")

        assert failed is False
        assert "Normalized 2 cards." in console.export_text()

    def test_failure_returns_true_lists_errors_and_retry(self) -> None:
        console = _console()
        outcome = NormalizeOutcome(
            results=[_result("a")], failed=[("b", "bad [thing]"), ("c", "worse")]
        )

        failed = report_outcome(console, outcome, action="Normalized", retry="uv run x")

        text = console.export_text()
        assert failed is True
        assert "Normalized 1 cards; 2 failed:" in text
        assert "b: bad [thing]" in text
        assert "c: worse" in text
        assert "uv run x --only b,c" in text

    def test_retry_command_not_wrapped_on_narrow_console(self) -> None:
        console = Console(record=True, width=40, color_system=None)
        retry = "uv run ft-normalize --deck thoth --provider api --model claude-sonnet-4-6"
        outcome = NormalizeOutcome(failed=[("death", "boom")])

        report_outcome(console, outcome, action="Synthesized", retry=retry)

        assert f"{retry} --only death" in console.export_text()

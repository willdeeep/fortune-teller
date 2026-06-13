"""Gradio UI — entry point for the ``fortune-teller`` console script.

The :func:`build_app` factory builds a :class:`gr.Blocks` app around an
injected :class:`~fortune_teller.application.services.reading.ReadingService`
and optional :class:`~fortune_teller.application.services.reading.HistoryStore`
so tests can drive the UI logic without spinning up a real Gradio server.
:func:`main` is the console-script entry point that wires up the service from
:class:`~fortune_teller.application.config.settings` and calls
:meth:`gr.Blocks.launch`.

The handler is a *generator*: each ``yield`` emits a new snapshot of
the UI state, so Gradio's streaming protocol populates the three card
panels left-to-right (one per deal) and the summary box only after
all three cards have been interpreted.

Run via::

    uv run fortune-teller
"""

from __future__ import annotations

import atexit
from collections.abc import Iterator
from typing import TYPE_CHECKING, cast
from uuid import UUID

import gradio as gr

if TYPE_CHECKING:
    from fortune_teller.application.models.domain import ReadingListItem
    from fortune_teller.application.services.reading import HistoryStore, ReadingService


def _format_card_text(card_name: str, orientation: str, text: str) -> str:
    """Render a single card panel as a 3-line block.

    Orientation arrow prefixes the line so it stays visible in the
    narrow Gradio textbox.
    """
    arrow = "▼" if orientation == "reversed" else "▲"
    label = "REVERSED" if orientation == "reversed" else "UPRIGHT"
    return f"{card_name}\n{arrow} {label}\n\n{text}"


def _format_reading_detail(reading_id: str, history_store: HistoryStore) -> str:
    """Render a full reading as a formatted string for the detail panel.

    Called when a user selects a row from the history Dataframe.
    Returns an empty string if the reading is not found.
    """
    reading = history_store.get(UUID(reading_id))
    if reading is None:
        return ""
    lines: list[str] = []
    lines.append(f"**Deck:** {reading.deck_id}")
    lines.append(f"**Spread:** {reading.spread_id}")
    lines.append(f"**Date:** {reading.created_at.strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("")
    for interp in reading.per_card:
        lines.append(
            _format_card_text(
                card_name=interp.card_name,
                orientation=interp.dealt.orientation.value,
                text=interp.text,
            )
        )
        lines.append("")
    lines.append(f"**Summary:** {reading.summary}")
    return "\n".join(lines)


def _load_history_list(history_store: HistoryStore) -> list[list[str]]:
    """Return history rows as a list of ``[date, spread, cards, summary]`` strings."""
    items: list[ReadingListItem] = history_store.list_recent()
    rows: list[list[str]] = []
    for item in items:
        date_str = item.created_at.strftime("%Y-%m-%d %H:%M")
        cards_str = ", ".join(item.card_names)
        rows.append([str(item.id), date_str, item.spread_id, cards_str, item.summary_preview])
    return rows


def run_reading_generator(
    reading_service: ReadingService,
) -> Iterator[tuple[str, ...]]:
    """Run one full reading, yielding a UI snapshot after each card.

    The Gradio ``.click()`` handler in :func:`build_app` consumes this
    generator one yield at a time so each card panel populates
    immediately. Exposed as a free function (not a closure) so tests
    can drive it without instantiating a Gradio Blocks widget.

    Yields:
        Tuples of ``(panel_0, panel_1, …, panel_N, summary)`` strings.
        The summary slot is empty until all cards have been dealt,
        then receives the finalised :attr:`Reading.summary` string.
    """
    handle = reading_service.start()
    positions = reading_service._spread.positions
    panels = [""] * len(positions)
    summary = ""

    for i, _pos in enumerate(positions):
        interp = reading_service.deal_next(handle)
        panels[i] = _format_card_text(
            card_name=interp.card_name,
            orientation=interp.dealt.orientation.value,
            text=interp.text,
        )
        yield (*panels, summary)

    reading = reading_service.finalize(handle)
    summary = reading.summary
    yield (*panels, summary)


def build_app(
    reading_service: ReadingService,
    history_store: HistoryStore | None = None,
) -> gr.Blocks:
    """Build the Gradio Blocks app bound to *reading_service*.

    *reading_service* is injected (not constructed inside) so tests
    can substitute a stub service without touching :mod:`config` or
    the on-disk data pipeline.

    *history_store* is optional; when provided the app includes a
    History tab for browsing past readings.
    """
    spread = reading_service._spread
    spread_name = spread.name
    positions = spread.positions

    with gr.Blocks(title="Fortune Teller") as demo:
        gr.Markdown(f"# Fortune Teller\n### {spread_name} · Book of Thoth")

        with gr.Tabs():
            with gr.Tab("Reading"):
                new_reading_btn = gr.Button("New Reading", variant="primary")

                with gr.Row():
                    card_panels = [
                        gr.Textbox(label=pos.name, lines=8, interactive=False) for pos in positions
                    ]

                summary_box = gr.Textbox(
                    label="Reading Summary",
                    lines=6,
                    interactive=False,
                )

                def run_reading() -> Iterator[tuple[str, ...]]:
                    """Gradio click handler — forwards to :func:`run_reading_generator`."""
                    yield from run_reading_generator(reading_service)

                new_reading_btn.click(
                    fn=run_reading,
                    inputs=[],
                    outputs=[*card_panels, summary_box],
                )

            with gr.Tab("History"):
                if history_store is not None:
                    _build_history_tab(history_store)
                else:
                    gr.Markdown("*No history store configured.*")

    return cast(gr.Blocks, demo)


def _build_history_tab(history_store: HistoryStore) -> None:
    """Construct the History tab UI inside the current Gradio Blocks context.

    Must be called inside a ``with gr.Tab("History"):`` block.
    """
    refresh_btn = gr.Button("Refresh", variant="secondary")

    history_df = gr.Dataframe(
        headers=["ID", "Date", "Spread", "Cards", "Summary"],
        label="Past Readings",
        interactive=False,
        datatype=["str", "str", "str", "str", "str"],
    )

    detail_box = gr.Textbox(
        label="Reading Detail",
        lines=16,
        interactive=False,
    )

    def refresh_history() -> list[list[str]]:
        return _load_history_list(history_store)

    refresh_btn.click(
        fn=refresh_history,
        inputs=[],
        outputs=[history_df],
    )

    def on_select(evt: gr.SelectData) -> str:
        return _format_reading_detail(evt.row[0], history_store)

    history_df.select(
        fn=on_select,
        inputs=[],
        outputs=[detail_box],
    )

    history_df.value = refresh_history()


def main() -> None:
    """Console-script entry point: build and launch the Gradio app."""
    from fortune_teller.application.config import settings  # noqa: PLC0415
    from fortune_teller.application.services.reading import (  # noqa: PLC0415
        build_reading_service,
    )
    from fortune_teller.application.stores.sqlite import SQLiteStore  # noqa: PLC0415

    history_store = SQLiteStore(settings.sqlite_path)
    history_store.open()
    atexit.register(history_store.close)

    service = build_reading_service(settings, history_store=history_store)
    demo = build_app(service, history_store=history_store)
    demo.launch(server_name="127.0.0.1", server_port=7860)


if __name__ == "__main__":
    main()

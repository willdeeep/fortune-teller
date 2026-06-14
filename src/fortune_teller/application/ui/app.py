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
from pathlib import Path
from typing import TYPE_CHECKING, cast
from uuid import UUID

import gradio as gr

from fortune_teller.application.stores.images import image_path_for

if TYPE_CHECKING:
    from fortune_teller.application.models.domain import Card, ReadingListItem
    from fortune_teller.application.services.reading import HistoryStore, ReadingService


def _format_card_text(
    card_name: str,
    orientation: str,
    text: str,
    position_name: str | None = None,
    position_meaning: str | None = None,
) -> str:
    """Render a single card panel as a formatted block.

    Orientation arrow prefixes the line so it stays visible in the
    narrow Gradio textbox. When *position_name* and *position_meaning*
    are provided, they are appended as a subtitle.
    """
    arrow = "▼" if orientation == "reversed" else "▲"
    label = "REVERSED" if orientation == "reversed" else "UPRIGHT"
    header = f"{card_name}\n{arrow} {label}"
    if position_name and position_meaning:
        header += f"\n*{position_name}: {position_meaning}*"
    return f"{header}\n\n{text}"


def _format_card_detail(
    card: Card,
    image_path: str | None = None,
) -> str:
    """Render a full card detail view as Markdown.

    Includes the card image (if available), all structured sections,
    and a source-attribution link. Degrades gracefully when sections
    or the image are absent.
    """
    lines: list[str] = []

    if image_path:
        lines.append(f"![{card.name}]({image_path})")
        lines.append("")

    lines.append(f"## {card.name}")

    if card.arcana == "major":
        lines.append(
            f"*Major Arcana*{'  ·  ' + str(card.number) if card.number is not None else ''}"
        )
    else:
        suit_label = card.suit.value.title() if card.suit else "Minor Arcana"
        lines.append(
            f"*{suit_label}*{'  ·  ' + str(card.number) if card.number is not None else ''}"
        )

    lines.append("")

    if card.sections:
        for section in card.sections:
            section_label = section.section.value.replace("_", " ").title()
            lines.append(f"**{section_label}:** {section.text}")
            lines.append("")
    else:
        lines.append("*No structured data available for this card.*")
        lines.append("")

    source_url = str(card.source_url)
    lines.append(f"[View source ↗]({source_url})")

    return "\n".join(lines)


def _format_position_info(
    position_name: str,
    position_meaning: str,
    source_url: str,
) -> str:
    """Render position meaning with a source link as a Markdown string."""
    return f"**{position_name}:** {position_meaning}  \n[Source ↗]({source_url})"


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
    images_dir: Path | None = None,
) -> Iterator[tuple[str | None, ...]]:
    """Run one full reading, yielding a UI snapshot after each card.

    The Gradio ``.click()`` handler in :func:`build_app` consumes this
    generator one yield at a time so each card panel populates
    immediately. Exposed as a free function (not a closure) so tests
    can drive it without instantiating a Gradio Blocks widget.

    Args:
        reading_service: The service that orchestrates readings.
        images_dir: Optional directory containing card artwork.  When
            provided, each dealt card's image path is resolved via
            :func:`~fortune_teller.application.stores.images.image_path_for`
            and included in the yield.

    Yields:
        Tuples of ``(img_0, …, img_N, panel_0, …, panel_N, summary)``.
        Image slots are ``None`` when the card has not yet been dealt or
        when no image file exists.  Text panel slots are ``""`` until
        the corresponding card is dealt.  The summary slot is ``""``
        until all cards have been interpreted, then receives the
        finalised :attr:`Reading.summary` string.
    """
    handle = reading_service.start()
    positions = reading_service._spread.positions
    n = len(positions)
    panels: list[str] = [""] * n
    images: list[str | None] = [None] * n
    summary = ""

    for i, pos in enumerate(positions):
        interp = reading_service.deal_next(handle)
        panels[i] = _format_card_text(
            card_name=interp.card_name,
            orientation=interp.dealt.orientation.value,
            text=interp.text,
            position_name=pos.name,
            position_meaning=pos.meaning,
        )
        if images_dir is not None:
            img_path = image_path_for(interp.dealt.card_id, images_dir)
            images[i] = str(img_path) if img_path is not None else None
        yield (*images, *panels, summary)

    reading = reading_service.finalize(handle)
    yield (*images, *panels, reading.summary)


def build_app(
    reading_service: ReadingService,
    history_store: HistoryStore | None = None,
    images_dir: Path | None = None,
) -> gr.Blocks:
    """Build the Gradio Blocks app bound to *reading_service*.

    *reading_service* is injected (not constructed inside) so tests
    can substitute a stub service without touching :mod:`config` or
    the on-disk data pipeline.

    *history_store* is optional; when provided the app includes a
    History tab for browsing past readings.

    *images_dir* is optional; when provided, card artwork is resolved
    from this directory and displayed above each panel.
    """
    spread = reading_service._spread
    spread_name = spread.name
    positions = spread.positions
    deck = reading_service._deck
    cards_by_id: dict[str, Card] = {c.id: c for c in deck.cards}

    def _show_card_detail_for_position(position_index: int, dealt_ids: list[str]) -> str:
        """Render the detail panel for the card at *position_index*."""
        if position_index >= len(dealt_ids) or not dealt_ids[position_index]:
            return "*No card dealt in this position yet. Click 'New Reading' first.*"
        card_id = dealt_ids[position_index]
        card = cards_by_id.get(card_id)
        if card is None:
            return f"*Card not found: {card_id}*"
        img_path: str | None = None
        if images_dir is not None:
            resolved = image_path_for(card_id, images_dir)
            img_path = str(resolved) if resolved is not None else None
        return _format_card_detail(card, image_path=img_path)

    # Default detail text shown before first reading
    _default_detail = "*Click a detail button to see card information.*"

    # Static position-meaning Markdown (set once at build time)
    _position_meanings = "  \n".join(
        _format_position_info(pos.name, pos.meaning, str(pos.source_url)) for pos in positions
    )

    with gr.Blocks(title="Fortune Teller") as demo:
        gr.Markdown(f"# Fortune Teller\n### {spread_name} · Book of Thoth")

        with gr.Tabs():
            with gr.Tab("Reading"):
                new_reading_btn = gr.Button("New Reading", variant="primary")

                with gr.Row():
                    card_images = [
                        gr.Image(label=pos.name, show_label=True, height=200, interactive=False)
                        for pos in positions
                    ]

                with gr.Row():
                    card_panels = [
                        gr.Textbox(label=pos.name, lines=8, interactive=False) for pos in positions
                    ]

                # Position info — always visible
                gr.Markdown(value=_position_meanings)

                # Card detail buttons — one per position
                with gr.Row():
                    detail_btns = [
                        gr.Button(f"📋 {pos.name} detail", size="sm", variant="secondary")
                        for pos in positions
                    ]

                # Card detail panel
                card_detail = gr.Markdown(value=_default_detail)

                summary_box = gr.Textbox(
                    label="Reading Summary",
                    lines=6,
                    interactive=False,
                )

                # Track dealt card IDs so detail buttons know which card to show
                dealt_card_ids = gr.State(value=[])

                def run_reading() -> Iterator[tuple[str | None, ...]]:
                    """Gradio click handler — forwards to :func:`run_reading_generator`."""
                    yield from run_reading_generator(reading_service, images_dir=images_dir)

                new_reading_btn.click(
                    fn=run_reading,
                    inputs=[],
                    outputs=[*card_images, *card_panels, summary_box],
                )

                # Wire each detail button to show that position's dealt card
                for i, btn in enumerate(detail_btns):
                    btn.click(
                        fn=lambda dealt_ids, idx=i: _show_card_detail_for_position(idx, dealt_ids),
                        inputs=[dealt_card_ids],
                        outputs=[card_detail],
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
    deck_images_dir = settings.images_dir / service.deck_id
    demo = build_app(
        service,
        history_store=history_store,
        images_dir=deck_images_dir,
    )
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        allowed_paths=[str(deck_images_dir)],
    )


if __name__ == "__main__":
    main()

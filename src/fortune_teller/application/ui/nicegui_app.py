"""NiceGUI UI — entry point for the ``fortune-teller`` console script.

The :func:`build_app` factory wires up the NiceGUI app around an injected
:class:`~fortune_teller.application.services.reading.ReadingService` and optional
:class:`~fortune_teller.application.services.reading.HistoryStore` so tests can
drive the UI logic without spinning up a real server.

:func:`main` is the console-script entry point that constructs the service from
:class:`~fortune_teller.application.config.settings` and calls :func:`ui.run`.

The reading handler uses ``asyncio.to_thread`` for blocking LLM calls so the
NiceGUI event loop stays responsive.  Card detail views use ``ui.dialog``
for the modal overlay (plan 0024).
"""

from __future__ import annotations

import asyncio
import atexit
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

from nicegui import Client, app, ui

from fortune_teller.application.stores.images import image_path_for

if TYPE_CHECKING:
    from fortune_teller.application.models.domain import Card, ReadingListItem, SpreadPosition
    from fortune_teller.application.services.reading import HistoryStore, ReadingService


# ---------------------------------------------------------------------------
# Module-level service references (set by build_app)
# ---------------------------------------------------------------------------

_service: ReadingService | None = None
_history_store: HistoryStore | None = None
_images_dir: Path | None = None
_cards_by_id: dict[str, Card] = {}


# ---------------------------------------------------------------------------
# Framework-agnostic formatters (unchanged from Gradio version)
# ---------------------------------------------------------------------------


def _format_card_text(
    card_name: str,
    orientation: str,
    text: str,
    position_name: str | None = None,
    position_meaning: str | None = None,
) -> str:
    """Render a single card panel as a formatted block.

    Orientation arrow prefixes the line so it stays visible. When
    *position_name* and *position_meaning* are provided, they are appended
    as a subtitle.
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
    and a source-attribution link.  Degrades gracefully when sections
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

    Called when a user selects a row from the history table.
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
    """Return history rows as a list of ``[id, date, spread, cards, summary]`` strings."""
    items: list[ReadingListItem] = history_store.list_recent()
    rows: list[list[str]] = []
    for item in items:
        date_str = item.created_at.strftime("%Y-%m-%d %H:%M")
        cards_str = ", ".join(item.card_names)
        rows.append([str(item.id), date_str, item.spread_id, cards_str, item.summary_preview])
    return rows


# ---------------------------------------------------------------------------
# Streaming generator (kept for backwards compatibility)
# ---------------------------------------------------------------------------


def run_reading_generator(
    reading_service: ReadingService,
    images_dir: Path | None = None,
) -> Iterator[tuple[str | None, ...]]:
    """Run one full reading, yielding a UI snapshot after each card.

    Each yield produces a tuple of ``(img_0, …, img_N, panel_0, …, panel_N, summary)``.
    Image slots are ``None`` when not yet dealt or when no image file exists.
    Text panel slots are ``""`` until the corresponding card is dealt.
    The summary slot is ``""`` until all cards are interpreted.
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


# ---------------------------------------------------------------------------
# Image URL resolution
# ---------------------------------------------------------------------------


def _image_url(card_id: str) -> str | None:
    """Resolve a card ID to a static-URL path for its artwork image.

    Returns ``None`` when ``_images_dir`` is unset or no image file exists.
    The URL is relative to the ``/images`` static-file mount registered
    by :func:`build_app`.
    """
    if _images_dir is None:
        return None
    path = image_path_for(card_id, _images_dir)
    if path is None:
        return None
    return f"/images/{path.name}"


# ---------------------------------------------------------------------------
# UI construction
# ---------------------------------------------------------------------------


def build_app(
    reading_service: ReadingService,
    history_store: HistoryStore | None = None,
    images_dir: Path | None = None,
) -> None:
    """Wire up dependencies and register static-file mounts.

    *reading_service* is injected (not constructed inside) so tests can
    substitute a stub service without touching :mod:`config` or the
    on-disk data pipeline.

    *history_store* is optional; when provided the app includes a
    History section for browsing past readings.

    *images_dir* is optional; when provided, card artwork is resolved
    from this directory and displayed above each panel.
    """
    global _service, _history_store, _images_dir, _cards_by_id  # noqa: PLW0603
    _service = reading_service
    _history_store = history_store
    _images_dir = images_dir
    _cards_by_id = {c.id: c for c in reading_service._deck.cards}

    if images_dir is not None and images_dir.is_dir():
        app.add_static_files("/images", str(images_dir))

    # Register the index page here (not as a module-level decorator) so that
    # the route is (re)created against whatever NiceGUI app is current — this is
    # what lets the ``nicegui.testing`` ``User`` simulation, which resets the
    # global app between tests, re-register the page on each run.  Idempotent:
    # repeated ``build_app`` calls within one process register the route once.
    if reading_page not in Client.page_routes:
        ui.page("/")(reading_page)


async def reading_page() -> None:
    """Build the reading + history UI per client connection."""
    if _service is None:
        ui.label("Error: service not configured. Call build_app() first.")
        return

    spread = _service._spread
    deck = _service._deck
    positions = spread.positions
    n = len(positions)

    ui.markdown(f"# Fortune Teller\n### {spread.name} · {deck.name}")
    position_meanings_md = "  \n".join(
        _format_position_info(pos.name, pos.meaning, str(pos.source_url)) for pos in positions
    )

    state: dict[str, list[str] | None] = {"dealt_ids": []}
    position_labels: list[ui.label] = []
    card_images: list[ui.image] = []
    card_texts: list[ui.markdown] = []

    _build_card_panels(positions, position_labels, card_images, card_texts)
    ui.markdown(position_meanings_md)

    detail_dialog, detail_content = _build_detail_dialog()
    _build_detail_buttons(positions, state, detail_content, detail_dialog)

    summary_md = ui.markdown()
    new_reading_btn = ui.button("New Reading", color="primary")

    async def do_reading() -> None:
        await _run_reading(
            _service, positions, n, state, card_images, card_texts, summary_md, new_reading_btn
        )

    new_reading_btn.on_click(do_reading)

    if _history_store is not None:
        _build_history_section(_history_store)


def _build_card_panels(
    positions: list[SpreadPosition],
    position_labels: list[ui.label],
    card_images: list[ui.image],
    card_texts: list[ui.markdown],
) -> None:
    """Construct the card image + text panels for each spread position."""
    with ui.row().classes("w-full justify-center"):
        for pos in positions:
            with ui.card().classes("w-1/3 min-w-[200px]"):
                position_labels.append(ui.label(pos.name).classes("text-h6"))
                img = ui.image().classes("hidden")
                card_images.append(img)
                txt = ui.markdown().classes("hidden")
                card_texts.append(txt)


def _build_detail_dialog() -> tuple[ui.dialog, ui.markdown]:
    """Construct the detail modal dialog and return (dialog, content)."""
    with ui.dialog() as dialog, ui.card().classes("w-96"):
        content = ui.markdown()
        ui.button("Close", on_click=dialog.close)
    return dialog, content


def _build_detail_buttons(
    positions: list[SpreadPosition],
    state: dict[str, list[str] | None],
    detail_content: ui.markdown,
    detail_dialog: ui.dialog,
) -> None:
    """Build one detail button per spread position."""
    with ui.row():
        for i, pos in enumerate(positions):
            ui.button(
                f"📋 {pos.name}",
                on_click=lambda idx=i: _show_detail(idx, state, detail_content, detail_dialog),
            )


async def _run_reading(
    service: ReadingService,
    positions: list[SpreadPosition],
    n: int,
    state: dict[str, list[str] | None],
    card_images: list[ui.image],
    card_texts: list[ui.markdown],
    summary_md: ui.markdown,
    new_reading_btn: ui.button,
) -> None:
    """Execute a full reading sequence with progressive UI updates."""
    new_reading_btn.disable()
    state["dealt_ids"] = []

    for i in range(n):
        card_images[i].set_source("").classes("hidden")
        card_texts[i].set_content("").classes("hidden")
    summary_md.set_content("")

    try:
        handle = await asyncio.to_thread(service.start)

        for i, pos in enumerate(positions):
            interp = await asyncio.to_thread(service.deal_next, handle)

            panel = _format_card_text(
                card_name=interp.card_name,
                orientation=interp.dealt.orientation.value,
                text=interp.text,
                position_name=pos.name,
                position_meaning=pos.meaning,
            )
            card_texts[i].set_content(panel).classes(remove="hidden")

            url = _image_url(interp.dealt.card_id)
            if url is not None:
                card_images[i].set_source(url).classes(remove="hidden")
            else:
                card_images[i].set_source("").classes("hidden")

            current = state.get("dealt_ids")
            if current is not None:
                current.append(interp.dealt.card_id)

        reading = await asyncio.to_thread(service.finalize, handle)
        summary_md.set_content(f"**Summary:** {reading.summary}")
    finally:
        new_reading_btn.enable()


def _show_detail(
    position_index: int,
    state: dict[str, list[str] | None],
    content: ui.markdown,
    dialog: ui.dialog,
) -> None:
    """Open the detail dialog for the card at *position_index*."""
    dealt_ids = state.get("dealt_ids") or []
    if position_index >= len(dealt_ids) or not dealt_ids[position_index]:
        content.set_content("*No card dealt in this position yet. Click 'New Reading' first.*")
    else:
        card_id = dealt_ids[position_index]
        card = _cards_by_id.get(card_id)
        if card is None:
            content.set_content(f"*Card not found: {card_id}*")
        else:
            img_url = _image_url(card_id)
            content.set_content(_format_card_detail(card, image_path=img_url))
    dialog.open()


def _build_history_section(history_store: HistoryStore) -> None:
    """Construct the history table + detail section."""
    ui.separator()
    ui.markdown("### History")

    columns = [
        {"name": "id", "label": "ID", "field": "id"},
        {"name": "date", "label": "Date", "field": "date"},
        {"name": "spread", "label": "Spread", "field": "spread"},
        {"name": "cards", "label": "Cards", "field": "cards"},
        {"name": "summary", "label": "Summary", "field": "summary"},
    ]

    rows = _history_rows(history_store)

    table = ui.table(
        columns=columns,
        rows=rows,
        row_key="id",
        selection="single",
        pagination={"rowsPerPage": 10},
    )

    detail_box = ui.markdown()

    def on_selection() -> None:
        selected = table.selected
        if selected:
            reading_id = selected[0]["id"]
            detail_box.set_content(_format_reading_detail(reading_id, history_store))

    table.on("selection", on_selection)

    def refresh() -> None:
        table.update_rows(_history_rows(history_store))

    ui.button("Refresh", on_click=refresh)


def _history_rows(history_store: HistoryStore) -> list[dict[str, str]]:
    """Convert history list items to NiceGUI table row dicts."""
    return [
        {
            "id": row[0],
            "date": row[1],
            "spread": row[2],
            "cards": row[3],
            "summary": row[4],
        }
        for row in _load_history_list(history_store)
    ]


# ---------------------------------------------------------------------------
# Console-script entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Console-script entry point: build and launch the NiceGUI app."""
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

    build_app(service, history_store=history_store, images_dir=deck_images_dir)

    ui.run(
        host="127.0.0.1",
        port=7860,
        show=False,
        title="Fortune Teller",
    )


if __name__ == "__main__":
    main()

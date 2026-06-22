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

Plan 0030 adds:
- Spread selector (``ui.select``) backed by :func:`list_spreads`.
- CSS-grid renderer for 2D spread layouts (e.g. Celtic Cross).
- Per-position ``transform:rotate()`` for the crossing card.
"""

from __future__ import annotations

import asyncio
import atexit
from collections.abc import Callable
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
_spread_options: list[tuple[str, str]] = []
_service_factory: Callable[[str], ReadingService] | None = None
_service_cache: dict[str, ReadingService] = {}


# ---------------------------------------------------------------------------
# Framework-agnostic formatters
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
# Layout helpers (framework-agnostic, unit-testable)
# ---------------------------------------------------------------------------


def _has_grid_layout(positions: list[SpreadPosition]) -> bool:
    """Return True when all positions have ``row`` and ``col`` set."""
    return all(p.row is not None and p.col is not None for p in positions)


def _grid_dimensions(positions: list[SpreadPosition]) -> tuple[int, int]:
    """Return ``(rows, cols)`` needed for a grid layout.

    Assumes all positions have ``row``/``col`` set (call after :func:`_has_grid_layout`).
    """
    max_row = max(p.row for p in positions if p.row is not None)
    max_col = max(p.col for p in positions if p.col is not None)
    return (max_row + 1, max_col + 1)


def _resolve_service(spread_id: str) -> ReadingService:
    """Return the service for *spread_id*, using cache or factory.

    Falls back to the default ``_service`` when no factory is configured.
    """
    if spread_id in _service_cache:
        return _service_cache[spread_id]
    if _service is not None and _service._spread.id == spread_id:
        _service_cache[spread_id] = _service
        return _service
    if _service_factory is not None:
        svc = _service_factory(spread_id)
        _service_cache[spread_id] = svc
        return svc
    if _service is not None:
        return _service
    raise RuntimeError("No service configured.")


# ---------------------------------------------------------------------------
# UI construction
# ---------------------------------------------------------------------------


def build_app(
    reading_service: ReadingService,
    history_store: HistoryStore | None = None,
    images_dir: Path | None = None,
    spread_options: list[tuple[str, str]] | None = None,
    service_factory: Callable[[str], ReadingService] | None = None,
) -> None:
    """Wire up dependencies and register static-file mounts.

    *reading_service* is injected (not constructed inside) so tests can
    substitute a stub service without touching :mod:`config` or the
    on-disk data pipeline.

    *history_store* is optional; when provided the app includes a
    History section for browsing past readings.

    *images_dir* is optional; when provided, card artwork is resolved
    from this directory and displayed above each panel.

    *spread_options* is an optional list of ``(spread_id, display_name)``
    tuples. When provided, a spread selector is shown so the user can
    choose between spreads (e.g. New Moon vs Celtic Cross).

    *service_factory* is an optional callable that takes a ``spread_id``
    and returns a :class:`ReadingService` for that spread. Used when
    *spread_options* is provided and the selected spread differs from
    the default service's spread.
    """
    global _service, _history_store, _images_dir, _cards_by_id  # noqa: PLW0603
    global _spread_options, _service_factory, _service_cache  # noqa: PLW0603
    _service = reading_service
    _history_store = history_store
    _images_dir = images_dir
    _cards_by_id = {c.id: c for c in reading_service._deck.cards}
    _spread_options = spread_options or []
    _service_factory = service_factory
    _service_cache = {}

    if images_dir is not None and images_dir.is_dir():
        app.add_static_files("/images", str(images_dir))

    if reading_page not in Client.page_routes:
        ui.page("/")(reading_page)


_GRID_CARD_STYLE = (
    "border:1px solid #888;border-radius:6px;padding:4px;"
    "display:flex;flex-direction:column;align-items:center;"
    "min-height:120px;max-width:90px;cursor:pointer;"
)


async def reading_page() -> None:
    """Build the reading + history UI per client connection."""
    if _service is None:
        ui.label("Error: service not configured. Call build_app() first.")
        return

    spread = _service._spread
    deck = _service._deck

    title_md = ui.markdown(f"# Fortune Teller\n### {spread.name} · {deck.name}")

    spread_select: ui.select | None = None
    if _spread_options:
        opts = {sid: name for sid, name in _spread_options}
        spread_select = ui.select(
            options=opts,
            value=spread.id,
            label="Spread",
        )

    state: dict[str, list[str] | None] = {"dealt_ids": []}
    detail_dialog, detail_content = _build_detail_dialog()

    card_images: list[ui.image] = []
    card_texts: list[ui.markdown] = []

    card_container = ui.column().classes("w-full")

    def rebuild_card_panels() -> None:
        """Clear and rebuild card panels for the current spread."""
        card_images.clear()
        card_texts.clear()
        state["dealt_ids"] = []
        card_container.clear()
        svc = _resolve_current_service(spread_select)
        positions = svc._spread.positions
        with card_container:
            _build_card_layout(
                positions, state, detail_content, detail_dialog, card_images, card_texts
            )
            position_meanings_md = "  \n".join(
                _format_position_info(pos.name, pos.meaning, str(pos.source_url))
                for pos in positions
            )
            ui.markdown(position_meanings_md)

    rebuild_card_panels()

    if spread_select is not None:

        def on_spread_change() -> None:
            svc = _resolve_current_service(spread_select)
            current_spread = svc._spread
            title_md.set_content(f"# Fortune Teller\n### {current_spread.name} · {svc._deck.name}")
            rebuild_card_panels()

        # ``on_value_change`` fires whenever the selected value changes (user
        # pick or programmatic), which is what we want and is reliably driven by
        # the test harness; the raw "update:model-value" client event is not.
        spread_select.on_value_change(lambda _e: on_spread_change())

    summary_md = ui.markdown()
    new_reading_btn = ui.button("New Reading", color="primary")

    async def do_reading() -> None:
        svc = _resolve_current_service(spread_select)
        positions = svc._spread.positions
        n = len(positions)
        await _run_reading(
            svc, positions, n, state, card_images, card_texts, summary_md, new_reading_btn
        )

    new_reading_btn.on_click(do_reading)

    if _history_store is not None:
        _build_history_section(_history_store)


def _resolve_current_service(spread_select: ui.select | None) -> ReadingService:
    """Return the service for the currently selected spread."""
    if _service is None:
        raise RuntimeError("No service configured.")
    if spread_select is not None and spread_select.value is not None:
        return _resolve_service(str(spread_select.value))
    return _service


def _build_card_layout(
    positions: list[SpreadPosition],
    state: dict[str, list[str] | None],
    detail_content: ui.markdown,
    detail_dialog: ui.dialog,
    card_images: list[ui.image],
    card_texts: list[ui.markdown],
) -> None:
    """Build card panels using grid or row layout based on position hints."""
    if _has_grid_layout(positions):
        _build_card_grid(positions, state, detail_content, detail_dialog, card_images, card_texts)
    else:
        _build_card_row(positions, state, detail_content, detail_dialog, card_images, card_texts)


def _build_card_grid(
    positions: list[SpreadPosition],
    state: dict[str, list[str] | None],
    detail_content: ui.markdown,
    detail_dialog: ui.dialog,
    card_images: list[ui.image],
    card_texts: list[ui.markdown],
) -> None:
    """Render cards in a CSS grid layout for 2D spreads (e.g. Celtic Cross)."""
    rows, cols = _grid_dimensions(positions)
    grid = ui.element("div").style(
        f"display:grid;"
        f"grid-template-columns:repeat({cols},100px);"
        f"grid-template-rows:repeat({rows},150px);"
        f"gap:8px;justify-content:center;"
    )
    with grid:
        for i, pos in enumerate(positions):
            rot = f"transform:rotate({pos.rotation}deg);" if pos.rotation else ""
            row_str = str((pos.row or 0) + 1)
            col_str = str((pos.col or 0) + 1)
            cell = ui.element("div").style(
                f"grid-row:{row_str};grid-column:{col_str};{_GRID_CARD_STYLE}{rot}"
            )
            with cell:
                ui.label(pos.name).classes("text-xs font-bold text-center")
                img = ui.image().classes("hidden").style("max-width:70px;max-height:100px;")
                txt = ui.markdown().classes("hidden text-xs text-center")
                card_images.append(img)
                card_texts.append(txt)

            cell.on(
                "click",
                lambda idx=i: _show_detail(idx, state, detail_content, detail_dialog),
            )


def _build_card_row(
    positions: list[SpreadPosition],
    state: dict[str, list[str] | None],
    detail_content: ui.markdown,
    detail_dialog: ui.dialog,
    card_images: list[ui.image],
    card_texts: list[ui.markdown],
) -> None:
    """Render cards in a simple row layout (fallback for spreads without grid hints)."""
    with ui.row().classes("w-full justify-center"):
        for i, pos in enumerate(positions):
            with ui.card().classes("w-1/3 min-w-[200px]"):
                ui.label(pos.name).classes("text-h6")
                img = ui.image().classes("hidden")
                txt = ui.markdown().classes("hidden")
                card_images.append(img)
                card_texts.append(txt)
                ui.button(
                    f"📋 {pos.name}",
                    on_click=lambda idx=i: _show_detail(idx, state, detail_content, detail_dialog),
                )


def _build_detail_dialog() -> tuple[ui.dialog, ui.markdown]:
    """Construct the detail modal dialog and return (dialog, content)."""
    with ui.dialog() as dialog, ui.card().classes("w-96"):
        content = ui.markdown()
        ui.button("Close", on_click=dialog.close)
    return dialog, content


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
    from fortune_teller.application.services.loading import list_spreads  # noqa: PLC0415
    from fortune_teller.application.services.reading import (  # noqa: PLC0415
        build_reading_service,
    )
    from fortune_teller.application.stores.sqlite import SQLiteStore  # noqa: PLC0415

    history_store = SQLiteStore(settings.sqlite_path)
    history_store.open()
    atexit.register(history_store.close)

    service = build_reading_service(settings, history_store=history_store)
    deck_images_dir = settings.images_dir / service.deck_id

    parsed_dir = settings.ft_data_dir / "parsed"
    spread_options = list_spreads(parsed_dir) if parsed_dir.is_dir() else None

    def factory(spread_id: str) -> ReadingService:
        return build_reading_service(settings, spread_id=spread_id, history_store=history_store)

    build_app(
        service,
        history_store=history_store,
        images_dir=deck_images_dir,
        spread_options=spread_options,
        service_factory=factory,
    )

    ui.run(
        host="127.0.0.1",
        port=7860,
        show=False,
        title="Fortune Teller",
    )


if __name__ == "__main__":
    main()

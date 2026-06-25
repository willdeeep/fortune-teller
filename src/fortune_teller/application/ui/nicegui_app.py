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

Plan 0036 supersedes the 0030 grid/row split: *every* spread now renders
through one :func:`_build_spread_layout` — a spatial grid of fixed-size card
boxes (CSS card backs that flip to face images on deal) placed by an "effective
grid" (:func:`_effective_grid`), with the interpretation **text** in a numbered
list below the grid (:func:`_format_list_item`) so overlapping cards (the 90°
crossing card sharing a cell) never hide it.

Plan 0030 adds:
- Spread selector (``ui.select``) backed by :func:`list_spreads`.
- 2D spread layouts (e.g. Celtic Cross) with ``row``/``col``/``rotation`` hints.
- Per-position ``transform:rotate()`` for the crossing card.

Plan 0024 adds:
- Position-title → meaning popover (click a position title to open a
  ``ui.dialog`` with ``SpreadPosition.meaning`` + source link via
  :func:`_format_position_info`).
- Reinforcing/opposing synergy references in the card-detail dialog
  (rendered in :func:`_format_card_detail` when *cards_by_id* is supplied).

Plan 0025 adds:
- Reversed cards display their artwork rotated 180° via CSS
  ``transform:rotate(180deg)`` on the image element (see
  :func:`rotation_style`), applied in :func:`_run_reading`.

Plan 0023 adds:
- Deck selector (``ui.select``) backed by :func:`list_decks` so the user
  can choose which deck a reading uses.  Services are cached per
  ``(deck_id, spread_id)`` pair.  Images resolve from the parent
  ``images_dir`` using the current deck ID.
"""

from __future__ import annotations

import asyncio
import atexit
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

from nicegui import Client, app, ui

from fortune_teller.application.models.domain import Orientation
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
_current_deck_id: str | None = None
_deck_options: list[tuple[str, str]] = []
_spread_options: list[tuple[str, str]] = []
_service_factory: Callable[[str, str], ReadingService] | None = None
_service_cache: dict[tuple[str, str], ReadingService] = {}


# ---------------------------------------------------------------------------
# Card-box geometry constants
# ---------------------------------------------------------------------------

# Canonical card-box geometry (portrait ~2:3). Tunable via the screenshot harness.
_CARD_W = 90
_CARD_H = 135
# A 90°-rotated box overhangs its cell by (H - W)/2 each side; the column gap must
# clear that overhang so a crossing card never touches its horizontal neighbours.
_COLUMN_GAP = (_CARD_H - _CARD_W) // 2 + 10  # == 32
_ROW_GAP = 10

# Generic CSS card back (no image asset; works for every deck). Per-deck art deferred.
_CARD_BACK_STYLE = (
    "flex:1;width:100%;"
    "background:"
    "radial-gradient(circle at 50% 50%, rgba(150,160,220,.35) 0 3px, transparent 4px),"
    "repeating-linear-gradient(45deg, rgba(110,120,180,.30) 0 5px, rgba(80,90,150,.30) 5px 10px);"
)
# Face image fills the box, bounded so artwork can never overflow.
_CARD_FACE_STYLE = "flex:1;width:100%;height:100%;object-fit:contain;"


# ---------------------------------------------------------------------------
# Framework-agnostic helpers
# ---------------------------------------------------------------------------


def rotation_style(orientation: Orientation) -> str:
    """Return a CSS transform string to rotate reversed card art 180°.

    Returns an empty string for upright cards so no transform is applied.
    """
    return "transform: rotate(180deg);" if orientation is Orientation.REVERSED else ""


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


def _format_list_item(
    number: int,
    position_name: str,
    card_name: str,
    orientation: str,
    text: str,
) -> str:
    """Render one numbered interpretation-list entry as Markdown.

    The list beneath the spatial grid is the source of truth for interpretation
    text, so this carries the full per-card body. Format:
    ``**N. Position** · Card ▲ UPRIGHT`` then a blank line then the body.
    """
    arrow = "▼" if orientation == "reversed" else "▲"
    label = "REVERSED" if orientation == "reversed" else "UPRIGHT"
    return f"**{number}. {position_name}** · {card_name} {arrow} {label}\n\n{text}"


def _format_card_detail(
    card: Card,
    image_path: str | None = None,
    cards_by_id: dict[str, Card] | None = None,
) -> str:
    """Render a full card detail view as Markdown.

    Includes the card image (if available), all structured sections,
    reinforcing/opposing references (when *cards_by_id* is provided),
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

    # Reinforcing / opposing references (v0.6.0 synergy fields).
    # Resolved to display names only when *cards_by_id* is provided so the
    # pure formatter stays deterministic and unit-testable in isolation.
    if cards_by_id is not None:
        if card.reinforcing_ids:
            names = [cards_by_id[rid].name for rid in card.reinforcing_ids if rid in cards_by_id]
            if names:
                lines.append(f"**Reinforcing:** {', '.join(names)}")
                lines.append("")
        if card.opposing_ids:
            names = [cards_by_id[oid].name for oid in card.opposing_ids if oid in cards_by_id]
            if names:
                lines.append(f"**Opposing:** {', '.join(names)}")
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
    by :func:`build_app`.  When a deck is selected the path includes the
    deck subdirectory (``/images/<deck_id>/<file>``).
    """
    if _images_dir is None or _current_deck_id is None:
        return None
    deck_dir = _images_dir / _current_deck_id
    path = image_path_for(card_id, deck_dir)
    if path is None:
        return None
    return f"/images/{_current_deck_id}/{path.name}"


# ---------------------------------------------------------------------------
# Layout helpers (framework-agnostic, unit-testable)
# ---------------------------------------------------------------------------


def _has_grid_layout(positions: list[SpreadPosition]) -> bool:
    """Return True when all positions have ``row`` and ``col`` set."""
    return all(p.row is not None and p.col is not None for p in positions)


def _grid_dimensions(positions: list[SpreadPosition]) -> tuple[int, int]:
    """Return ``(rows, cols)`` needed for a grid layout.

    Assumes all positions have ``row``/``col`` set (call after :func:`_has_grid_layout`).

    Retained as a tested pure helper; the renderer itself sizes the grid via
    :func:`_effective_dimensions` over :func:`_effective_grid` coords (plan 0036).
    """
    max_row = max(p.row for p in positions if p.row is not None)
    max_col = max(p.col for p in positions if p.col is not None)
    return (max_row + 1, max_col + 1)


def _effective_grid(positions: list[SpreadPosition]) -> list[tuple[int, int]]:
    """Return the ``(row, col)`` cell for each position.

    Grid spreads (every position has ``row``/``col``) map to their own
    coordinates; linear spreads (no coordinates) flow left→right as
    ``row=0, col=index`` so they render through the same grid renderer.
    """
    if _has_grid_layout(positions):
        return [(p.row or 0, p.col or 0) for p in positions]
    return [(0, i) for i in range(len(positions))]


def _effective_dimensions(coords: list[tuple[int, int]]) -> tuple[int, int]:
    """Return ``(rows, cols)`` spanning the effective-grid *coords*."""
    rows = max((r for r, _ in coords), default=0) + 1
    cols = max((c for _, c in coords), default=0) + 1
    return rows, cols


def _resolve_service(deck_id: str, spread_id: str) -> ReadingService:
    """Return the service for ``(deck_id, spread_id)``, using cache or factory.

    Falls back to the default ``_service`` when no factory is configured.
    """
    key = (deck_id, spread_id)
    if key in _service_cache:
        return _service_cache[key]
    if _service is not None and _service._spread.id == spread_id and _service.deck_id == deck_id:
        _service_cache[key] = _service
        return _service
    if _service_factory is not None:
        svc = _service_factory(deck_id, spread_id)
        _service_cache[key] = svc
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
    deck_options: list[tuple[str, str]] | None = None,
    service_factory: Callable[[str, str], ReadingService] | None = None,
) -> None:
    """Wire up dependencies and register static-file mounts.

    *reading_service* is injected (not constructed inside) so tests can
    substitute a stub service without touching :mod:`config` or the
    on-disk data pipeline.

    *history_store* is optional; when provided the app includes a
    History section for browsing past readings.

    *images_dir* is optional; when provided, card artwork is resolved
    from this directory and displayed above each panel.  Should be the
    **parent** images directory (e.g. ``settings.images_dir``); the
    current deck ID is appended at resolve time.

    *spread_options* is an optional list of ``(spread_id, display_name)``
    tuples. When provided, a spread selector is shown so the user can
    choose between spreads (e.g. New Moon vs Celtic Cross).

    *deck_options* is an optional list of ``(deck_id, display_name)``
    tuples. When provided, a deck selector is shown so the user can
    choose which deck a reading uses.

    *service_factory* is an optional callable that takes
    ``(deck_id, spread_id)`` and returns a :class:`ReadingService` for
    that combination. Used when *spread_options* or *deck_options* is
    provided and the selected combination differs from the default
    service.
    """
    global _service, _history_store, _images_dir, _cards_by_id  # noqa: PLW0603
    global _spread_options, _service_factory, _service_cache  # noqa: PLW0603
    global _deck_options, _current_deck_id  # noqa: PLW0603
    _service = reading_service
    _history_store = history_store
    _images_dir = images_dir
    _cards_by_id = {c.id: c for c in reading_service._deck.cards}
    _current_deck_id = reading_service.deck_id
    _spread_options = spread_options or []
    _deck_options = deck_options or []
    _service_factory = service_factory
    _service_cache = {}

    if images_dir is not None and images_dir.is_dir():
        app.add_static_files("/images", str(images_dir))

    if reading_page not in Client.page_routes:
        ui.page("/")(reading_page)

    def _on_unhandled_exception(exc: Exception) -> None:
        ui.notify(f"Unexpected error: {exc}", type="negative")

    app.on_exception(_on_unhandled_exception)


def _grid_container_style(rows: int, cols: int) -> str:
    """CSS for the spread grid: fixed card-sized tracks + the overhang-clearing gap."""
    return (
        "display:grid;"
        f"grid-template-columns:repeat({cols},{_CARD_W}px);"
        f"grid-template-rows:repeat({rows},{_CARD_H}px);"
        f"column-gap:{_COLUMN_GAP}px;row-gap:{_ROW_GAP}px;"
        "justify-content:center;align-items:center;"
    )


def _card_box_style(row: int, col: int, z: int, rotation: int) -> str:
    """CSS for one card box: cell placement, centring, stacking, and rotation.

    *row*/*col* are 0-based effective-grid coordinates (CSS grid lines are
    1-based, hence the ``+ 1``). Overlapping positions share a cell and are
    centred via ``place-self`` with *z* controlling stack order; *rotation*
    (e.g. 90° for the crossing card) rotates the whole box.
    """
    rot = f"transform:rotate({rotation}deg);" if rotation else ""
    return (
        f"grid-row:{row + 1};grid-column:{col + 1};place-self:center;z-index:{z};"
        f"width:{_CARD_W}px;height:{_CARD_H}px;"
        "border:1px solid #888;border-radius:8px;overflow:hidden;"
        "display:flex;flex-direction:column;cursor:pointer;"
        "box-shadow:0 1px 4px rgba(0,0,0,.3);"
        f"{rot}"
    )


async def reading_page() -> None:  # noqa: PLR0915
    """Build the reading + history UI per client connection."""
    if _service is None:
        ui.label("Error: service not configured. Call build_app() first.")
        return

    spread = _service._spread
    deck = _service._deck

    title_md = ui.markdown(f"# Fortune Teller\n### {spread.name} · {deck.name}")

    deck_select: ui.select | None = None
    if _deck_options:
        deck_opts = {did: name for did, name in _deck_options}
        deck_select = ui.select(
            options=deck_opts,
            value=deck.id,
            label="Deck",
        )

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
    position_dialog, position_content = _build_position_dialog()

    card_images: list[ui.image] = []
    card_backs: list[ui.element] = []
    card_items: list[ui.markdown] = []

    card_container = ui.column().classes("w-full items-center")

    def rebuild_card_panels() -> None:
        """Clear and rebuild the spread layout for the current spread/deck."""
        card_images.clear()
        card_backs.clear()
        card_items.clear()
        state["dealt_ids"] = []
        card_container.clear()
        svc = _resolve_current_service(deck_select, spread_select)
        positions = svc._spread.positions
        with card_container:
            _build_spread_layout(
                positions,
                state,
                detail_content,
                detail_dialog,
                position_content,
                position_dialog,
                card_images,
                card_backs,
                card_items,
            )

    rebuild_card_panels()

    def on_selection_change() -> None:
        """Handle deck or spread selection change."""
        global _current_deck_id, _cards_by_id  # noqa: PLW0603
        svc = _resolve_current_service(deck_select, spread_select)
        _current_deck_id = svc.deck_id
        _cards_by_id = {c.id: c for c in svc._deck.cards}
        current_spread = svc._spread
        title_md.set_content(f"# Fortune Teller\n### {current_spread.name} · {svc._deck.name}")
        rebuild_card_panels()

    if deck_select is not None:
        deck_select.on_value_change(lambda _e: on_selection_change())

    if spread_select is not None:
        # ``on_value_change`` fires whenever the selected value changes (user
        # pick or programmatic), which is what we want and is reliably driven by
        # the test harness; the raw "update:model-value" client event is not.
        spread_select.on_value_change(lambda _e: on_selection_change())

    summary_md = ui.markdown()
    new_reading_btn = ui.button("New Reading", color="primary")

    async def do_reading() -> None:
        svc = _resolve_current_service(deck_select, spread_select)
        positions = svc._spread.positions
        n = len(positions)
        await _run_reading(
            svc,
            positions,
            n,
            state,
            card_images,
            card_backs,
            card_items,
            summary_md,
            new_reading_btn,
        )

    new_reading_btn.on_click(do_reading)

    if _history_store is not None:
        _build_history_section(_history_store)


def _resolve_current_service(
    deck_select: ui.select | None,
    spread_select: ui.select | None,
) -> ReadingService:
    """Return the service for the currently selected deck and spread."""
    if _service is None:
        raise RuntimeError("No service configured.")
    deck_id = (
        str(deck_select.value)
        if deck_select is not None and deck_select.value is not None
        else _service.deck_id
    )
    spread_id = (
        str(spread_select.value)
        if spread_select is not None and spread_select.value is not None
        else _service._spread.id
    )
    return _resolve_service(deck_id, spread_id)


def _build_card_box(
    pos: SpreadPosition,
    index: int,
    row: int,
    col: int,
    state: dict[str, list[str] | None],
    detail_content: ui.markdown,
    detail_dialog: ui.dialog,
    position_content: ui.markdown,
    position_dialog: ui.dialog,
    card_images: list[ui.image],
    card_backs: list[ui.element],
) -> None:
    """Build one fixed-size card box: name label, CSS back, hidden face image.

    The box opens the card-detail dialog on click; the name label opens the
    position-meaning dialog (``stopPropagation`` keeps the two handlers apart).
    The face image starts hidden and bounded by ``object-fit:contain`` so deal
    just unhides it; the back starts visible so the spread's shape shows before
    any card is dealt.
    """
    box = (
        ui.element("div")
        .classes("ft-card-box")
        .style(_card_box_style(row, col, index, pos.rotation))
    )
    with box:
        title = (
            ui.label(pos.name)
            .classes("text-xs font-bold text-center")
            .style("padding:2px;cursor:pointer;")
        )
        title.on(
            "click",
            lambda _, p=pos: _show_position_meaning(p, position_content, position_dialog),
            js_handler="(e) => { e.stopPropagation(); emit(e); }",
        )
        back = ui.element("div").classes("ft-card-back").style(_CARD_BACK_STYLE)
        img = ui.image().classes("ft-card-face hidden").style(_CARD_FACE_STYLE)
        card_images.append(img)
        card_backs.append(back)

    box.on(
        "click",
        lambda idx=index: _show_detail(idx, state, detail_content, detail_dialog),
    )


def _build_spread_layout(
    positions: list[SpreadPosition],
    state: dict[str, list[str] | None],
    detail_content: ui.markdown,
    detail_dialog: ui.dialog,
    position_content: ui.markdown,
    position_dialog: ui.dialog,
    card_images: list[ui.image],
    card_backs: list[ui.element],
    card_items: list[ui.markdown],
) -> None:
    """Render any spread: a spatial grid of card boxes + a numbered list below.

    Every spread uses this one path. Card boxes are placed by the effective grid
    (grid spreads keep their coords; linear spreads flow row 0, col=index).
    Positions sharing a cell overlap (centred, stacked, rotated). The numbered
    list beneath the grid is the source of truth for interpretation text; each
    item has a ``Details · <name>`` button opening the card-detail dialog.
    """
    coords = _effective_grid(positions)
    rows, cols = _effective_dimensions(coords)
    grid = ui.element("div").style(_grid_container_style(rows, cols))
    with grid:
        for i, (pos, (row, col)) in enumerate(zip(positions, coords, strict=True)):
            _build_card_box(
                pos,
                i,
                row,
                col,
                state,
                detail_content,
                detail_dialog,
                position_content,
                position_dialog,
                card_images,
                card_backs,
            )

    with ui.column().classes("w-full max-w-2xl gap-1"):
        for i, pos in enumerate(positions):
            with ui.row().classes("w-full items-start gap-2 ft-list-row"):
                item = ui.markdown().classes("ft-list-item grow")
                card_items.append(item)
                ui.button(
                    f"Details · {pos.name}",
                    on_click=lambda idx=i: _show_detail(idx, state, detail_content, detail_dialog),
                ).props("flat dense")


def _build_detail_dialog() -> tuple[ui.dialog, ui.markdown]:
    """Construct the detail modal dialog and return (dialog, content)."""
    with ui.dialog() as dialog, ui.card().classes("w-96"):
        content = ui.markdown()
        ui.button("Close", on_click=dialog.close)
    return dialog, content


def _build_position_dialog() -> tuple[ui.dialog, ui.markdown]:
    """Construct the position-meaning modal dialog and return (dialog, content)."""
    with ui.dialog() as dialog, ui.card().classes("w-72"):
        content = ui.markdown()
        ui.button("Close", on_click=dialog.close)
    return dialog, content


def _show_position_meaning(
    position: SpreadPosition,
    content: ui.markdown,
    dialog: ui.dialog,
) -> None:
    """Open the position-meaning dialog for *position*."""
    content.set_content(
        _format_position_info(position.name, position.meaning, str(position.source_url))
    )
    dialog.open()


async def _run_reading(
    service: ReadingService,
    positions: list[SpreadPosition],
    n: int,
    state: dict[str, list[str] | None],
    card_images: list[ui.image],
    card_backs: list[ui.element],
    card_items: list[ui.markdown],
    summary_md: ui.markdown,
    new_reading_btn: ui.button,
) -> None:
    """Execute a reading: per card, flip its box back→face and fill its list item."""
    new_reading_btn.disable()
    state["dealt_ids"] = []

    # Reset to the all-backs, empty-list state.
    for i in range(n):
        card_images[i].set_source("").classes(add="hidden")
        card_images[i].style.pop("transform", None)
        card_backs[i].classes(remove="hidden")
        card_items[i].set_content("")
    summary_md.set_content("")

    try:
        handle = await asyncio.to_thread(service.start)

        for i, pos in enumerate(positions):
            interp = await asyncio.to_thread(service.deal_next, handle)

            card_items[i].set_content(
                _format_list_item(
                    i + 1,
                    pos.name,
                    interp.card_name,
                    interp.dealt.orientation.value,
                    interp.text,
                )
            )

            # Flip back→face; the face shows even without an image asset
            # (blank/bordered), so the deal is always visible.
            card_backs[i].classes(add="hidden")
            url = _image_url(interp.dealt.card_id)
            card_images[i].set_source(url or "").classes(remove="hidden")

            # Reversed cards rotate the face image 180° (plan 0025).
            rot = rotation_style(interp.dealt.orientation)
            if rot:
                card_images[i].style(rot)
            else:
                card_images[i].style.pop("transform", None)

            current = state.get("dealt_ids")
            if current is not None:
                current.append(interp.dealt.card_id)

        reading = await asyncio.to_thread(service.finalize, handle)
        summary_md.set_content(f"**Summary:** {reading.summary}")
    except Exception as exc:
        summary_md.set_content(f"⚠️ Reading failed: {exc}")
        ui.notify("Reading failed — see summary area for details.", type="negative")
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
            content.set_content(
                _format_card_detail(card, image_path=img_url, cards_by_id=_cards_by_id)
            )
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
    from fortune_teller.application.services.loading import (  # noqa: PLC0415
        list_decks,
        list_spreads,
    )
    from fortune_teller.application.services.reading import (  # noqa: PLC0415
        build_reading_service,
    )
    from fortune_teller.application.stores.sqlite import SQLiteStore  # noqa: PLC0415

    history_store = SQLiteStore(settings.sqlite_path)
    history_store.open()
    atexit.register(history_store.close)

    service = build_reading_service(settings, history_store=history_store)

    parsed_dir = settings.ft_data_dir / "parsed"
    spread_options = list_spreads(parsed_dir) if parsed_dir.is_dir() else None
    deck_options = list_decks(parsed_dir) if parsed_dir.is_dir() else None

    def factory(deck_id: str, spread_id: str) -> ReadingService:
        return build_reading_service(
            settings,
            deck_id=deck_id,
            spread_id=spread_id,
            history_store=history_store,
        )

    build_app(
        service,
        history_store=history_store,
        images_dir=settings.images_dir,
        spread_options=spread_options,
        deck_options=deck_options,
        service_factory=factory,
    )

    ui.run(
        host="127.0.0.1",
        port=7860,
        show=False,
        reload=False,
        title="Fortune Teller",
    )


if __name__ in {"__main__", "__mp_main__"}:
    main()

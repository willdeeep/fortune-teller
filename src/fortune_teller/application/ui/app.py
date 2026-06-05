"""Gradio UI — entry point for the ``fortune-teller`` console script.

The :func:`build_app` factory builds a :class:`gr.Blocks` app around an
injected :class:`~fortune_teller.application.services.reading.ReadingService`
so tests can drive the UI logic without spinning up a real Gradio
server. :func:`main` is the console-script entry point that wires up
the service from :class:`~fortune_teller.application.config.settings`
and calls :meth:`gr.Blocks.launch`.

The handler is a *generator*: each ``yield`` emits a new snapshot of
the UI state, so Gradio's streaming protocol populates the three card
panels left-to-right (one per deal) and the summary box only after
all three cards have been interpreted.

Run via::

    uv run fortune-teller
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, cast

import gradio as gr

if TYPE_CHECKING:
    from fortune_teller.application.services.reading import ReadingService


def _format_card_text(card_name: str, orientation: str, text: str) -> str:
    """Render a single card panel as a 3-line block.

    Orientation arrow prefixes the line so it stays visible in the
    narrow Gradio textbox.
    """
    arrow = "▼" if orientation == "reversed" else "▲"
    label = "REVERSED" if orientation == "reversed" else "UPRIGHT"
    return f"{card_name}\n{arrow} {label}\n\n{text}"


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


def build_app(reading_service: ReadingService) -> gr.Blocks:
    """Build the Gradio Blocks app bound to *reading_service*.

    *reading_service* is injected (not constructed inside) so tests
    can substitute a stub service without touching :mod:`config` or
    the on-disk data pipeline.
    """
    spread = reading_service._spread
    spread_name = spread.name
    positions = spread.positions

    with gr.Blocks(title="Fortune Teller") as demo:
        gr.Markdown(f"# Fortune Teller\n### {spread_name} · Book of Thoth")

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

    return cast(gr.Blocks, demo)


def main() -> None:
    """Console-script entry point: build and launch the Gradio app."""
    # Lazy imports so importing this module doesn't load config or
    # heavy services; only the entry point pays the cost.
    from fortune_teller.application.config import settings
    from fortune_teller.application.services.reading import build_reading_service

    service = build_reading_service(settings)
    demo = build_app(service)
    demo.launch(server_name="127.0.0.1", server_port=7860)


if __name__ == "__main__":
    main()

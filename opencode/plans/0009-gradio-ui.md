# 0009 — Gradio UI (Spike)

Module: `fortune_teller.application.ui.app`

Entry point: `fortune-teller` → `fortune_teller.application.ui.app:main`

---

## Layout

```
┌──────────────────────────────────────────────────────────────────────────┐
│  Fortune Teller — New Moon Spread (Book of Thoth)                        │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  [ New Reading ]                                                         │
│                                                                          │
│  ┌────────── Past ──────────┐ ┌──── Present ────┐ ┌────── Future ──────┐│
│  │  Card Name               │ │  Card Name       │ │  Card Name         ││
│  │  Orientation (▲/▼)       │ │  Orientation     │ │  Orientation       ││
│  │                          │ │                  │ │                    ││
│  │  Brief interpretation    │ │  Brief interp.   │ │  Brief interp.     ││
│  │  text (3–5 sentences)    │ │                  │ │                    ││
│  └──────────────────────────┘ └──────────────────┘ └────────────────────┘│
│                                                                          │
│  ┌─────────────────────── Reading Summary ─────────────────────────────┐ │
│  │  Full summary text (4–8 sentences)                                  │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Implementation

```python
import gradio as gr
from fortune_teller.application.services.reading import ReadingService
from fortune_teller.application.config import settings


def build_app(reading_service: ReadingService) -> gr.Blocks:
    """
    Build the Gradio Blocks app.

    reading_service is injected so it can be replaced with a stub in tests.
    """

    with gr.Blocks(title="Fortune Teller") as demo:
        gr.Markdown("# Fortune Teller\n### New Moon Three-Card Spread · Book of Thoth")

        new_reading_btn = gr.Button("New Reading", variant="primary")

        with gr.Row():
            card_panels = [
                gr.Textbox(label=pos_name, lines=8, interactive=False)
                for pos_name in ["Past", "Present", "Future"]
            ]

        summary_box = gr.Textbox(
            label="Reading Summary",
            lines=6,
            interactive=False,
        )

        def run_reading():
            """
            Generator: yields progressive updates as each card is dealt.

            Yields:
                Tuple of (past_text, present_text, future_text, summary_text)
            """
            handle = reading_service.start()
            texts = ["", "", ""]
            summary = ""

            for i in range(3):
                interp = reading_service.deal_next(handle)
                orientation_symbol = "▼ REVERSED" if interp.dealt.orientation == "reversed" else "▲ UPRIGHT"
                texts[i] = f"{interp.card_name}\n{orientation_symbol}\n\n{interp.text}"
                yield (*texts, summary)

            reading = reading_service.finalize(handle)
            summary = reading.summary
            yield (*texts, summary)

        new_reading_btn.click(
            fn=run_reading,
            inputs=[],
            outputs=[*card_panels, summary_box],
        )

    return demo


def main() -> None:
    from fortune_teller.application.services.reading import build_reading_service

    service = build_reading_service(settings)
    demo = build_app(service)
    demo.launch(server_name="127.0.0.1", server_port=7860)
```

---

## Progressive Reveal

- The `run_reading` function is a generator (uses `yield`).
- Gradio's streaming support (`outputs` passed to `.click`) means each
  `yield` updates the UI immediately.
- Card panels fill left-to-right (Past → Present → Future).
- Summary box stays empty until all three cards are dealt, then populates
  with the final reading summary.
- While the reading is in progress, the "New Reading" button is naturally
  disabled by Gradio's event queue.

---

## Dependency Wiring (`build_reading_service`)

```python
def build_reading_service(settings: Settings) -> ReadingService:
    from fortune_teller.application.stores.embeddings import Embedder
    from fortune_teller.application.stores.vector import VectorStore
    from fortune_teller.application.chains.per_card import build_per_card_chain
    from fortune_teller.application.chains.summary import build_summary_chain
    from langchain_openai import ChatOpenAI

    embedder = Embedder()
    vector_store = VectorStore(settings.ft_data_dir / "duckdb/fortune.duckdb", embedder)
    llm = ChatOpenAI(
        base_url=settings.openai_base_url,
        api_key=settings.openai_api_key,
        model=settings.chat_model,
        temperature=0.0,
    )
    deck = _load_deck(settings)     # loads from parsed JSON or DuckDB
    spread = _load_spread(settings)  # loads New Moon spread

    return ReadingService(
        deck=deck,
        spread=spread,
        per_card_chain=build_per_card_chain(llm),
        summary_chain=build_summary_chain(llm),
        vector_store=vector_store,
    )
```

---

## Out of Scope (spike)

- Deck selector
- Spread selector
- Manual card entry
- Reading history panel
- Login / user profile

These are tracked in the roadmap (see README.md).

---

## Smoke Test

There is no automated Gradio test in the spike (Gradio UI testing requires
a running server). Acceptance is verified manually against criterion 1–5
of `0013-spike-acceptance.md`.

Future: add `pytest-playwright` or `gradio.testing` for E2E UI tests.

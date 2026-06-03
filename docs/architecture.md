# Architecture

## Component Overview

```
fortune-teller/
├── application/     Runtime: Gradio UI, LangChain chains, domain models,
│                    DuckDB vector store, SQLite history
└── developer/       Tooling: httpx scraper, selectolax parser,
                     sentence-transformers embedder, DuckDB index builder
```

## Build-Time Data Flow

```
thothreadings.com
      │  httpx AsyncClient (rate-limited 1 req/s, cached to disk)
      ▼
data/cache/thothreadings.com/<slug>.html
      │  selectolax parser (extracts 11 structured sections per card)
      ▼
data/parsed/book-of-thoth/<slug>.json   ← Card pydantic model JSON
data/parsed/spreads/<spread>.json       ← Spread pydantic model JSON
      │  HuggingFaceEmbeddings (BAAI/bge-small-en-v1.5, MPS on M2)
      ▼
data/embedded/<slug>.json               ← Chunks with embedding vectors
      │  DuckDB VSS (HNSW index, cosine metric)
      ▼
data/duckdb/fortune.duckdb
  └── chunks (id, type, card_id, section, text, embedding FLOAT[384])
       └── HNSW index
```

## Runtime Data Flow (New Reading)

```
User clicks "New Reading"
      │
      ▼
ReadingService.start(deck_id, spread_id)
      │  DeckSession initialised — all 78 cards available
      │
      ▼  × 3 (one per spread position)
DeckSession.deal_one(position_index)
      │  card drawn — removed from remaining pool
      │  orientation randomly assigned (50/50)
      │
      ▼
VectorStore.search_card_section(card_id, query, k=4)
VectorStore.search_spread_position(spread_id, position_index)
      │  top-k chunks retrieved (cosine similarity)
      │
      ▼
PerCardChain (ChatPromptTemplate → ChatOpenAI @ temperature=0)
      │  grounded 3-5 sentence interpretation
      │
      ▼
Gradio card panel updated (progressive reveal)
      │
      ▼  (after all 3 cards)
SummaryChain (all 3 interpretations + retrieved context bundled)
      │  4-8 sentence pattern-spotting summary
      │
      ▼
Gradio summary panel updated
```

## Key Design Decisions

| Decision | Rationale |
| -------- | --------- |
| DuckDB VSS | Lightweight, single-file, no server. HNSW cosine search handles ~860 vectors trivially. |
| sentence-transformers locally | No embedding server needed. BAAI/bge-small-en-v1.5 is 384-dim, fast on MPS, strong quality. |
| temperature=0 | Readings are grounded in retrieved text. The LLM surfaces patterns, not creative interpretations. |
| No-replace at DeckSession level | Invariant enforced in domain logic, not UI — survives future UI changes. |
| One chunk per card section | Enables precise retrieval (e.g. only the "reversed" section when a card is inverted). |

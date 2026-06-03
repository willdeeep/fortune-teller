# 0007 — Embeddings & Vector Store

Modules:
- `fortune_teller.application.stores.embeddings` — `Embedder` wrapper
- `fortune_teller.application.stores.vector` — `VectorStore` (DuckDB VSS)
- `fortune_teller.developer.embed.cli` — `ft-embed` entry point
- `fortune_teller.developer.build_index.cli` — `ft-build-index` entry point

---

## Embedding Model

```
Model:   BAAI/bge-small-en-v1.5
Dims:    384
Source:  HuggingFace Hub (downloaded on first run to ~/.cache/huggingface)
Device:  mps  (M2 Apple Silicon)  → auto-fallback to cpu
Norm:    normalize_embeddings=True
```

### `Embedder` wrapper

```python
from langchain_huggingface import HuggingFaceEmbeddings


class Embedder:
    """
    Thin wrapper around HuggingFaceEmbeddings for synchronous and
    batch async use.
    """

    MODEL_NAME = "BAAI/bge-small-en-v1.5"

    def __init__(self) -> None:
        self._model = HuggingFaceEmbeddings(
            model_name=self.MODEL_NAME,
            model_kwargs={"device": _detect_device()},
            encode_kwargs={"normalize_embeddings": True},
        )

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self._model.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._model.embed_query(text)


def _detect_device() -> str:
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"
```

---

## Chunking Strategy

One chunk per `CardSection` per card → ~78 × ~11 ≈ ~858 card chunks.
One chunk per `SpreadPosition` → 3 spread chunks (for the New Moon spread).

**Total at spike:** ~861 vectors × 384 dims — negligible for DuckDB VSS.

### Building chunks from parsed data

```python
def card_to_chunks(card: Card, deck_id: str) -> list[Chunk]:
    chunks = []
    for section in card.sections:
        chunks.append(Chunk(
            chunk_type=ChunkType.CARD_SECTION,
            deck_id=deck_id,
            card_id=card.id,
            card_name=card.name,
            section=section.section,
            source_url=str(card.source_url),
            text=section.text,
        ))
    return chunks


def spread_to_chunks(spread: Spread) -> list[Chunk]:
    return [
        Chunk(
            chunk_type=ChunkType.SPREAD_POSITION,
            spread_id=spread.id,
            position_index=pos.index,
            source_url=str(pos.source_url),
            text=f"{pos.name}: {pos.meaning}",
        )
        for pos in spread.positions
    ]
```

---

## DuckDB VSS Schema

```sql
-- Run once at index build time
INSTALL vss;
LOAD vss;

CREATE TABLE IF NOT EXISTS chunks (
    id               UUID PRIMARY KEY,
    chunk_type       VARCHAR NOT NULL,
    deck_id          VARCHAR,
    card_id          VARCHAR,
    card_name        VARCHAR,
    section          VARCHAR,
    spread_id        VARCHAR,
    position_index   INTEGER,
    source_url       VARCHAR NOT NULL,
    text             VARCHAR NOT NULL,
    embedding        FLOAT[384] NOT NULL
);

CREATE INDEX IF NOT EXISTS hnsw_chunks
    ON chunks USING HNSW (embedding)
    WITH (metric = 'cosine');
```

---

## `VectorStore` API

```python
import duckdb
from fortune_teller.application.models.domain import Chunk
from fortune_teller.application.stores.embeddings import Embedder


class VectorStore:
    """
    DuckDB-backed vector store using the VSS extension.

    All search methods return chunks sorted by cosine similarity (desc).
    """

    DIMENSIONS = 384

    def __init__(self, db_path: str | Path, embedder: Embedder) -> None:
        self._conn = duckdb.connect(str(db_path))
        self._embedder = embedder
        self._ensure_schema()

    def add(self, chunks: Iterable[Chunk]) -> None:
        """Embed and insert chunks. Idempotent on re-run (upsert by id)."""
        ...

    def search_card_section(
        self,
        query: str,
        *,
        card_id: str,
        k: int = 3,
    ) -> list[Chunk]:
        """Return top-k sections for a specific card, ranked by query similarity."""
        ...

    def search_spread_position(
        self,
        query: str,
        *,
        spread_id: str,
        position_index: int,
    ) -> list[Chunk]:
        """Return the spread position chunk(s) for context."""
        ...

    def search_global(self, query: str, *, k: int = 5) -> list[Chunk]:
        """Unrestricted similarity search across all chunks."""
        ...

    def close(self) -> None:
        self._conn.close()

    def _ensure_schema(self) -> None:
        self._conn.execute("INSTALL vss; LOAD vss;")
        # CREATE TABLE IF NOT EXISTS ...
        # CREATE INDEX IF NOT EXISTS ...
        ...
```

---

## CLI Pipeline

### `ft-embed`

Reads all `data/parsed/**/*.json` files, runs `Embedder.embed_texts()` in
batches of 64, writes augmented JSON files (with `embedding` field) to
`data/embedded/`.

### `ft-build-index`

Reads `data/embedded/` files, inserts rows into
`data/duckdb/fortune.duckdb` using `VectorStore.add()`.

---

## Integration Tests

```python
@pytest.mark.integration
def test_vector_store_round_trip(tmp_path: Path, sample_chunks: list[Chunk]) -> None:
    """Insert chunks, search, verify top result is correct card."""
    embedder = Embedder()
    store = VectorStore(tmp_path / "test.duckdb", embedder)
    store.add(sample_chunks)
    results = store.search_card_section("creative potential", card_id="the-fool", k=3)
    assert len(results) > 0
    assert all(r.card_id == "the-fool" for r in results)


@pytest.mark.integration
def test_vector_store_persistence(tmp_path: Path, sample_chunks: list[Chunk]) -> None:
    """Data survives closing and reopening the connection."""
    embedder = Embedder()
    path = tmp_path / "persist.duckdb"
    store = VectorStore(path, embedder)
    store.add(sample_chunks)
    store.close()
    store2 = VectorStore(path, embedder)
    results = store2.search_global("journey", k=5)
    assert len(results) > 0


@pytest.mark.integration
def test_spread_position_retrieval(tmp_path: Path, spread_chunks: list[Chunk]) -> None:
    embedder = Embedder()
    store = VectorStore(tmp_path / "test.duckdb", embedder)
    store.add(spread_chunks)
    results = store.search_spread_position(
        "new beginnings", spread_id="new-moon-three-card", position_index=0
    )
    assert len(results) >= 1
    assert results[0].position_index == 0
```

---

## Notes

- The `sentence-transformers` and `torch` packages will be downloaded by
  HuggingFace on first use. On M2, `torch` should pick up MPS automatically.
- DuckDB's `vss` extension is bundled with DuckDB >= 1.1 — no extra install.
- Index build is a one-time operation; the DuckDB file is a development
  artefact, not committed to git (listed in `.gitignore`).

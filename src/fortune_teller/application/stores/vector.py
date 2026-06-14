"""DuckDB VSS vector store adapter.

Single-file DuckDB database with the VSS extension loaded. Chunks are stored
in a ``chunks`` table and indexed with an HNSW index using cosine similarity.
This module hides all DuckDB-specific SQL behind a small Python API used by
the rest of the application.

Typical usage::

    store = VectorStore(Path("data/duckdb/fortune.duckdb"))
    with store:
        store.add_chunks(chunks)
        hits = store.search(query_embedding, k=4)

The store can be used in two modes:

1. **Persistent** — pass a file path; the database is opened from that file
   and the schema is created on first open. Reopening reuses the existing
   data.
2. **Ephemeral** — pass ``":memory:"`` as the path; useful for tests. The
   VSS extension must be loadable from the in-memory database.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import duckdb

from fortune_teller.application.models.domain import Chunk, SearchHit

if TYPE_CHECKING:
    from collections.abc import Sequence

#: Embedding dimensionality of the default model.
DEFAULT_EMBEDDING_DIMENSION = 384

#: DuckDB column type for embedding vectors. Sized at module load time
#: against the default; the schema is built with this literal so it can be
#: templated per-instance.
_EMBEDDING_TYPE_TEMPLATE = "FLOAT[{dimension}]"


def _build_schema(dimension: int) -> str:
    """Return the ``CREATE TABLE`` DDL for the chunks table."""
    emb_type = _EMBEDDING_TYPE_TEMPLATE.format(dimension=dimension)
    return f"""
        CREATE TABLE IF NOT EXISTS chunks (
            id             VARCHAR PRIMARY KEY,
            chunk_type     VARCHAR NOT NULL,
            deck_id        VARCHAR,
            card_id        VARCHAR,
            card_name      VARCHAR,
            section        VARCHAR,
            spread_id      VARCHAR,
            position_index INTEGER,
            source_url     VARCHAR NOT NULL,
            text           VARCHAR NOT NULL,
            embedding      {emb_type} NOT NULL
        )
        """


def _build_index_sql(dimension: int) -> str:
    """Return the ``CREATE INDEX`` DDL for the HNSW index.

    The *dimension* argument is accepted for symmetry with
    :func:`_build_schema` — the index itself is dimension-agnostic since
    DuckDB infers the column type from the table schema.
    """
    _ = dimension  # accepted for API symmetry; not used in the DDL
    return """
        CREATE INDEX IF NOT EXISTS chunks_hnsw_idx
        ON chunks USING HNSW (embedding) WITH (metric = 'cosine')
        """


class VectorStore:
    """DuckDB-backed vector store for chunked text.

    Use as a context manager to ensure the connection is closed::

        with VectorStore(path) as store:
            store.add_chunks(chunks)
            hits = store.search(query, k=4)

    Args:
        path:      File path to the DuckDB database, or ``":memory:"`` for
                   an in-memory database (tests).
        dimension: Embedding vector dimensionality. Must match the actual
                   model output dimension. Defaults to 384 (bge-small-en-v1.5).
    """

    def __init__(
        self,
        path: Path | str,
        dimension: int = DEFAULT_EMBEDDING_DIMENSION,
    ) -> None:
        self._path = str(path) if isinstance(path, Path) else path
        self._dimension = dimension
        self._conn: duckdb.DuckDBPyConnection | None = None

    # ------------------------------------------------------------------
    # Context manager / lifecycle
    # ------------------------------------------------------------------

    def __enter__(self) -> VectorStore:
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def open(self) -> None:
        """Open the database connection and ensure the schema exists.

        The VSS extension is installed and loaded on every open — DuckDB
        caches the install, so this is a no-op after the first call.
        For file-backed databases, experimental HNSW persistence is
        enabled (HNSW indexes cannot otherwise be created on a disk
        database). In-memory databases ignore the flag.
        """
        if self._conn is not None:
            return
        self._conn = duckdb.connect(self._path)
        self._conn.execute("INSTALL vss; LOAD vss;")
        # Allow HNSW indexes on file-backed DuckDB databases. Must come
        # after ``LOAD vss`` because the setting is provided by the
        # extension itself.
        self._conn.execute("SET hnsw_enable_experimental_persistence = true;")
        self._conn.execute(_build_schema(self._dimension))
        # HNSW index is created on first data load; see :meth:`add_chunks`.

    def close(self) -> None:
        """Close the database connection. Safe to call multiple times."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    @property
    def dimension(self) -> int:
        """Return the configured embedding vector dimensionality."""
        return self._dimension

    def _require_conn(self) -> duckdb.DuckDBPyConnection:
        if self._conn is None:
            raise RuntimeError(
                "VectorStore is not open. Use it as a context manager "
                "(`with VectorStore(path) as store: ...`) or call .open() first."
            )
        return self._conn

    # ------------------------------------------------------------------
    # Schema management
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Drop the chunks table and HNSW index.

        Used by the build-index CLI to rebuild from scratch. Safe to call
        on a fresh database (no-op if the table does not exist).
        """
        conn = self._require_conn()
        conn.execute("DROP INDEX IF EXISTS chunks_hnsw_idx")
        conn.execute("DROP TABLE IF EXISTS chunks")

    def ensure_schema(self) -> None:
        """Recreate the chunks table and HNSW index if missing.

        Idempotent: safe to call on an existing database. The HNSW index is
        built on an empty table — it will be populated as data is inserted
        and may be slow to build for large datasets.
        """
        conn = self._require_conn()
        conn.execute(_build_schema(self._dimension))

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def add_chunks(self, chunks: Sequence[Chunk]) -> None:
        """Insert or replace *chunks* into the store.

        Each chunk is upserted on its primary key (``id``), so re-running
        this method with the same chunks is a no-op. Embeddings are
        converted from Python lists to the DuckDB ``FLOAT[N]`` array type
        via a JSON intermediate.

        The HNSW index is created lazily — it is built once after the
        first batch is inserted, then maintained incrementally on
        subsequent inserts.
        """
        conn = self._require_conn()
        if not chunks:
            return

        # Ensure the table exists — this is a no-op on a fresh store
        # (already created in :meth:`open`) but matters when called
        # immediately after :meth:`clear`, which drops the table.
        conn.execute(_build_schema(self._dimension))

        rows = [_chunk_to_row(c) for c in chunks]
        # ``executemany`` with a parameterised statement is the most
        # portable path: no pandas/pyarrow dependency required and DuckDB
        # binds each tuple efficiently. The embedding is passed as a
        # JSON array which the CAST turns into a ``FLOAT[]``.
        conn.executemany(
            """
            INSERT OR REPLACE INTO chunks (
                id, chunk_type, deck_id, card_id, card_name, section,
                spread_id, position_index, source_url, text, embedding
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CAST(? AS FLOAT[]))
            """,
            rows,
        )

        # HNSW index maintenance: rebuild after bulk inserts for small
        # datasets (< 10k chunks). For the Book of Thoth (~860 chunks)
        # this is cheap and avoids drift.
        self._rebuild_hnsw_index()

    def _rebuild_hnsw_index(self) -> None:
        """Drop and recreate the HNSW index.

        Idempotent and cheap at our scale (~860 vectors). For larger
        datasets, switch to incremental ``INSERT`` into the index.
        """
        conn = self._require_conn()
        conn.execute("DROP INDEX IF EXISTS chunks_hnsw_idx")
        conn.execute(_build_index_sql(self._dimension))

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def _search_sql(self) -> str:
        """Return the similarity-ranked SELECT statement.

        The cast ``?::FLOAT[N]`` is templated against the store's
        dimension because DuckDB's ``array_cosine_similarity`` only
        accepts a matching ``FLOAT[ANY]`` pair — an unparameterised
        ``FLOAT[]`` cast is rejected.
        """
        n = self._dimension
        return (
            "SELECT id, chunk_type, deck_id, card_id, card_name, section, "
            "spread_id, position_index, source_url, text, embedding, "
            f"array_cosine_similarity(embedding, ?::FLOAT[{n}]) AS score "
            "FROM chunks"
        )

    def search(
        self,
        query_embedding: Sequence[float],
        k: int = 4,
    ) -> list[SearchHit]:
        """Return the *k* chunks most similar to *query_embedding*.

        Uses cosine similarity (HNSW index). Hits are returned in
        descending similarity order.

        Args:
            query_embedding: A single vector of length :attr:`dimension`.
            k:               Maximum number of hits to return.

        Returns:
            Up to *k* :class:`SearchHit` results, ordered by descending
            score. Empty list if the store is empty.
        """
        conn = self._require_conn()
        emb = _embedding_to_duckdb_array(query_embedding, self._dimension)
        sql = f"{self._search_sql()} ORDER BY score DESC LIMIT ?"
        rows = conn.execute(sql, [emb, k]).fetchall()
        return [_row_to_hit(row) for row in rows]

    def search_card_section(
        self,
        deck_id: str,
        card_id: str,
        query_embedding: Sequence[float],
        k: int = 4,
    ) -> list[SearchHit]:
        """Return top-k card-section chunks for *card_id* by similarity.

        Filters to chunks where ``chunk_type = 'card_section'`` AND
        ``deck_id = deck_id`` AND ``card_id = card_id`` before ranking.
        Useful for the per-card interpretation chain, which only wants
        context for the card that was just dealt.

        Note:
            *deck_id* is required to prevent cross-deck contamination
            when multiple decks are stored in the same vector database.
        """
        conn = self._require_conn()
        emb = _embedding_to_duckdb_array(query_embedding, self._dimension)
        sql = (
            f"{self._search_sql()} "
            "WHERE chunk_type = 'card_section' AND deck_id = ? AND card_id = ? "
            "ORDER BY score DESC LIMIT ?"
        )
        rows = conn.execute(sql, [emb, deck_id, card_id, k]).fetchall()
        return [_row_to_hit(row) for row in rows]

    def search_spread_position(
        self,
        spread_id: str,
        position_index: int,
    ) -> list[SearchHit]:
        """Return all spread-position chunks for one position of a spread.

        These are not similarity-ranked — a spread position has only one
        chunk by construction, so the result is a list of length 0 or 1.
        The signature is shaped like :meth:`search` for ergonomic symmetry
        in the chain code (plan 0008).
        """
        conn = self._require_conn()
        rows = conn.execute(
            """
            SELECT id, chunk_type, deck_id, card_id, card_name, section,
                   spread_id, position_index, source_url, text, embedding,
                   1.0 AS score
            FROM chunks
            WHERE chunk_type = 'spread_position'
              AND spread_id = ?
              AND position_index = ?
            """,
            [spread_id, position_index],
        ).fetchall()
        return [_row_to_hit(row) for row in rows]

    def count(self) -> int:
        """Return the number of chunks in the store.

        If the table does not exist (e.g. immediately after :meth:`clear`
        and before any re-insert), the count is ``0`` rather than an
        error — an empty store has zero rows by definition.
        """
        conn = self._require_conn()
        try:
            result = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        except duckdb.CatalogException:
            return 0
        return int(result[0]) if result else 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk_to_row(chunk: Chunk) -> tuple[Any, ...]:
    """Convert a :class:`Chunk` to a tuple matching the chunks table schema.

    Column order MUST match the INSERT statement in :meth:`VectorStore.add_chunks`:

        (id, chunk_type, deck_id, card_id, card_name, section,
         spread_id, position_index, source_url, text, embedding)
    """
    return (
        str(chunk.id),
        chunk.chunk_type.value,
        chunk.deck_id,
        chunk.card_id,
        chunk.card_name,
        chunk.section.value if chunk.section is not None else None,
        chunk.spread_id,
        chunk.position_index,
        chunk.source_url,
        chunk.text,
        # Pass the vector through JSON so DuckDB can CAST it to FLOAT[].
        json.dumps(list(chunk.embedding or [])),
    )


def _embedding_to_duckdb_array(
    embedding: Sequence[float],
    expected_dim: int,
) -> str:
    """Format *embedding* as the DuckDB ``FLOAT[N]`` literal.

    DuckDB accepts a JSON array as input for casting, so we serialise
    rather than building a literal like ``[0.1, 0.2, ...]::FLOAT[N]``
    (which is also valid but less robust to edge cases like NaN).
    """
    if len(embedding) != expected_dim:
        raise ValueError(
            f"Embedding dimension mismatch: expected {expected_dim}, got {len(embedding)}"
        )
    return json.dumps(list(embedding))


def _row_to_hit(row: tuple[Any, ...]) -> SearchHit:
    """Convert a DuckDB result row into a :class:`SearchHit`."""
    (
        id_,
        chunk_type,
        deck_id,
        card_id,
        card_name,
        section,
        spread_id,
        position_index,
        source_url,
        text,
        embedding,
        score,
    ) = row
    return SearchHit(
        chunk=Chunk(
            id=id_,
            chunk_type=chunk_type,
            deck_id=deck_id,
            card_id=card_id,
            card_name=card_name,
            section=section,
            spread_id=spread_id,
            position_index=position_index,
            source_url=source_url,
            text=text,
            embedding=list(embedding) if embedding is not None else None,
        ),
        score=float(score),
    )

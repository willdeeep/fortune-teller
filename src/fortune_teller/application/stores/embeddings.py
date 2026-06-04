"""Embedder wrapper around ``HuggingFaceEmbeddings``.

Wraps the LangChain ``HuggingFaceEmbeddings`` interface so that the rest of
the codebase depends on a small, testable protocol rather than the underlying
library. The HuggingFace model is loaded lazily on first use to keep import
times low and to make the test suite trivially stubbable.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from langchain_huggingface import HuggingFaceEmbeddings

from fortune_teller.application.config import settings

#: Embedding dimensionality of the default model
#: (``BAAI/bge-small-en-v1.5``). Used to size the vector-store schema.
DEFAULT_EMBEDDING_DIMENSION = 384


@runtime_checkable
class _EmbedderBackend(Protocol):
    """Minimal backend protocol — the subset of ``HuggingFaceEmbeddings`` we use.

    Defined as a :class:`~typing.Protocol` so a stub can satisfy the type
    checker without subclassing the heavy ``HuggingFaceEmbeddings`` class.
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents and return one vector per text."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text and return one vector."""
        ...


class Embedder:
    """Thin wrapper around :class:`HuggingFaceEmbeddings`.

    The underlying model is loaded on first call to :meth:`embed_texts` or
    :meth:`embed_query` so that simply instantiating the wrapper does not
    download any weights. Tests inject a stub backend via
    :meth:`set_backend` to avoid touching the network.

    Args:
        model_name: Name of the HuggingFace sentence-transformers model to
            use. Defaults to ``settings.embedding_model``
            (``BAAI/bge-small-en-v1.5`` by default).
        dimension:   Vector dimensionality of the model. Defaults to
            :data:`DEFAULT_EMBEDDING_DIMENSION` (matches bge-small-en-v1.5).
    """

    def __init__(
        self,
        model_name: str | None = None,
        dimension: int = DEFAULT_EMBEDDING_DIMENSION,
    ) -> None:
        self._model_name = model_name or settings.embedding_model
        self._dimension = dimension
        self._backend: _EmbedderBackend | None = None

    @property
    def model_name(self) -> str:
        """Return the configured model name."""
        return self._model_name

    @property
    def dimension(self) -> int:
        """Return the embedding vector dimensionality."""
        return self._dimension

    def set_backend(self, backend: _EmbedderBackend) -> None:
        """Inject a backend (used by tests to stub the network call)."""
        self._backend = backend

    def _get_backend(self) -> _EmbedderBackend:
        if self._backend is None:
            self._backend = HuggingFaceEmbeddings(model_name=self._model_name)
        return self._backend

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of *texts*.

        Args:
            texts: List of strings to embed. May be empty.

        Returns:
            One embedding vector per input text, in the same order. Each
            vector has length :attr:`dimension`.
        """
        if not texts:
            return []
        return self._get_backend().embed_documents(list(texts))

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query *text*.

        Some embedding models use a different prefix/prompt for queries vs
        documents; the underlying ``HuggingFaceEmbeddings`` already handles
        that distinction, so callers should not pre-process the text.
        """
        return self._get_backend().embed_query(text)

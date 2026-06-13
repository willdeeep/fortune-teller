"""Embedder wrapper around ``HuggingFaceEmbeddings``.

Wraps the LangChain ``HuggingFaceEmbeddings`` interface so that the rest of
the codebase depends on a small, testable protocol rather than the underlying
library. The HuggingFace model is loaded lazily on first use to keep import
times low and to make the test suite trivially stubbable.

When a local model snapshot is present at ``settings.embedding_model_path``,
the model is loaded from disk in offline mode (no HuggingFace Hub contact).
Otherwise it falls back to loading by hub name, preserving the existing
behaviour for anyone who has not run ``ft-fetch-models``.
"""

from __future__ import annotations

import os
from pathlib import Path
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


def _detect_device() -> dict[str, str]:
    """Return the device kwarg for HuggingFaceEmbeddings."""
    return {}


class Embedder:
    """Thin wrapper around :class:`HuggingFaceEmbeddings`.

    Resolution order:
    1. If ``settings.embedding_model_path`` exists on disk, load from that
       path in offline mode (``HF_HUB_OFFLINE=1``).
    2. Otherwise fall back to ``settings.embedding_model`` (hub name).

    Args:
        model_name: Override the hub model name. Primarily for testing.
            Defaults to ``settings.embedding_model``.
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
        """Return the resolved model name (local path or hub name)."""
        return self._resolve_model(name=self._model_name)

    @property
    def dimension(self) -> int:
        """Return the embedding vector dimensionality."""
        return self._dimension

    def set_backend(self, backend: _EmbedderBackend) -> None:
        """Inject a backend (used by tests to stub the network call)."""
        self._backend = backend

    @staticmethod
    def _resolve_model(name: str | None = None) -> str:
        """Resolve the model reference: local path if present, else hub name.

        Resolution order:
        1. If *name* is explicitly provided and differs from
           ``settings.embedding_model``, use it directly (caller override).
        2. If ``settings.embedding_model_path`` exists on disk, load from that
           path in offline mode (``HF_HUB_OFFLINE=1``).
        3. Fall back to *name* or ``settings.embedding_model`` (hub name).

        Args:
            name: Optional override (e.g. from the constructor). When set to a
                value different from ``settings.embedding_model`` it takes
                precedence over the local-path check.
        """
        # An explicit caller override beats the local-path check.
        if name is not None and name != settings.embedding_model:
            return name

        local = Path(settings.embedding_model_path)
        if local.exists():
            # Forbid any hub round-trip when loading from a local snapshot.
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            return str(local)
        return name or settings.embedding_model

    def _get_backend(self) -> _EmbedderBackend:
        if self._backend is None:
            model_ref = self._resolve_model(name=self._model_name)
            self._backend = HuggingFaceEmbeddings(
                model_name=model_ref,
                model_kwargs=_detect_device(),
                encode_kwargs={"normalize_embeddings": True},
            )
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

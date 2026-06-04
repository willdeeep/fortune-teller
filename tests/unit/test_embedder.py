"""Unit tests for ``Embedder`` — no network calls, stubbed backend only."""

from __future__ import annotations

import pytest

from fortune_teller.application.stores.embeddings import (
    DEFAULT_EMBEDDING_DIMENSION,
    Embedder,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _CountingBackend:
    """Stub backend that records call counts and returns deterministic vectors."""

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim
        self.embed_documents_calls: list[list[str]] = []
        self.embed_query_calls: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.embed_documents_calls.append(list(texts))
        return [[0.1 * (i + 1)] * self.dim for i in range(len(texts))]

    def embed_query(self, text: str) -> list[float]:
        self.embed_query_calls.append(text)
        return [0.5] * self.dim


# ---------------------------------------------------------------------------
# Configuration & defaults
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEmbedderDefaults:
    def test_dimension_defaults_to_384(self) -> None:
        embedder = Embedder()
        assert embedder.dimension == DEFAULT_EMBEDDING_DIMENSION == 384

    def test_model_name_defaults_to_settings(self) -> None:
        embedder = Embedder()
        assert embedder.model_name == "BAAI/bge-small-en-v1.5"

    def test_explicit_model_name_is_respected(self) -> None:
        embedder = Embedder(model_name="custom-model")
        assert embedder.model_name == "custom-model"

    def test_explicit_dimension_is_respected(self) -> None:
        embedder = Embedder(dimension=768)
        assert embedder.dimension == 768


# ---------------------------------------------------------------------------
# Lazy backend loading
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEmbedderLazyLoading:
    def test_backend_is_none_at_construction(self) -> None:
        embedder = Embedder()
        assert embedder._backend is None

    def test_set_backend_injects_stub(self) -> None:
        embedder = Embedder(dimension=4)
        backend = _CountingBackend(dim=4)
        embedder.set_backend(backend)
        assert embedder._backend is backend

    def test_first_call_uses_injected_backend_without_loading(self, monkeypatch) -> None:
        """Verify the heavy HuggingFaceEmbeddings constructor is never called
        when a stub backend has been injected."""
        embedder = Embedder(dimension=4)
        backend = _CountingBackend(dim=4)
        embedder.set_backend(backend)

        called = {"hf": False}

        class _FakeHF:
            def __init__(self, *args, **kwargs):  # noqa: ARG002
                called["hf"] = True

        monkeypatch.setattr(
            "fortune_teller.application.stores.embeddings.HuggingFaceEmbeddings",
            _FakeHF,
        )

        embedder.embed_query("hello")
        assert called["hf"] is False
        assert backend.embed_query_calls == ["hello"]


# ---------------------------------------------------------------------------
# embed_texts
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEmbedderEmbedTexts:
    def test_empty_input_returns_empty_list(self) -> None:
        embedder = Embedder(dimension=4)
        embedder.set_backend(_CountingBackend(dim=4))
        assert embedder.embed_texts([]) == []

    def test_returns_one_vector_per_text(self) -> None:
        embedder = Embedder(dimension=8)
        embedder.set_backend(_CountingBackend(dim=8))
        vectors = embedder.embed_texts(["a", "b", "c"])
        assert len(vectors) == 3
        for v in vectors:
            assert len(v) == 8

    def test_vectors_match_dimension(self) -> None:
        embedder = Embedder(dimension=16)
        embedder.set_backend(_CountingBackend(dim=16))
        vectors = embedder.embed_texts(["x"])
        assert len(vectors[0]) == 16


# ---------------------------------------------------------------------------
# embed_query
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEmbedderEmbedQuery:
    def test_returns_single_vector(self) -> None:
        embedder = Embedder(dimension=4)
        embedder.set_backend(_CountingBackend(dim=4))
        vec = embedder.embed_query("a question")
        assert len(vec) == 4

    def test_uses_query_method_not_documents(self) -> None:
        embedder = Embedder(dimension=4)
        backend = _CountingBackend(dim=4)
        embedder.set_backend(backend)
        embedder.embed_query("hi")
        assert backend.embed_documents_calls == []
        assert backend.embed_query_calls == ["hi"]


# ---------------------------------------------------------------------------
# _get_backend
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestEmbedderGetBackend:
    def test_loads_default_backend_on_first_use(self, monkeypatch) -> None:
        """When no backend is injected, the first call materialises
        HuggingFaceEmbeddings. We monkeypatch it to a stub so the test
        remains offline."""
        embedder = Embedder(model_name="stub-model", dimension=4)
        calls = {"init": 0}

        class _FakeHF:
            def __init__(self, *args, **kwargs):  # noqa: ARG002
                calls["init"] += 1

            def embed_query(self, text: str) -> list[float]:  # noqa: ARG002
                return [0.0] * 4

        monkeypatch.setattr(
            "fortune_teller.application.stores.embeddings.HuggingFaceEmbeddings",
            _FakeHF,
        )
        embedder.embed_query("hi")
        embedder.embed_query("again")
        # Default backend is only constructed once (cached).
        assert calls["init"] == 1

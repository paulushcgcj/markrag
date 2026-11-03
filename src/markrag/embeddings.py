"""Embedding support for markrag.

Embeddings are produced by a pluggable provider:

- ``fastembed`` (default) runs the ONNX ``all-MiniLM-L6-v2`` model locally on
  the CPU. No API key, no hosted service. The model is downloaded on first use.
- An OpenAI-compatible endpoint can replace it through environment variables,
  which covers BGE-large self-hosted servers and commercial APIs alike.

Every provider is selected through :func:`get_embedding_provider`, so swapping
one for another never changes the retrieval code.
"""

import os
from typing import Any, Protocol, runtime_checkable

import numpy as np

DEFAULT_FASTEMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_OPENAI_MODEL = "text-embedding-ada-002"
DEFAULT_OPENAI_BASE_URL = "http://localhost:8080"

# Short names accepted for convenience, mapped to their canonical names.
_MODEL_ALIASES = {
    "all-minilm-l6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "bge-small-en": "BAAI/bge-small-en-v1.5",
    "bge-small": "BAAI/bge-small-en-v1.5",
    "bge-base-en": "BAAI/bge-base-en-v1.5",
    "bge-large-en": "BAAI/bge-large-en-v1.5",
}


def normalise_model_name(model_name: str) -> str:
    """Resolve a short model alias to its canonical name.

    Args:
        model_name: Model name or alias.

    Returns:
        The canonical model name.
    """
    return _MODEL_ALIASES.get(model_name.strip().lower(), model_name)


def _first_nonempty(*candidates: str | None) -> str:
    """Return the first candidate that is set and not blank.

    Environment variables that exist but hold only whitespace are treated as
    unset, so a stray empty value never selects a broken model or endpoint.

    Args:
        candidates: Values to try, in order.

    Returns:
        The first usable value, or an empty string when none is usable.
    """
    for candidate in candidates:
        if candidate and candidate.strip():
            return candidate.strip()
    return ""


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol implemented by every embedding backend."""

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string."""
        ...

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Embed a batch of text strings."""
        ...


class FastEmbedProvider:
    """Embedding provider using fastembed's local ONNX models."""

    def __init__(self, model_name: str | None = None) -> None:
        """Initialize the provider.

        Args:
            model_name: Model to load. Defaults to ``MARKRAG_EMBEDDING_MODEL``
                or ``all-MiniLM-L6-v2``.
        """
        self.model_name = normalise_model_name(
            _first_nonempty(
                model_name,
                os.getenv("MARKRAG_EMBEDDING_MODEL"),
                DEFAULT_FASTEMBED_MODEL,
            )
        )
        self._embedding_model: Any = None

    @property
    def embedding_model(self) -> Any:
        """The lazily loaded fastembed model."""
        if self._embedding_model is None:
            try:
                import fastembed
            except ImportError as exc:  # pragma: no cover - dependency is declared
                raise ImportError(
                    "fastembed is required for local embeddings. Install it with: uv add fastembed"
                ) from exc
            self._embedding_model = fastembed.TextEmbedding(model_name=self.model_name)
        return self._embedding_model

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Embed a batch of text strings."""
        if not texts:
            return []
        return [
            np.asarray(vector, dtype=np.float32) for vector in self.embedding_model.embed(texts)
        ]


class OpenAICompatibleProvider:
    """Embedding provider for OpenAI-compatible HTTP endpoints."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model_name: str | None = None,
    ) -> None:
        """Initialize the provider.

        Args:
            base_url: API base URL. Defaults to ``MARKRAG_EMBEDDING_BASE_URL``.
            api_key: API key. Defaults to ``MARKRAG_EMBEDDING_API_KEY``.
            model_name: Model name. Defaults to ``MARKRAG_EMBEDDING_MODEL``.
        """
        try:
            import httpx  # noqa: F401
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "httpx is required for OpenAI-compatible embeddings. Install it with: uv add httpx"
            ) from exc

        self.base_url = _first_nonempty(
            base_url,
            os.getenv("MARKRAG_EMBEDDING_BASE_URL"),
            DEFAULT_OPENAI_BASE_URL,
        )
        self.api_key = _first_nonempty(api_key, os.getenv("MARKRAG_EMBEDDING_API_KEY")) or None
        self.model_name = _first_nonempty(
            model_name,
            os.getenv("MARKRAG_EMBEDDING_MODEL"),
            DEFAULT_OPENAI_MODEL,
        )

        if not self.api_key:
            raise ValueError(
                "API key is required for OpenAI-compatible embeddings. "
                "Set MARKRAG_EMBEDDING_API_KEY environment variable."
            )

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Embed a batch of text strings."""
        import httpx

        if not texts:
            return []

        with httpx.Client(base_url=self.base_url, timeout=60.0) as client:
            response = client.post(
                "/v1/embeddings",
                json={"model": self.model_name, "input": texts},
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
            response.raise_for_status()
            payload = response.json()

        return [np.asarray(item["embedding"], dtype=np.float32) for item in payload.get("data", [])]


class CachedEmbeddingProvider:
    """Provider wrapper that memoises embeddings by text."""

    def __init__(self, provider: EmbeddingProvider) -> None:
        """Initialize the wrapper.

        Args:
            provider: The provider to cache.
        """
        self.provider = provider
        self._cache: dict[str, np.ndarray] = {}

    def embed(self, text: str) -> np.ndarray:
        """Embed with caching."""
        cached = self._cache.get(text)
        if cached is None:
            cached = self.provider.embed(text)
            self._cache[text] = cached
        return cached

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Embed a batch, reusing cached vectors where possible."""
        if not texts:
            return []

        results: list[np.ndarray | None] = [None] * len(texts)
        missing: list[str] = []
        missing_positions: list[int] = []

        for position, text in enumerate(texts):
            cached = self._cache.get(text)
            if cached is not None:
                results[position] = cached
            else:
                missing.append(text)
                missing_positions.append(position)

        if missing:
            vectors = self.provider.embed_batch(missing)
            for position, text, vector in zip(missing_positions, missing, vectors, strict=True):
                self._cache[text] = vector
                results[position] = vector

        return [vector for vector in results if vector is not None]


def provider_signature(provider: EmbeddingProvider | None) -> str:
    """Describe an embedding configuration in a way that can be compared.

    The index records this signature alongside the bundle fingerprint. When it
    changes - embeddings turned on, a different model chosen - the index is
    rebuilt instead of silently keeping stale vectors.

    Args:
        provider: The configured provider, or ``None`` when embeddings are off.

    Returns:
        A short string identifying the configuration.
    """
    if provider is None:
        return "none"
    inner = getattr(provider, "provider", provider)
    kind = type(inner).__name__
    model = getattr(inner, "model_name", "unknown")
    return f"{kind}:{model}"


def get_embedding_provider() -> EmbeddingProvider | None:
    """Build the configured embedding provider.

    ``MARKRAG_EMBEDDING_PROVIDER=openai`` selects the OpenAI-compatible
    endpoint; anything else selects local fastembed embeddings. Setting
    ``MARKRAG_EMBEDDINGS=off`` disables embeddings entirely, which leaves
    retrieval working on keyword search alone.

    Returns:
        A caching embedding provider, or ``None`` when embeddings are disabled
        or the configured provider cannot be constructed.
    """
    if os.getenv("MARKRAG_EMBEDDINGS", "").strip().lower() in {"0", "off", "false", "none"}:
        return None

    provider = os.getenv("MARKRAG_EMBEDDING_PROVIDER", "fastembed")
    try:
        if provider == "openai":
            return CachedEmbeddingProvider(OpenAICompatibleProvider())
        return CachedEmbeddingProvider(FastEmbedProvider())
    except (ImportError, ValueError):
        # A misconfigured optional provider must not stop keyword retrieval.
        return None

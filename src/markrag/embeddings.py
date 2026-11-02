"""Embedding support for markrag using fastembed.

Provides a pluggable embedding interface with:
- Default: fastembed with all-MiniLM-L6-v2 (local, CPU-only, no API key)
- Optional: BGE-large or OpenAI-compatible endpoint via env vars
"""

import os
from typing import Protocol

import numpy as np


class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string."""
        ...

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Embed a batch of text strings."""
        ...


class FastEmbedProvider:
    """Embedding provider using fastembed."""

    def __init__(self, model_name: str | None = None):
        """Initialize fastembed provider.

        Args:
            model_name: Model to use. Defaults to all-MiniLM-L6-v2.
        """
        try:
            import fastembed
        except ImportError:
            raise ImportError(
                "fastembed is required for local embeddings. "
                "Install with: uv add fastembed"
            )

        self.model_name = model_name or os.getenv(
            "MARKRAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2"
        )
        self._embedding_model = None

    @property
    def embedding_model(self):
        """Lazy-load the embedding model."""
        if self._embedding_model is None:
            import fastembed

            # Initialize the model
            self._embedding_model = fastembed.TextEmbedding(
                model_name=self.model_name
            )
        return self._embedding_model

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Embed a batch of text strings."""
        if not texts:
            return []

        # fastembed returns a list of embeddings
        embeddings = list(self.embedding_model.embed(texts))
        return [np.array(e) for e in embeddings]


class OpenAICompatibleProvider:
    """Embedding provider for OpenAI-compatible APIs."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model_name: str | None = None,
    ):
        """Initialize OpenAI-compatible provider.

        Args:
            base_url: API base URL (defaults to MARKRAG_EMBEDDING_BASE_URL)
            api_key: API key (defaults to MARKRAG_EMBEDDING_API_KEY)
            model_name: Model to use (defaults to MARKRAG_EMBEDDING_MODEL)
        """
        try:
            import httpx
            import numpy as np
        except ImportError:
            raise ImportError(
                "httpx and numpy are required for OpenAI-compatible embeddings. "
                "Install with: uv add httpx numpy"
            )

        self.base_url = base_url or os.getenv("MARKRAG_EMBEDDING_BASE_URL", "http://localhost:8080")
        self.api_key = api_key or os.getenv("MARKRAG_EMBEDDING_API_KEY")
        self.model_name = model_name or os.getenv(
            "MARKRAG_EMBEDDING_MODEL", "text-embedding-ada-002"
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

        client = httpx.Client(base_url=self.base_url)

        try:
            response = client.post(
                "/v1/embeddings",
                json={
                    "model": self.model_name,
                    "input": texts,
                },
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
            response.raise_for_status()

            data = response.json()
            embeddings = data["data"]

            return [np.array(e["embedding"]) for e in embeddings]
        finally:
            client.close()


def get_embedding_provider() -> EmbeddingProvider:
    """Get the configured embedding provider.

    Priority:
    1. If MARKRAG_EMBEDDING_PROVIDER=openai, use OpenAI-compatible
    2. Otherwise, use fastembed (default)

    Returns:
        Configured embedding provider
    """
    provider = os.getenv("MARKRAG_EMBEDDING_PROVIDER", "fastembed")

    if provider == "openai":
        return OpenAICompatibleProvider()
    else:
        return FastEmbedProvider()


class CachedEmbeddingProvider:
    """Wrapper that caches embeddings."""

    def __init__(self, provider: EmbeddingProvider):
        """Initialize cached provider."""
        self.provider = provider
        self._cache: dict[str, np.ndarray] = {}

    def embed(self, text: str) -> np.ndarray:
        """Embed with caching."""
        if text not in self._cache:
            self._cache[text] = self.provider.embed(text)
        return self._cache[text]

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Embed batch with caching."""
        uncached = []
        uncached_indices = []

        results = [None] * len(texts)

        for i, text in enumerate(texts):
            if text in self._cache:
                results[i] = self._cache[text]
            else:
                uncached.append(text)
                uncached_indices.append(i)

        if uncached:
            embeddings = self.provider.embed_batch(uncached)
            for idx, emb in zip(uncached_indices, embeddings):
                text = texts[idx]
                self._cache[text] = emb
                results[idx] = emb

        return [r for r in results if r is not None]

"""Tests for the embedding providers."""

import numpy as np
import pytest

from markrag.embeddings import (
    DEFAULT_FASTEMBED_MODEL,
    CachedEmbeddingProvider,
    EmbeddingProvider,
    FastEmbedProvider,
    OpenAICompatibleProvider,
    _first_nonempty,
    get_embedding_provider,
    normalise_model_name,
)


class StubProvider:
    """A provider that returns a deterministic vector per text."""

    def __init__(self) -> None:
        """Track how often the underlying provider was called."""
        self.calls: list[list[str]] = []

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Embed a batch of text strings."""
        self.calls.append(list(texts))
        return [np.asarray([float(len(text)), 1.0], dtype=np.float32) for text in texts]


class TestModelNames:
    """Test model name normalisation."""

    def test_canonical_name_is_unchanged(self):
        """A canonical name passes through untouched."""
        assert normalise_model_name(DEFAULT_FASTEMBED_MODEL) == DEFAULT_FASTEMBED_MODEL

    @pytest.mark.parametrize(
        "alias",
        ["all-MiniLM-L6-v2", "all-minilm-l6-v2", "MiniLM", "minilm"],
    )
    def test_aliases_resolve_to_the_canonical_name(self, alias):
        """Short aliases map to the name fastembed expects."""
        assert normalise_model_name(alias) == DEFAULT_FASTEMBED_MODEL

    def test_bge_aliases_resolve(self):
        """The BGE aliases map to their canonical Hugging Face names."""
        assert normalise_model_name("bge-small") == "BAAI/bge-small-en-v1.5"
        assert normalise_model_name("bge-large-en") == "BAAI/bge-large-en-v1.5"

    def test_unknown_names_pass_through(self):
        """An unrecognised name is left alone."""
        assert normalise_model_name("some/other-model") == "some/other-model"

    def test_surrounding_whitespace_is_ignored(self):
        """Whitespace never breaks the alias lookup."""
        assert normalise_model_name("  minilm  ") == DEFAULT_FASTEMBED_MODEL


class TestFirstNonempty:
    """Test the environment fallback helper."""

    def test_first_value_wins(self):
        """The first usable candidate is returned."""
        assert _first_nonempty("a", "b", "c") == "a"

    def test_none_is_skipped(self):
        """A missing value is skipped."""
        assert _first_nonempty(None, "b") == "b"

    def test_blank_values_are_skipped(self):
        """Whitespace-only values are treated as unset."""
        assert _first_nonempty("   ", None, "c") == "c"

    def test_no_usable_value_returns_empty(self):
        """Nothing usable gives an empty string."""
        assert _first_nonempty(None, "  ") == ""


class TestCachedProvider:
    """Test the caching wrapper."""

    def test_repeated_text_is_embedded_once(self):
        """A cached text is not sent to the provider again."""
        stub = StubProvider()
        cached = CachedEmbeddingProvider(stub)

        first = cached.embed("hello")
        second = cached.embed("hello")

        assert np.array_equal(first, second)
        assert stub.calls == [["hello"]]

    def test_batch_reuses_cached_vectors(self):
        """A batch only sends the texts it has not seen."""
        stub = StubProvider()
        cached = CachedEmbeddingProvider(stub)

        cached.embed_batch(["a", "b"])
        cached.embed_batch(["b", "c"])

        assert stub.calls == [["a", "b"], ["c"]]

    def test_empty_batch_is_free(self):
        """An empty batch never reaches the provider."""
        stub = StubProvider()
        cached = CachedEmbeddingProvider(stub)

        assert cached.embed_batch([]) == []
        assert cached.embed_batch([]) == []
        assert stub.calls == []

    def test_satisfies_the_provider_protocol(self):
        """The wrapper can stand in for any provider."""
        assert isinstance(CachedEmbeddingProvider(StubProvider()), EmbeddingProvider)


class TestProviderSelection:
    """Test how the configured provider is chosen."""

    @pytest.mark.parametrize("value", ["off", "OFF", "0", "false", "none", " off "])
    def test_embeddings_can_be_turned_off(self, monkeypatch, value):
        """Disabling embeddings returns no provider at all."""
        monkeypatch.setenv("MARKRAG_EMBEDDINGS", value)
        assert get_embedding_provider() is None

    def test_openai_provider_without_a_key_is_disabled(self, monkeypatch):
        """A misconfigured provider never breaks keyword retrieval."""
        monkeypatch.delenv("MARKRAG_EMBEDDINGS", raising=False)
        monkeypatch.setenv("MARKRAG_EMBEDDING_PROVIDER", "openai")
        monkeypatch.delenv("MARKRAG_EMBEDDING_API_KEY", raising=False)

        assert get_embedding_provider() is None

    def test_openai_provider_with_a_key_is_selected(self, monkeypatch):
        """A configured OpenAI-compatible endpoint is used."""
        monkeypatch.delenv("MARKRAG_EMBEDDINGS", raising=False)
        monkeypatch.setenv("MARKRAG_EMBEDDING_PROVIDER", "openai")
        monkeypatch.setenv("MARKRAG_EMBEDDING_API_KEY", "test-key")
        monkeypatch.setenv("MARKRAG_EMBEDDING_BASE_URL", "http://localhost:9999")

        provider = get_embedding_provider()
        assert provider is not None
        assert isinstance(provider, CachedEmbeddingProvider)
        assert isinstance(provider.provider, OpenAICompatibleProvider)

    def test_local_provider_is_selected_by_default(self, monkeypatch):
        """Without configuration the local fastembed provider is used."""
        monkeypatch.delenv("MARKRAG_EMBEDDINGS", raising=False)
        monkeypatch.delenv("MARKRAG_EMBEDDING_PROVIDER", raising=False)

        provider = get_embedding_provider()
        assert provider is not None
        assert isinstance(provider, CachedEmbeddingProvider)
        assert isinstance(provider.provider, FastEmbedProvider)


class TestFastEmbedProvider:
    """Test the local embedding provider's configuration."""

    def test_model_name_comes_from_the_environment(self, monkeypatch):
        """The environment selects the model."""
        monkeypatch.setenv("MARKRAG_EMBEDDING_MODEL", "bge-small")

        provider = FastEmbedProvider()
        assert provider.model_name == "BAAI/bge-small-en-v1.5"

    def test_explicit_model_name_wins(self, monkeypatch):
        """An explicit argument beats the environment."""
        monkeypatch.setenv("MARKRAG_EMBEDDING_MODEL", "bge-small")

        provider = FastEmbedProvider("minilm")
        assert provider.model_name == DEFAULT_FASTEMBED_MODEL

    def test_blank_environment_value_falls_back_to_the_default(self, monkeypatch):
        """A blank environment variable does not select an empty model."""
        monkeypatch.setenv("MARKRAG_EMBEDDING_MODEL", "   ")

        provider = FastEmbedProvider()
        assert provider.model_name == DEFAULT_FASTEMBED_MODEL


class TestOpenAICompatibleProvider:
    """Test the HTTP embedding provider's configuration."""

    def test_base_url_and_model_come_from_the_environment(self, monkeypatch):
        """The environment configures the endpoint."""
        monkeypatch.setenv("MARKRAG_EMBEDDING_BASE_URL", "http://example.test")
        monkeypatch.setenv("MARKRAG_EMBEDDING_MODEL", "custom-model")
        monkeypatch.setenv("MARKRAG_EMBEDDING_API_KEY", "key")

        provider = OpenAICompatibleProvider()
        assert provider.base_url == "http://example.test"
        assert provider.model_name == "custom-model"
        assert provider.api_key == "key"

    def test_a_missing_api_key_is_an_error(self, monkeypatch):
        """An endpoint without a key cannot be used."""
        monkeypatch.delenv("MARKRAG_EMBEDDING_API_KEY", raising=False)

        with pytest.raises(ValueError, match="API key"):
            OpenAICompatibleProvider()

    def test_embeddings_are_requested_from_the_v1_endpoint(self, monkeypatch):
        """The provider posts to the standard embeddings route."""
        import httpx

        requested: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            requested["url"] = str(request.url)
            requested["auth"] = request.headers.get("authorization")
            requested["body"] = request.content.decode()
            return httpx.Response(
                200,
                json={"data": [{"embedding": [0.1, 0.2, 0.3]} for _ in range(2)]},
            )

        transport = httpx.MockTransport(handler)
        original_client = httpx.Client

        def patched_client(*args, **kwargs):
            kwargs["transport"] = transport
            return original_client(*args, **kwargs)

        monkeypatch.setattr(httpx, "Client", patched_client)

        provider = OpenAICompatibleProvider(base_url="http://example.test", api_key="secret")
        vectors = provider.embed_batch(["one", "two"])

        assert len(vectors) == 2
        assert vectors[0].dtype == np.float32
        assert requested["url"] == "http://example.test/v1/embeddings"
        assert requested["auth"] == "Bearer secret"
        assert "one" in str(requested["body"])

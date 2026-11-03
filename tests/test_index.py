"""Tests for the hybrid retrieval index."""

import numpy as np
import pytest

from markrag.index import KnowledgeIndex, bundle_fingerprint
from markrag.okf import Bundle


class ScriptedProvider:
    """Deterministic embeddings driven by explicit text needles.

    Real models are not available in CI, and the fusion logic needs exact
    control over which document is the closest *semantic* match. This provider
    returns a scripted vector for the first needle found in the text.
    """

    def __init__(self, mapping: dict[str, list[float]], default: list[float] | None = None) -> None:
        """Initialize the provider.

        Args:
            mapping: Text needle to vector.
            default: Vector used when no needle matches.
        """
        self.mapping = mapping
        self.default = default or [0.0, 0.0]
        self.calls: list[str] = []

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string."""
        self.calls.append(text)
        for needle, vector in self.mapping.items():
            if needle in text:
                return np.asarray(vector, dtype=np.float32)
        return np.asarray(self.default, dtype=np.float32)

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Embed a batch of text strings."""
        return [self.embed(text) for text in texts]


@pytest.fixture
def retrieval_bundle(tmp_path):
    """Build a bundle with one keyword-exact and one semantically-close document.

    ``tech/database`` contains the query words, so keyword search finds it.
    ``tech/cache`` shares no words with the query but is the closest
    *semantic* match under the scripted provider.
    """
    wiki = tmp_path / "retrieval.wiki"
    (wiki / "tech").mkdir(parents=True)

    (wiki / "tech" / "database.md").write_text(
        """---
type: Technology
title: Database Configuration
description: Database settings overview.
tags: [data]
status: stable
---

# Database Configuration

## Connection Pooling

The pool is sized to ten connections.
""",
        encoding="utf-8",
    )

    (wiki / "tech" / "cache.md").write_text(
        """---
type: Technology
title: Redis Cache
description: Entry expiry overview.
tags: [cache]
status: draft
---

# Redis Cache

## Eviction

Entries expire after an hour.
""",
        encoding="utf-8",
    )

    bundle = Bundle(root_path=wiki)
    bundle.load()
    return bundle


@pytest.fixture
def scripted_index(tmp_path, retrieval_bundle):
    """Build an index whose vector half favours the semantically-close document."""
    provider = ScriptedProvider(
        {
            # The query vector points at the cache document.
            "connection pooling": [1.0, 0.0],
            "Redis Cache": [1.0, 0.0],
            "Database Configuration": [0.0, 1.0],
        }
    )
    index = KnowledgeIndex(tmp_path / "index.db", "retrieval", provider=provider)
    index.build(retrieval_bundle)
    yield index
    index.close()


class TestHybridSearch:
    """Test the hybrid keyword and vector search."""

    def test_keyword_match_outranks_semantically_close_document(self, scripted_index):
        """A keyword-exact match must outrank a semantically-close-but-wrong doc."""
        response = scripted_index.search("connection pooling", top_k=5)

        concept_ids = [hit.concept_id for hit in response.hits]
        assert "tech/database" in concept_ids
        assert "tech/cache" in concept_ids
        assert concept_ids.index("tech/database") < concept_ids.index("tech/cache")

        database_hit = next(h for h in response.hits if h.concept_id == "tech/database")
        cache_hit = next(h for h in response.hits if h.concept_id == "tech/cache")
        assert database_hit.score > cache_hit.score

    def test_both_halves_contribute_to_a_hit(self, scripted_index):
        """A document found by both halves scores higher than one found by one."""
        response = scripted_index.search("connection pooling", top_k=5)

        database_hit = next(h for h in response.hits if h.concept_id == "tech/database")
        cache_hit = next(h for h in response.hits if h.concept_id == "tech/cache")

        assert database_hit.fts_rank is not None
        assert database_hit.vector_rank is not None
        assert cache_hit.fts_rank is None
        assert cache_hit.vector_rank is not None

    def test_search_never_returns_full_bodies(self, scripted_index):
        """Tier 1 returns excerpts, not whole documents."""
        response = scripted_index.search("connection pooling", top_k=5)

        for hit in response.hits:
            assert len(hit.excerpt_text) < 2000

    def test_metadata_prefilter_narrows_candidates(self, scripted_index):
        """Type, tag, and status filters are applied before scoring."""
        assert scripted_index.search("connection pooling", type_="Technology").hits
        assert not scripted_index.search("connection pooling", type_="Process").hits
        assert scripted_index.search("connection pooling", tags=["data"]).hits
        assert not scripted_index.search("connection pooling", tags=["nonexistent"]).hits
        assert scripted_index.search("connection pooling", status="draft").hits
        assert scripted_index.search("connection pooling", status="stable").hits
        assert scripted_index.search("connection pooling", status="any").hits

    def test_top_k_limits_results(self, scripted_index):
        """The result count never exceeds top_k."""
        assert len(scripted_index.search("connection pooling", top_k=1).hits) == 1

    def test_section_anchor_points_at_the_matched_section(self, scripted_index):
        """The anchor names the section that matched."""
        response = scripted_index.search("connection pooling", top_k=5)

        database_hit = next(h for h in response.hits if h.concept_id == "tech/database")
        assert database_hit.anchor == "Connection Pooling"

    def test_query_with_only_fts_operators_is_safe(self, scripted_index):
        """FTS5 operators typed by a user must not break the query."""
        response = scripted_index.search('"connection" AND (pooling OR NEAR)', top_k=5)
        assert response.hits


class TestIndexLifecycle:
    """Test rebuilding and disposal of the index."""

    def test_index_is_rebuilt_when_the_bundle_changes(self, tmp_path, retrieval_bundle):
        """Editing a document invalidates the index."""
        index = KnowledgeIndex(tmp_path / "index.db", "retrieval")
        index.build(retrieval_bundle)
        assert index.is_current(retrieval_bundle)

        concept = retrieval_bundle.root_path / "tech" / "cache.md"
        concept.write_text(
            concept.read_text(encoding="utf-8").replace("Eviction", "Eviction Policy"),
            encoding="utf-8",
        )

        assert not index.is_current(retrieval_bundle)
        index.build(retrieval_bundle)
        assert index.is_current(retrieval_bundle)
        index.close()

    def test_index_can_be_rebuilt_from_files_alone(self, tmp_path, retrieval_bundle):
        """Deleting the database loses nothing: the files are the source of truth."""
        db_path = tmp_path / "index.db"
        index = KnowledgeIndex(db_path, "retrieval")
        index.build(retrieval_bundle)
        index.close()

        db_path.unlink()

        rebuilt = KnowledgeIndex(db_path, "retrieval")
        rebuilt.build(retrieval_bundle)
        hits = rebuilt.search("connection pooling", top_k=5)
        assert hits.hits
        assert hits.fts_available
        rebuilt.close()

    def test_fingerprint_changes_with_content(self, retrieval_bundle):
        """The fingerprint tracks the bundle's contents."""
        before = bundle_fingerprint(retrieval_bundle)

        concept = retrieval_bundle.root_path / "tech" / "cache.md"
        concept.write_text(concept.read_text(encoding="utf-8") + "\nMore.\n", encoding="utf-8")

        assert bundle_fingerprint(retrieval_bundle) != before

    def test_fingerprint_ignores_navigation_files(self, retrieval_bundle):
        """Regenerating the indexes does not force a reindex."""
        before = bundle_fingerprint(retrieval_bundle)

        (retrieval_bundle.root_path / "index.md").write_text("# Index\n", encoding="utf-8")
        (retrieval_bundle.root_path / "log.md").write_text("# Log\n", encoding="utf-8")
        (retrieval_bundle.root_path / "tech" / "index.md").write_text("# Tech\n", encoding="utf-8")

        assert bundle_fingerprint(retrieval_bundle) == before

    def test_vectors_are_absent_without_a_provider(self, tmp_path, retrieval_bundle):
        """Without a provider the index still works, on keywords alone."""
        index = KnowledgeIndex(tmp_path / "index.db", "retrieval")
        assert index.build(retrieval_bundle) is False
        assert not index.vectors_available()

        response = index.search("connection pooling", top_k=5)
        assert response.hits
        assert not response.vector_available
        index.close()

    def test_changing_the_embedding_configuration_invalidates_the_index(
        self, tmp_path, retrieval_bundle
    ):
        """Turning embeddings on rebuilds the index instead of keeping it stale."""
        index = KnowledgeIndex(tmp_path / "index.db", "retrieval")
        index.build(retrieval_bundle)
        assert index.is_current(retrieval_bundle)

        # Same documents, different embedding configuration.
        index.embedding_signature = "FastEmbedProvider:sentence-transformers/all-MiniLM-L6-v2"
        assert not index.is_current(retrieval_bundle)

        index.build(retrieval_bundle)
        assert index.is_current(retrieval_bundle)
        index.close()

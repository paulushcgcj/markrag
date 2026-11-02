"""SQLite FTS5 + vector index for OKF concepts.

Provides hybrid search with:
- FTS5 full-text search
- Vector similarity search via fastembed
- Reciprocal Rank Fusion (RRF)
- Metadata pre-filtering
"""

import hashlib
import os
import pickle
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from markrag.chunking import Chunk, chunk_concept, chunk_description
from markrag.okf import Bundle, Concept, derive_trust_level, is_stale


class KnowledgeIndex:
    """SQLite-based index for OKF bundle with FTS5 and vector storage."""

    def __init__(self, db_path: Path, bundle_name: str):
        """Initialize the index.

        Args:
            db_path: Path to the SQLite database file
            bundle_name: Name of the bundle (used for table naming)
        """
        self.db_path = db_path
        self.bundle_name = bundle_name
        self._conn: sqlite3.Connection | None = None

    def _ensure_conn(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
            # Enable WAL mode for better concurrent access
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
        return self._conn

    def _table_name(self, suffix: str) -> str:
        """Generate table name with bundle prefix."""
        safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", self.bundle_name)
        return f"{safe_name}_{suffix}"

    def initialize(self) -> None:
        """Initialize database schema."""
        conn = self._ensure_conn()
        cursor = conn.cursor()

        # Create FTS5 table for full-text search
        fts_table = self._table_name("fts")
        cursor.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS {fts_table} USING fts5(
                concept_id,
                chunk_index,
                text,
                heading,
                type,
                status,
                trust_level,
                tokenize="unicode61 remove_diacritics 2"
            )
        """)

        # Create metadata table for concepts
        meta_table = self._table_name("metadata")
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {meta_table} (
                concept_id TEXT PRIMARY KEY,
                type TEXT,
                title TEXT,
                description TEXT,
                status TEXT,
                trust_level TEXT,
                stale BOOLEAN,
                stale_days INTEGER,
                tags TEXT,
                frontmatter_blob BLOB
            )
        """)

        # Create vector table
        vec_table = self._table_name("vectors")
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {vec_table} (
                concept_id TEXT,
                chunk_index INTEGER,
                embedding BLOB,
                PRIMARY KEY (concept_id, chunk_index)
            )
        """)

        # Create index on FTS table for filtering
        cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{fts_table}_concept 
            ON {fts_table}(concept_id)
        """)
        cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{fts_table}_type 
            ON {fts_table}(type)
        """)
        cursor.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{fts_table}_status 
            ON {fts_table}(status)
        """)

        conn.commit()

    def clear(self) -> None:
        """Clear all data from the index."""
        conn = self._ensure_conn()
        cursor = conn.cursor()

        for suffix in ["fts", "metadata", "vectors"]:
            table = self._table_name(suffix)
            cursor.execute(f"DELETE FROM {table}")

        conn.commit()

    def index_bundle(self, bundle: Bundle) -> None:
        """Index all concepts in a bundle.

        Args:
            bundle: The bundle to index
        """
        self.clear()

        conn = self._ensure_conn()
        cursor = conn.cursor()

        fts_table = self._table_name("fts")
        meta_table = self._table_name("metadata")
        vec_table = self._table_name("vectors")

        # Index each concept
        for concept_id, concept in bundle.concepts.items():
            # Skip reserved files (shouldn't be in concepts dict, but just in case)
            if concept.is_reserved:
                continue

            trust_level = derive_trust_level(concept)
            stale = is_stale(concept)
            stale_days = None
            if stale:
                from markrag.okf import get_stale_days
                stale_days = get_stale_days(concept)

            # Store metadata
            frontmatter_blob = pickle.dumps(concept.frontmatter.to_dict())
            tags_str = ",".join(concept.frontmatter.tags)

            cursor.execute(f"""
                INSERT INTO {meta_table} (
                    concept_id, type, title, description, status,
                    trust_level, stale, stale_days, tags, frontmatter_blob
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                concept_id,
                concept.frontmatter.type,
                concept.frontmatter.title,
                concept.frontmatter.description,
                concept.frontmatter.status,
                trust_level,
                stale,
                stale_days,
                tags_str,
                frontmatter_blob,
            ))

            # Get all chunks for this concept
            body_chunks = chunk_concept(concept)
            desc_chunks = chunk_description(concept)
            all_chunks = desc_chunks + body_chunks

            for chunk in all_chunks:
                # Insert into FTS
                cursor.execute(f"""
                    INSERT INTO {fts_table} (
                        concept_id, chunk_index, text, heading, type, status, trust_level
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    concept_id,
                    chunk.chunk_index,
                    chunk.text,
                    chunk.heading,
                    concept.frontmatter.type,
                    chunk.status,
                    chunk.trust_level,
                ))

                # Store vector placeholder (will be updated with actual embeddings)
                cursor.execute(f"""
                    INSERT INTO {vec_table} (
                        concept_id, chunk_index, embedding
                    ) VALUES (?, ?, ?)
                """, (
                    concept_id,
                    chunk.chunk_index,
                    None,  # Placeholder
                ))

        conn.commit()

    def update_vectors(
        self, concept_id: str, embeddings: dict[int, np.ndarray]
    ) -> None:
        """Update vector embeddings for a concept's chunks.

        Args:
            concept_id: The concept ID
            embeddings: Dict mapping chunk_index to embedding array
        """
        conn = self._ensure_conn()
        cursor = conn.cursor()
        vec_table = self._table_name("vectors")

        for chunk_index, embedding in embeddings.items():
            # Convert numpy array to bytes
            embedding_bytes = pickle.dumps(embedding)
            cursor.execute(f"""
                UPDATE {vec_table} 
                SET embedding = ? 
                WHERE concept_id = ? AND chunk_index = ?
            """, (embedding_bytes, concept_id, chunk_index))

        conn.commit()

    def fts_search(
        self,
        query: str,
        type_filter: str | None = None,
        status_filter: str | None = None,
        top_k: int = 50,
    ) -> list[dict[str, Any]]:
        """Perform FTS5 search.

        Args:
            query: Search query
            type_filter: Filter by concept type
            status_filter: Filter by status
            top_k: Number of results to return

        Returns:
            List of results with concept_id, score, and metadata
        """
        conn = self._ensure_conn()
        cursor = conn.cursor()
        fts_table = self._table_name("fts")

        # Build query
        where_clauses = []
        params: list[Any] = []

        if type_filter:
            where_clauses.append("type = ?")
            params.append(type_filter)

        if status_filter and status_filter != "any":
            where_clauses.append("status = ?")
            params.append(status_filter)

        where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"

        # Use FTS5 match with NEAR for phrase matching
        # The query syntax supports AND, OR, NOT, NEAR, quotes for phrases
        fts_query = f"{fts_table} q='{query}'"

        sql = f"""
            SELECT 
                concept_id,
                chunk_index,
                text,
                heading,
                type,
                status,
                trust_level,
                bm25({fts_table}, 1.0, 1.0) as score
            FROM {fts_table}
            WHERE {where_sql}
            ORDER BY score DESC
            LIMIT ?
        """
        params.extend([top_k])

        cursor.execute(sql, params)

        results = []
        for row in cursor.fetchall():
            results.append({
                "concept_id": row["concept_id"],
                "chunk_index": row["chunk_index"],
                "text": row["text"],
                "heading": row["heading"],
                "type": row["type"],
                "status": row["status"],
                "trust_level": row["trust_level"],
                "score": row["score"],
                "source": "fts",
            })

        return results

    def vector_search(
        self,
        query_embedding: np.ndarray,
        type_filter: str | None = None,
        status_filter: str | None = None,
        top_k: int = 50,
    ) -> list[dict[str, Any]]:
        """Perform vector similarity search.

        Args:
            query_embedding: Query embedding vector
            type_filter: Filter by concept type
            status_filter: Filter by status
            top_k: Number of results to return

        Returns:
            List of results with concept_id, score (cosine similarity), and metadata
        """
        conn = self._ensure_conn()
        cursor = conn.cursor()
        vec_table = self._table_name("vectors")
        fts_table = self._table_name("fts")

        # First, get all vectors that match the filters
        where_clauses = []
        params: list[Any] = []

        if type_filter:
            where_clauses.append(f"v.concept_id IN (SELECT concept_id FROM {fts_table} WHERE type = ?)")
            params.append(type_filter)

        if status_filter and status_filter != "any":
            where_clauses.append(f"v.concept_id IN (SELECT concept_id FROM {fts_table} WHERE status = ?)")
            params.append(status_filter)

        where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"

        # Get all embeddings and their metadata
        sql = f"""
            SELECT 
                v.concept_id,
                v.chunk_index,
                v.embedding,
                f.heading,
                f.type,
                f.status,
                f.trust_level
            FROM {vec_table} v
            JOIN {fts_table} f ON v.concept_id = f.concept_id AND v.chunk_index = f.chunk_index
            WHERE v.embedding IS NOT NULL AND {where_sql}
        """

        cursor.execute(sql, params)

        # Calculate cosine similarity for each
        results = []
        query_norm = np.linalg.norm(query_embedding)

        for row in cursor.fetchall():
            embedding_bytes = row["embedding"]
            try:
                embedding = pickle.loads(embedding_bytes)
                doc_norm = np.linalg.norm(embedding)
                if doc_norm == 0 or query_norm == 0:
                    similarity = 0.0
                else:
                    similarity = float(np.dot(query_embedding, embedding) / (query_norm * doc_norm))

                results.append({
                    "concept_id": row["concept_id"],
                    "chunk_index": row["chunk_index"],
                    "heading": row["heading"],
                    "type": row["type"],
                    "status": row["status"],
                    "trust_level": row["trust_level"],
                    "score": similarity,
                    "source": "vector",
                })
            except Exception:
                continue

        # Sort by similarity (descending)
        results.sort(key=lambda x: x["score"], reverse=True)

        return results[:top_k]

    def hybrid_search(
        self,
        query: str,
        query_embedding: np.ndarray | None = None,
        type_filter: str | None = None,
        status_filter: str | None = None,
        top_k: int = 8,
        k1: int = 60,
    ) -> list[dict[str, Any]]:
        """Perform hybrid search with Reciprocal Rank Fusion (RRF).

        Args:
            query: Text query for FTS
            query_embedding: Vector for vector search (optional)
            type_filter: Filter by concept type
            status_filter: Filter by status
            top_k: Number of final results
            k1: RRF constant (default 60)

        Returns:
            Fused results from FTS and vector search
        """
        # Get FTS results
        fts_results = self.fts_search(
            query, type_filter, status_filter, top_k=top_k * 2
        )

        # Get vector results if embedding provided
        vector_results: list[dict[str, Any]] = []
        if query_embedding is not None:
            vector_results = self.vector_search(
                query_embedding, type_filter, status_filter, top_k=top_k * 2
            )

        # Fuse results using RRF
        all_results = fts_results + vector_results

        # Create scoring dict: concept_id -> {source -> rank}
        concept_rankings: dict[str, dict[str, int]] = {}

        for rank, result in enumerate(all_results, 1):
            concept_id = result["concept_id"]
            source = result["source"]

            if concept_id not in concept_rankings:
                concept_rankings[concept_id] = {}

            # Only keep the best rank per source
            if source not in concept_rankings[concept_id] or rank < concept_rankings[concept_id][source]:
                concept_rankings[concept_id][source] = rank

        # Calculate RRF score for each concept
        scored_results = []
        for concept_id, source_ranks in concept_rankings.items():
            score = 0.0
            for source_rank in source_ranks.values():
                score += 1.0 / (k1 + source_rank)

            # Get metadata from one of the results (they should be similar)
            sample_result = next(r for r in all_results if r["concept_id"] == concept_id)

            scored_results.append({
                **sample_result,
                "rrf_score": score,
            })

        # Sort by RRF score
        scored_results.sort(key=lambda x: x["rrf_score"], reverse=True)

        return scored_results[:top_k]

    def get_metadata(self, concept_id: str) -> dict[str, Any] | None:
        """Get metadata for a concept."""
        conn = self._ensure_conn()
        cursor = conn.cursor()
        meta_table = self._table_name("metadata")

        cursor.execute(f"SELECT * FROM {meta_table} WHERE concept_id = ?", (concept_id,))
        row = cursor.fetchone()

        if row is None:
            return None

        return {
            "concept_id": row["concept_id"],
            "type": row["type"],
            "title": row["title"],
            "description": row["description"],
            "status": row["status"],
            "trust_level": row["trust_level"],
            "stale": bool(row["stale"]),
            "stale_days": row["stale_days"],
            "tags": row["tags"].split(",") if row["tags"] else [],
            "frontmatter": pickle.loads(row["frontmatter_blob"]) if row["frontmatter_blob"] else {},
        }

    def get_all_metadata(self) -> list[dict[str, Any]]:
        """Get metadata for all concepts."""
        conn = self._ensure_conn()
        cursor = conn.cursor()
        meta_table = self._table_name("metadata")

        cursor.execute(f"SELECT * FROM {meta_table}")

        results = []
        for row in cursor.fetchall():
            results.append({
                "concept_id": row["concept_id"],
                "type": row["type"],
                "title": row["title"],
                "description": row["description"],
                "status": row["status"],
                "trust_level": row["trust_level"],
                "stale": bool(row["stale"]),
                "stale_days": row["stale_days"],
                "tags": row["tags"].split(",") if row["tags"] else [],
            })

        return results

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        self._ensure_conn()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Import re for the table name sanitization
import re

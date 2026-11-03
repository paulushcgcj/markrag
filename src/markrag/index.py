"""SQLite FTS5 + vector index for OKF concepts.

The index is a *derived artifact*: the Markdown files are the single source of
truth and the database can always be thrown away and rebuilt. It stores

- an FTS5 table over every chunk's text, and
- the embedding vector for every chunk, alongside the chunk text.

Retrieval fuses the two ranked lists with Reciprocal Rank Fusion. Metadata
filters (type, tags, status) are applied *before* scoring, so the candidate set
is narrowed first.
"""

import hashlib
import os
import pickle
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np

from markrag.chunking import Chunk, chunk_all
from markrag.embeddings import EmbeddingProvider
from markrag.okf import Bundle, is_reserved_filename

# Reciprocal Rank Fusion constant. 60 is the value used by the original
# "Reciprocal Rank Fusion outperforms Condorcet and individual PRF" paper.
RRF_K = 60

# Number of candidates pulled from each ranked list before fusion.
CANDIDATE_POOL = 50

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


@dataclass(frozen=True)
class SearchHit:
    """A single scored concept returned by :meth:`KnowledgeIndex.search`."""

    concept_id: str
    chunk_index: int
    kind: str
    heading: str | None
    text: str
    excerpt_text: str
    anchor: str | None
    score: float
    fts_rank: int | None
    vector_rank: int | None


@dataclass(frozen=True)
class SearchResponse:
    """The outcome of a hybrid search."""

    hits: tuple[SearchHit, ...]
    fts_available: bool
    vector_available: bool
    candidate_count: int


def _fts_query(query: str) -> str | None:
    """Build a safe FTS5 MATCH expression from a free-text query.

    Every token is quoted so that FTS5 operators typed by a user cannot cause a
    syntax error, and the full phrase is added as an extra term so that exact
    wording outranks scattered token matches.

    Args:
        query: Raw user query.

    Returns:
        The MATCH expression, or ``None`` when the query has no usable tokens.
    """
    tokens = [token.lower() for token in _TOKEN_RE.findall(query)]
    if not tokens:
        return None

    terms = [f'"{token}"' for token in dict.fromkeys(tokens)]
    if len(tokens) > 1:
        terms.insert(0, '"' + " ".join(tokens) + '"')
    return " OR ".join(terms)


def _normalise(vector: np.ndarray) -> np.ndarray:
    """Return a unit-length vector, or a zero vector when the norm is zero."""
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return np.zeros_like(vector)
    return vector / norm


def bundle_fingerprint(bundle: Bundle) -> str:
    """Compute a fingerprint of the bundle's Markdown files.

    The fingerprint changes whenever a document is added, removed, or edited,
    which is what tells the index that it must be rebuilt.

    Reserved files are excluded: they hold navigation and history rather than
    content, and markrag regenerates them on every write.

    Args:
        bundle: The bundle to fingerprint.

    Returns:
        A hex digest of the bundle contents.
    """
    digest = hashlib.sha256()
    for path in sorted(bundle.root_path.rglob("*.md")):
        if is_reserved_filename(path.name):
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        digest.update(str(path.relative_to(bundle.root_path)).encode("utf-8"))
        digest.update(str(stat.st_mtime_ns).encode("utf-8"))
        digest.update(str(stat.st_size).encode("utf-8"))
    return digest.hexdigest()


class KnowledgeIndex:
    """SQLite-backed hybrid index for a single OKF bundle."""

    def __init__(
        self,
        db_path: Path | str,
        bundle_name: str,
        provider: EmbeddingProvider | None = None,
        embedding_signature: str = "",
    ) -> None:
        """Initialize the index.

        Args:
            db_path: Path to the SQLite database file.
            bundle_name: Name of the bundle, used to prefix table names.
            provider: Embedding provider. When omitted, vectors are not built
                and retrieval falls back to keyword search.
            embedding_signature: Identifies the embedding configuration. The
                index is rebuilt when it changes, so switching embeddings on
                or choosing a different model takes effect.
        """
        self.db_path = Path(db_path)
        safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", bundle_name)
        self._prefix = f"{safe_name}_" if safe_name else ""
        self._provider = provider
        self.embedding_signature = embedding_signature
        self._conn: sqlite3.Connection | None = None

    # -- schema -----------------------------------------------------------

    @property
    def chunks_table(self) -> str:
        """Name of the table holding chunk text, metadata, and vectors."""
        return f"{self._prefix}chunks"

    @property
    def fts_table(self) -> str:
        """Name of the FTS5 table holding indexed chunk text."""
        return f"{self._prefix}fts"

    @property
    def meta_table(self) -> str:
        """Name of the table holding index metadata."""
        return f"{self._prefix}meta"

    def _connect(self) -> sqlite3.Connection:
        """Open the database connection, creating it if needed."""
        if self._conn is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._conn = conn
        return self._conn

    def initialize(self) -> None:
        """Create the schema if it does not exist yet."""
        conn = self._connect()
        conn.executescript(
            f"""
            CREATE TABLE IF NOT EXISTS {self.chunks_table} (
                concept_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                kind TEXT NOT NULL,
                heading TEXT,
                text TEXT NOT NULL,
                type TEXT NOT NULL,
                status TEXT NOT NULL,
                trust_level TEXT NOT NULL,
                tags TEXT NOT NULL,
                embedding BLOB,
                PRIMARY KEY (concept_id, chunk_index)
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS {self.fts_table} USING fts5(
                concept_id UNINDEXED,
                chunk_index UNINDEXED,
                heading,
                text,
                tokenize="unicode61 remove_diacritics 2"
            );

            CREATE TABLE IF NOT EXISTS {self.meta_table} (
                key TEXT PRIMARY KEY,
                value TEXT
            );
            """
        )
        conn.commit()

    # -- building ---------------------------------------------------------

    def clear(self) -> None:
        """Remove every indexed row.

        The recorded fingerprint is dropped too: an index with no rows must
        never look current, or the next search would find nothing and stop
        rebuilding.
        """
        conn = self._connect()
        self.initialize()
        conn.execute(f"DELETE FROM {self.chunks_table}")
        conn.execute(f"DELETE FROM {self.fts_table}")
        conn.execute(f"DELETE FROM {self.meta_table}")
        conn.commit()

    def _embed(self, texts: list[str]) -> list[bytes | None]:
        """Embed chunk texts, returning ``None`` for each on failure."""
        if self._provider is None or not texts:
            return [None] * len(texts)

        try:
            vectors = self._provider.embed_batch(texts)
        except Exception:
            # Embeddings are an enhancement: a missing model or an unreachable
            # endpoint must not stop keyword retrieval from working.
            return [None] * len(texts)

        encoded: list[bytes | None] = []
        for vector in vectors:
            encoded.append(
                pickle.dumps(np.asarray(vector, dtype=np.float32), protocol=pickle.HIGHEST_PROTOCOL)
            )
        # A provider that returns fewer vectors than texts leaves the tail
        # without an embedding; those rows stay keyword-only.
        while len(encoded) < len(texts):
            encoded.append(None)
        return encoded

    def build(self, bundle: Bundle) -> bool:
        """Rebuild the index from a bundle.

        Args:
            bundle: The bundle to index.

        Returns:
            ``True`` when vectors were produced, ``False`` when the embedding
            provider was unavailable and only keyword retrieval is possible.
        """
        self.initialize()
        self.clear()

        conn = self._connect()
        chunks: list[Chunk] = []
        for concept_id in sorted(bundle.concepts):
            concept = bundle.concepts[concept_id]
            chunks.extend(chunk_all(concept))

        embeddings = self._embed([chunk.text for chunk in chunks])
        vectors_available = any(embedding is not None for embedding in embeddings)

        chunk_rows = []
        fts_rows = []
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            tags = "|".join([f"|{tag}|" for tag in chunk.frontmatter.get("tags", [])])
            chunk_rows.append(
                (
                    chunk.concept_id,
                    chunk.chunk_index,
                    chunk.kind,
                    chunk.heading,
                    chunk.text,
                    chunk.frontmatter.get("type", ""),
                    chunk.status,
                    chunk.trust_level,
                    tags,
                    embedding,
                )
            )
            fts_rows.append((chunk.concept_id, chunk.chunk_index, chunk.heading or "", chunk.text))

        if chunk_rows:
            conn.executemany(
                f"""
                INSERT OR REPLACE INTO {self.chunks_table}
                    (concept_id, chunk_index, kind, heading, text, type, status,
                     trust_level, tags, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                chunk_rows,
            )
            conn.executemany(
                f"""
                INSERT INTO {self.fts_table} (concept_id, chunk_index, heading, text)
                VALUES (?, ?, ?, ?)
                """,
                fts_rows,
            )

        conn.execute(
            f"INSERT OR REPLACE INTO {self.meta_table} (key, value) VALUES (?, ?)",
            ("fingerprint", self._index_fingerprint(bundle)),
        )
        conn.execute(
            f"INSERT OR REPLACE INTO {self.meta_table} (key, value) VALUES (?, ?)",
            ("vectors_available", "1" if vectors_available else "0"),
        )
        conn.commit()
        return vectors_available

    # -- searching --------------------------------------------------------

    def _filter_clause(
        self,
        type_: str | None,
        tags: list[str] | None,
        status: str | None,
    ) -> tuple[str, list[Any]]:
        """Build the metadata pre-filter clause and its parameters."""
        clauses: list[str] = []
        params: list[Any] = []

        if type_:
            clauses.append("c.type = ?")
            params.append(type_)

        if status and status != "any":
            clauses.append("c.status = ?")
            params.append(status)

        for tag in tags or []:
            clauses.append("c.tags LIKE ?")
            params.append(f"%|{tag}|%")

        return (" AND ".join(clauses) if clauses else "1=1"), params

    def _fts_candidates(
        self,
        query: str,
        type_: str | None,
        tags: list[str] | None,
        status: str | None,
        limit: int,
    ) -> list[sqlite3.Row]:
        """Run the FTS5 half of the search."""
        match = _fts_query(query)
        if match is None:
            return []

        conn = self._connect()
        self.initialize()
        where_sql, params = self._filter_clause(type_, tags, status)
        # FTS5 only accepts the MATCH operator against the table name itself,
        # so the table must not be aliased here.
        sql = f"""
            SELECT {self.fts_table}.concept_id, {self.fts_table}.chunk_index,
                   {self.fts_table}.heading, {self.fts_table}.text,
                   c.kind, c.type, c.status, c.trust_level,
                   bm25({self.fts_table}) AS rank
            FROM {self.fts_table}
            JOIN {self.chunks_table} c
              ON c.concept_id = {self.fts_table}.concept_id
             AND c.chunk_index = {self.fts_table}.chunk_index
            WHERE {self.fts_table} MATCH ? AND {where_sql}
            ORDER BY rank
            LIMIT ?
        """
        rows = conn.execute(sql, [match, *params, limit]).fetchall()
        return list(rows)

    def _vector_candidates(
        self,
        query: str,
        type_: str | None,
        tags: list[str] | None,
        status: str | None,
        limit: int,
    ) -> list[sqlite3.Row]:
        """Run the vector half of the search using cosine similarity."""
        if self._provider is None:
            return []

        conn = self._connect()
        self.initialize()
        where_sql, params = self._filter_clause(type_, tags, status)
        sql = f"""
            SELECT c.concept_id, c.chunk_index, c.kind, c.heading, c.text,
                   c.type, c.status, c.trust_level, c.embedding
            FROM {self.chunks_table} c
            WHERE {where_sql} AND c.embedding IS NOT NULL
        """
        rows = list(conn.execute(sql, params).fetchall())
        if not rows:
            return []

        try:
            query_vector = _normalise(np.asarray(self._provider.embed(query), dtype=np.float32))
        except Exception:
            return []
        if query_vector.size == 0 or not float(np.linalg.norm(query_vector)):
            return []

        scored: list[tuple[float, sqlite3.Row]] = []
        for row in rows:
            vector = pickle.loads(row["embedding"])  # noqa: S301 - own database
            normed = _normalise(np.asarray(vector, dtype=np.float32))
            if normed.shape != query_vector.shape:
                continue
            scored.append((float(np.dot(normed, query_vector)), row))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [row for _, row in scored[:limit]]

    def search(
        self,
        query: str,
        type_: str | None = None,
        tags: list[str] | None = None,
        status: str | None = None,
        top_k: int = 8,
    ) -> SearchResponse:
        """Run a hybrid search.

        Args:
            query: Free-text query.
            type_: Restrict to a concept type.
            tags: Restrict to concepts carrying every listed tag.
            status: Restrict to a lifecycle status. ``"any"`` disables the filter.
            top_k: Maximum number of concepts to return.

        Returns:
            The ranked hits plus which halves of the search were available.
        """
        fts_rows = self._fts_candidates(query, type_, tags, status, CANDIDATE_POOL)
        vector_rows = self._vector_candidates(query, type_, tags, status, CANDIDATE_POOL)

        fts_ranked = self._best_per_concept(fts_rows)
        vector_ranked = self._best_per_concept(vector_rows)

        fts_ranks = {row["concept_id"]: position for position, row in enumerate(fts_ranked, 1)}
        vector_ranks = {
            row["concept_id"]: position for position, row in enumerate(vector_ranked, 1)
        }

        # The excerpt should come from a body chunk when one matched; title and
        # description chunks make poor excerpts.
        body_rows = self._best_body_per_concept([*fts_rows, *vector_rows])

        fused: list[tuple[float, int, int, str, sqlite3.Row, str]] = []
        for concept_id in set(fts_ranks) | set(vector_ranks):
            score = 0.0
            fts_rank = fts_ranks.get(concept_id)
            vector_rank = vector_ranks.get(concept_id)
            if fts_rank is not None:
                score += 1.0 / (RRF_K + fts_rank)
            if vector_rank is not None:
                score += 1.0 / (RRF_K + vector_rank)

            best_row = next(
                (row for row in fts_ranked if row["concept_id"] == concept_id),
                None,
            ) or next(row for row in vector_ranked if row["concept_id"] == concept_id)
            kind = str(best_row["kind"])
            fused.append(
                (
                    score,
                    fts_rank if fts_rank is not None else CANDIDATE_POOL + 1,
                    vector_rank if vector_rank is not None else CANDIDATE_POOL + 1,
                    concept_id,
                    best_row,
                    kind,
                )
            )

        # Highest fused score first. Ties prefer the keyword match, then the
        # semantic match, then the concept ID so ordering is deterministic.
        fused.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))

        hits: list[SearchHit] = []
        for score, _, _, concept_id, row, kind in fused[:top_k]:
            body_row = body_rows.get(concept_id, row)
            hits.append(
                SearchHit(
                    concept_id=concept_id,
                    chunk_index=int(row["chunk_index"]),
                    kind=kind,
                    heading=row["heading"] or None,
                    text=str(row["text"]),
                    excerpt_text=str(body_row["text"]),
                    anchor=(str(body_row["heading"]) if str(body_row["kind"]) == "body" else None),
                    score=score,
                    fts_rank=fts_ranks.get(concept_id),
                    vector_rank=vector_ranks.get(concept_id),
                )
            )

        return SearchResponse(
            hits=tuple(hits),
            fts_available=bool(fts_rows) or _fts_query(query) is None,
            vector_available=bool(vector_rows),
            candidate_count=len(set(fts_ranks) | set(vector_ranks)),
        )

    @staticmethod
    def _best_per_concept(rows: list[sqlite3.Row]) -> list[sqlite3.Row]:
        """Keep only the best-ranked chunk for each concept, preserving order."""
        seen: set[str] = set()
        best: list[sqlite3.Row] = []
        for row in rows:
            concept_id = str(row["concept_id"])
            if concept_id in seen:
                continue
            seen.add(concept_id)
            best.append(row)
        return best

    @staticmethod
    def _best_body_per_concept(rows: list[sqlite3.Row]) -> dict[str, sqlite3.Row]:
        """Pick the best body chunk per concept, falling back to any chunk."""
        best: dict[str, sqlite3.Row] = {}
        for row in rows:
            concept_id = str(row["concept_id"])
            existing = best.get(concept_id)
            if existing is None:
                best[concept_id] = row
            elif str(existing["kind"]) != "body" and str(row["kind"]) == "body":
                best[concept_id] = row
        return best

    # -- lifecycle --------------------------------------------------------

    def fingerprint(self) -> str | None:
        """Return the fingerprint recorded when the index was last built."""
        conn = self._connect()
        self.initialize()
        row = conn.execute(
            f"SELECT value FROM {self.meta_table} WHERE key = 'fingerprint'"
        ).fetchone()
        return str(row["value"]) if row else None

    def vectors_available(self) -> bool:
        """Return whether the index contains embedding vectors."""
        conn = self._connect()
        self.initialize()
        row = conn.execute(
            f"SELECT value FROM {self.meta_table} WHERE key = 'vectors_available'"
        ).fetchone()
        return bool(row and row["value"] == "1")

    def is_current(self, bundle: Bundle) -> bool:
        """Return whether the index matches the current bundle contents.

        The index is stale when the documents changed *or* when the embedding
        configuration changed, so turning embeddings on takes effect without
        having to delete the index by hand.
        """
        return self.fingerprint() == self._index_fingerprint(bundle)

    def _index_fingerprint(self, bundle: Bundle) -> str:
        """Fingerprint of the documents plus the embedding configuration."""
        return f"{bundle_fingerprint(bundle)}|{self.embedding_signature}"

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> Self:
        """Enter the context manager."""
        self.initialize()
        return self

    def __exit__(self, *_exc: object) -> None:
        """Leave the context manager."""
        self.close()


def default_index_path(bundle_name: str) -> Path:
    """Return the default on-disk location of a bundle's index.

    The index lives outside the wiki repository so it never shows up in a
    reviewable diff. ``MARKRAG_INDEX_DIR`` overrides the location, which keeps
    test runs isolated from the user's home directory.

    Args:
        bundle_name: Name of the bundle.

    Returns:
        Path to the SQLite file.
    """
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", bundle_name)
    base = os.path.expanduser(os.getenv("MARKRAG_INDEX_DIR", "~/.markrag"))
    return Path(base).expanduser() / f"{safe}.db"

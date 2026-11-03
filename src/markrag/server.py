"""MCP server for markrag - OKF-backed RAG for per-repo wikis.

Tools exposed to agents:

- ``okf_index``  - tier 0, the bundle map
- ``okf_search`` - tier 1, hybrid keyword + vector search
- ``okf_read``   - tier 2, a full document or one section
- ``okf_write``  - create or update a concept
- ``okf_status`` - bundle statistics for orientation

Every tool obeys the same disclosure discipline: no bodies outside
``okf_read``, excerpts capped at roughly forty tokens, and a trust line on
anything that surfaces a document.
"""

import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from markrag import __version__
from markrag.bootstrap import bootstrap_wiki
from markrag.bundle_map import write_bundle_map
from markrag.embeddings import EmbeddingProvider, get_embedding_provider, provider_signature
from markrag.index import KnowledgeIndex, default_index_path
from markrag.okf import (
    RESERVED_FILES,
    SHELL_FILENAMES,
    Bundle,
    Concept,
    Frontmatter,
    Generated,
    Source,
    append_log_entry,
    derive_trust_level,
    get_stale_days,
    is_stale,
    trust_line,
    wiki_path_for_repo,
)

# Excerpts outside okf_read are capped at roughly forty tokens.
EXCERPT_TOKENS = 40

# Global state.
_bundle: Bundle | None = None
_bundle_path: Path | None = None
_index: KnowledgeIndex | None = None
_index_bundle_path: Path | None = None
_embedding_provider: EmbeddingProvider | None = None
_embedding_provider_loaded = False

_CONCEPT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")

# Files the server owns; a tool call may never write them.
RESERVED_STEMS = {name.removesuffix(".md").lower() for name in RESERVED_FILES}


# ---------------------------------------------------------------------------
# Bundle and index management
# ---------------------------------------------------------------------------


def _repo_path(repo_path: str | Path | None = None) -> Path:
    """Resolve the repository path from an argument, the environment, or cwd."""
    raw = repo_path if repo_path is not None else os.getenv("MARKRAG_REPO", ".")
    return Path(raw).expanduser().resolve()


def _get_bundle(
    repo_path: str | Path | None = None,
    *,
    allow_bootstrap: bool = False,
) -> Bundle:
    """Get or load the bundle for the current repository.

    The wiki lives in a sibling ``<name>.wiki`` directory of the repository.
    When ``allow_bootstrap`` is set and the wiki does not exist yet, it is
    created and seeded before the bundle is loaded.

    Args:
        repo_path: Repository path. Defaults to ``MARKRAG_REPO`` or the cwd.
        allow_bootstrap: Bootstrap the wiki when it is missing.

    Returns:
        The loaded bundle.
    """
    global _bundle, _bundle_path

    if repo_path is None and _bundle is not None:
        return _reload_if_stale(_bundle)

    repo = _repo_path(repo_path)
    wiki = wiki_path_for_repo(repo)

    if _bundle is not None and _bundle_path == wiki:
        return _reload_if_stale(_bundle)

    if allow_bootstrap and not any(wiki.rglob("*.md")):
        # Only a wiki with no documents at all is bootstrapped. A wiki that
        # already has content is left exactly as it is.
        bootstrap_wiki(repo, wiki, version=__version__)

    bundle = Bundle(root_path=wiki)
    try:
        bundle.load()
    except FileNotFoundError:
        bundle = Bundle(root_path=wiki)

    _bundle = bundle
    _bundle_path = wiki
    return bundle


def _reload_if_stale(bundle: Bundle) -> Bundle:
    """Reload a bundle when its files changed on disk.

    The wiki is human-curated ground truth: a document edited outside the
    server must be visible to the next read.
    """
    if bundle.is_stale():
        bundle.load()
    return bundle


def _provider() -> EmbeddingProvider | None:
    """Return the embedding provider, constructing it once per process."""
    global _embedding_provider, _embedding_provider_loaded
    if not _embedding_provider_loaded:
        _embedding_provider = get_embedding_provider()
        _embedding_provider_loaded = True
    return _embedding_provider


def _get_index(bundle: Bundle) -> KnowledgeIndex:
    """Return the search index for a bundle, rebuilding it when stale.

    The index is derived from the Markdown files, so it can always be thrown
    away. A fingerprint of the bundle's files decides whether a rebuild is
    needed.
    """
    global _index, _index_bundle_path

    bundle_name = bundle.root_path.name.removesuffix(".wiki")
    if _index is None or _index_bundle_path != bundle.root_path:
        provider = _provider()
        _index = KnowledgeIndex(
            default_index_path(bundle_name),
            bundle_name,
            provider=provider,
            embedding_signature=provider_signature(provider),
        )
        _index_bundle_path = bundle.root_path

    if not _index.is_current(bundle):
        _index.build(bundle)

    return _index


def _invalidate_index() -> None:
    """Force the index to be rebuilt on the next search."""
    global _index
    if _index is not None:
        _index.clear()


def _close_index() -> None:
    """Close the search index and release its database connection."""
    global _index, _index_bundle_path
    if _index is not None:
        _index.close()
    _index = None
    _index_bundle_path = None


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _excerpt(text: str, max_tokens: int = EXCERPT_TOKENS) -> str:
    """Cap text to roughly ``max_tokens`` whitespace-delimited tokens."""
    if not text:
        return ""
    words = text.split()
    if len(words) <= max_tokens:
        return text
    return " ".join(words[:max_tokens]) + "…"


def _stale_warning(concept: Concept) -> str | None:
    """Return a staleness warning for a concept, if it is past stale_after."""
    if not is_stale(concept):
        return None
    stale_days = get_stale_days(concept)
    if stale_days:
        return f"WARNING: Document is {stale_days} days past stale_after"
    return "WARNING: Document is stale"


def _concept_summary(concept: Concept, concept_id: str) -> dict[str, Any]:
    """Build the shared metadata block for a concept."""
    frontmatter = concept.frontmatter
    return {
        "concept_id": concept_id,
        "title": concept.title or concept_id,
        "description": frontmatter.description or "",
        "type": frontmatter.type,
        "status": frontmatter.status,
        "trust_level": derive_trust_level(concept),
        "stale": is_stale(concept),
        "trust_line": trust_line(concept),
    }


def _reserved_message(stem: str) -> str:
    """Explain why a reserved filename cannot be written."""
    if f"{stem.lower()}.md" in SHELL_FILENAMES:
        return f"'{stem}.md' is user-facing wiki chrome and cannot be written"
    return f"'{stem}.md' is server-managed and cannot be written"


def _validate_concept_id(concept_id: str) -> str | None:
    """Validate a concept ID, returning an error message when it is unsafe."""
    if not concept_id or not concept_id.strip():
        return "concept_id must not be empty"

    if concept_id.startswith("/") or concept_id.startswith("\\"):
        return "concept_id must be relative to the bundle root"

    if re.match(r"^[A-Za-z]:", concept_id):
        return "concept_id must be relative to the bundle root"

    parts = concept_id.replace("\\", "/").split("/")
    if any(part in ("", ".", "..") for part in parts):
        return "concept_id must not contain empty, '.' or '..' path segments"

    # Reserved names are reserved at every level of the hierarchy, and the
    # comparison is case-insensitive: on a case-insensitive filesystem "Index"
    # and "index.md" are the same file.
    for part in parts:
        if part.lower() in RESERVED_STEMS:
            return _reserved_message(part)

    if not _CONCEPT_ID_RE.match(concept_id):
        return "concept_id may only contain letters, digits, '.', '_', '-' and '/'"

    if concept_id.endswith(".md"):
        return "concept_id must not include the '.md' extension"

    return None


def _log_directories(concept_id: str) -> tuple[str, ...]:
    """Return the log files a concept's change belongs in.

    A change is recorded in the log of the directory that holds the concept,
    in the log of each of that directory's ancestors, and in the wiki's root
    log. The empty string stands for the wiki root.

    Args:
        concept_id: The concept ID, a path relative to the bundle root.

    Returns:
        The directories whose ``log.md`` receives the entry.
    """
    parts = concept_id.split("/")[:-1]
    directories = [""]
    for depth in range(1, len(parts) + 1):
        directories.append("/".join(parts[:depth]))
    return tuple(directories)


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "markrag",
    instructions="""
You have access to a curated knowledge wiki for the current repository via markrag.

## Usage Rules

1. **Tier 0 (okf_index)**: Get the bundle map - concept IDs, titles, descriptions, status. No bodies.
2. **Tier 1 (okf_search)**: Hybrid search with ~40-token excerpts. Use for discovery.
3. **Tier 2 (okf_read)**: Full document content. Use when you need the complete information.

## Trust Model

Every document carries a trust level:
- `unverified`: no `verified` field present
- `machine-confirmed`: only agent/process verifiers (`producer/version`, `process:<id>`)
- `human-reviewed`: at least one human verifier (`human:<id>`)

Documents may also be flagged as stale when they are past their `stale_after` date.

## When to use what

- Start with `okf_index` to understand the wiki structure
- Use `okf_search` to find relevant concepts
- Use `okf_read` only when you need the full content
- Always check the trust line and staleness warning before relying on information

## Important

- Generated documents are always stamped with `generated.by` and `status: draft`
- Verification is a human act only - never mark content as verified yourself
- Never fabricate sources, verification events, or timestamps
""",
)


@mcp.tool()
def okf_index(path: str | None = None) -> dict[str, Any]:
    """Tier 0. Bundle map for the root or a subdirectory path.

    Returns each entry as: concept_id, title, description (one line), status,
    trust level and staleness. No bodies.

    Args:
        path: Optional subdirectory path within the bundle (e.g. 'processes').

    Returns:
        Dictionary with 'concepts' list, the resolved 'path', and a 'count'.
    """
    bundle = _get_bundle()

    if path:
        prefix = path.strip("/") + "/"
        entries = [
            (concept_id, concept)
            for concept_id, concept in bundle.concepts.items()
            if concept_id.startswith(prefix)
        ]
    else:
        entries = list(bundle.concepts.items())

    concepts = [
        _concept_summary(concept, concept_id)
        for concept_id, concept in sorted(entries, key=lambda item: item[0])
    ]

    return {
        "path": path or ".",
        "concepts": concepts,
        "count": len(concepts),
    }


@mcp.tool()
def okf_search(
    query: str,
    type_: str | None = None,
    tags: list[str] | None = None,
    status: str | None = "any",
    top_k: int = 8,
) -> dict[str, Any]:
    """Tier 1. Hybrid keyword and vector search with 1-hop link expansion.

    Per hit returns: concept_id, title, description, trust level, stale flag,
    the best-matching section anchor, and a short excerpt of roughly forty
    tokens. Plus 'related': concept IDs and descriptions reachable in one hop
    through the document links. Never full bodies.

    Args:
        query: Search query string.
        type_: Restrict results to a concept type.
        tags: Restrict results to concepts carrying every listed tag.
        status: Restrict to 'draft', 'stable', 'deprecated', or 'any'.
        top_k: Maximum number of concepts to return.

    Returns:
        Dictionary with 'results' list, 'filters', and retrieval diagnostics.
    """
    bundle = _get_bundle()

    if top_k <= 0:
        top_k = 1

    index = _get_index(bundle)
    response = index.search(
        query,
        type_=type_,
        tags=tags,
        status=status,
        top_k=top_k,
    )

    results: list[dict[str, Any]] = []
    for hit in response.hits:
        concept = bundle.get_concept(hit.concept_id)
        if concept is None:
            continue

        summary = _concept_summary(concept, hit.concept_id)
        excerpt_source = hit.excerpt_text or concept.frontmatter.description or ""
        anchor = hit.anchor

        related = [
            {"concept_id": linked_id, "description": description}
            for linked_id, description in bundle.get_1hop_links(hit.concept_id)
        ]

        results.append(
            {
                **summary,
                "section": anchor,
                "excerpt": _excerpt(excerpt_source),
                "related": related,
                "score": round(hit.score, 6),
                "matched_by": _match_sources(hit),
            }
        )

    return {
        "query": query,
        "results": results,
        "count": len(results),
        "filters": {"type": type_, "tags": tags, "status": status},
        "retrieval": {
            "keyword": response.fts_available,
            "vector": response.vector_available,
            "candidates": response.candidate_count,
        },
    }


def _match_sources(hit: Any) -> list[str]:
    """Describe which retrieval halves contributed to a hit."""
    sources: list[str] = []
    if hit.fts_rank is not None:
        sources.append("keyword")
    if hit.vector_rank is not None:
        sources.append("vector")
    return sources


@mcp.tool()
def okf_read(
    concept_id: str,
    section: str | None = None,
) -> dict[str, Any]:
    """Tier 2. Full document, or a single ## section when 'section' is given.

    The section is matched against the heading text exactly. Output: trust
    line, staleness warning if any, frontmatter summary with sources, then the
    Markdown content.

    Args:
        concept_id: The concept ID to read.
        section: Optional section heading to extract.

    Returns:
        Dictionary with the document content and its trust metadata.
    """
    bundle = _get_bundle()

    concept = bundle.get_concept(concept_id)
    if concept is None:
        reserved_stems = {name.removesuffix(".md") for name in RESERVED_FILES}
        if concept_id in reserved_stems:
            return {
                "error": f"{concept_id} is a reserved navigation file, not a concept",
                "concept_id": concept_id,
            }
        return {
            "error": f"Concept not found: {concept_id}",
            "concept_id": concept_id,
        }

    result: dict[str, Any] = {
        "concept_id": concept_id,
        "trust_level": derive_trust_level(concept),
        "trust_line": trust_line(concept),
        "stale": is_stale(concept),
        "stale_warning": _stale_warning(concept),
        "frontmatter": concept.frontmatter.to_dict(),
    }

    if concept.frontmatter.sources:
        result["sources"] = [
            {"id": source.id, "resource": source.resource, "title": source.title}
            for source in concept.frontmatter.sources
        ]

    if section:
        content = _extract_section(concept, section)
        result["section"] = section
        result["content"] = content
        if not content:
            result["error"] = f"Section '{section}' not found"
    else:
        result["content"] = concept.body

    return result


def _extract_section(concept: Concept, section: str) -> str:
    """Extract a single ``##`` section, matching the heading text exactly."""
    lines = concept.body.split("\n")
    collected: list[str] = []
    found = False

    for line in lines:
        match = re.match(r"^##\s+(.+?)\s*$", line)
        if match:
            if found:
                break
            if match.group(1) == section:
                found = True
                collected.append(line)
            continue
        if found:
            collected.append(line)

    return "\n".join(collected).strip()


@mcp.tool()
def okf_write(
    concept_id: str,
    title: str,
    description: str,
    type_: str,
    body_markdown: str,
    tags: list[str] | None = None,
    sources: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Create or update a concept.

    ALWAYS stamps generated{by, at} and status=draft. An update preserves the
    existing verification history and any frontmatter fields the caller does
    not manage; it never adds or forges verification. Every write records the
    change in the wiki's logs and regenerates every ``index.md`` from the
    frontmatter.

    Args:
        concept_id: The concept ID, a path relative to the bundle root.
        title: Concept title.
        description: One-line description.
        type_: Concept type.
        body_markdown: Markdown body content.
        tags: Optional list of tags.
        sources: Optional list of source dictionaries, each with a 'resource'.

    Returns:
        Dictionary with the written concept_id and path.
    """
    error = _validate_concept_id(concept_id)
    if error is not None:
        return {"error": error, "concept_id": concept_id, "status": "rejected"}

    source_objects: list[Source] = []
    for source in sources or []:
        resource = str(source.get("resource", "")).strip()
        if not resource:
            return {
                "error": f"source '{source.get('id', '')}' is missing a resource",
                "concept_id": concept_id,
                "status": "rejected",
            }
        source_objects.append(
            Source(
                id=str(source.get("id", "")),
                resource=resource,
                title=source.get("title"),
            )
        )

    bundle = _get_bundle()
    wiki_path = bundle.root_path
    wiki_path.mkdir(parents=True, exist_ok=True)

    file_path = wiki_path / f"{concept_id}.md"
    file_path.parent.mkdir(parents=True, exist_ok=True)

    existing = bundle.get_concept(concept_id)
    existing_frontmatter = existing.frontmatter if existing is not None else None

    # Merge against the file on disk, not the cached copy: a human may have
    # edited the document since the bundle was last loaded.
    if file_path.exists():
        try:
            on_disk = Concept.from_file(file_path, bundle.root_path)
            existing_frontmatter = on_disk.frontmatter
        except Exception:
            existing_frontmatter = existing_frontmatter

    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    # A field the caller did not supply keeps its stored value; an explicit
    # empty list clears it. Sources and tags are provenance, so a rewrite that
    # never mentioned them must not silently drop them.
    if sources is None:
        resolved_sources = list(existing_frontmatter.sources) if existing_frontmatter else []
    else:
        resolved_sources = source_objects

    if tags is None:
        resolved_tags = list(existing_frontmatter.tags) if existing_frontmatter else []
    else:
        resolved_tags = list(tags)

    frontmatter = Frontmatter(
        type=type_,
        title=title,
        description=description,
        resource=existing_frontmatter.resource if existing_frontmatter else None,
        tags=resolved_tags,
        sources=resolved_sources,
        generated=Generated(by=f"markrag/{__version__}", at=timestamp),
        # Verification is a human act: preserve it, never write it.
        verified=list(existing_frontmatter.verified) if existing_frontmatter else [],
        status="draft",
        stale_after=existing_frontmatter.stale_after if existing_frontmatter else None,
        # Unknown fields are preserved verbatim across rewrites.
        unknown_fields=dict(existing_frontmatter.unknown_fields) if existing_frontmatter else {},
        present_fields=(
            existing_frontmatter.present_fields if existing_frontmatter else frozenset()
        ),
    )

    concept = Concept(
        path=file_path,
        concept_id=concept_id,
        frontmatter=frontmatter,
        body=body_markdown,
    )
    file_path.write_text(concept.to_markdown(), encoding="utf-8")

    action = "updated" if existing is not None else "created"
    entry = f"- [{timestamp}] {action}: [{title}](/{concept_id}.md) by markrag"
    for directory in _log_directories(concept_id):
        append_log_entry(wiki_path, entry, directory=directory)

    bundle.load()
    write_bundle_map(bundle)
    _invalidate_index()

    return {
        "concept_id": concept_id,
        "path": str(file_path),
        "status": "written",
        "action": action,
    }


@mcp.tool()
def okf_status() -> dict[str, Any]:
    """Bundle statistics for orientation.

    Returns: document counts by type, status and trust level, the list of
    stale documents with days past, and when the search index was last
    rebuilt. Cheap; no bodies.

    Returns:
        Dictionary with bundle statistics.
    """
    bundle = _get_bundle()

    type_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    trust_counts: dict[str, int] = {}

    for concept in bundle.concepts.values():
        type_counts[concept.frontmatter.type] = type_counts.get(concept.frontmatter.type, 0) + 1
        status = concept.frontmatter.status
        status_counts[status] = status_counts.get(status, 0) + 1
        trust = derive_trust_level(concept)
        trust_counts[trust] = trust_counts.get(trust, 0) + 1

    stale_docs: list[dict[str, Any]] = []
    for concept_id, concept in bundle.concepts.items():
        stale_days = get_stale_days(concept)
        if stale_days is not None:
            stale_docs.append(
                {
                    "concept_id": concept_id,
                    "title": concept.title or concept_id,
                    "days_past": stale_days,
                }
            )
    stale_docs.sort(key=lambda doc: doc["days_past"], reverse=True)

    index_path = bundle.root_path / "index.md"
    last_rebuild = None
    if index_path.exists():
        last_rebuild = datetime.fromtimestamp(index_path.stat().st_mtime, tz=UTC).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

    index_built = None
    vectors = False
    if _index is not None and _index_bundle_path == bundle.root_path:
        index_built = _index.fingerprint()
        vectors = _index.vectors_available()

    return {
        "total_docs": len(bundle.concepts),
        "by_type": type_counts,
        "by_status": status_counts,
        "by_trust_level": trust_counts,
        "stale_docs": stale_docs,
        "stale_count": len(stale_docs),
        "last_rebuild": last_rebuild,
        "index": {
            "built": index_built is not None,
            "vectors": vectors,
        },
    }


def main() -> None:
    """Run the MCP server over stdio."""
    repo = _repo_path()

    # First run bootstraps the wiki when the sibling folder is missing.
    bundle = _get_bundle(repo, allow_bootstrap=True)

    # Build the search index up front so the first tool call is fast.
    try:
        _get_index(bundle)
    except Exception:  # pragma: no cover - index failures must not stop serving
        pass

    mcp.run()


if __name__ == "__main__":
    main()

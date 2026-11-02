"""MCP server for markrag - OKF-backed RAG for per-repo wikis.

This module provides the MCP server with tools for:
- okf_index: Bundle map (tier 0)
- okf_search: Hybrid FTS+vector search (tier 1)
- okf_read: Full document retrieval (tier 2)
- okf_write: Create/update concepts
- okf_status: Bundle statistics
"""

import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from markrag.okf import (
    Bundle,
    Concept,
    Frontmatter,
    RESERVED_FILES,
    derive_trust_level,
    get_stale_days,
    is_stale,
)

# Global state
_bundle: Bundle | None = None
_bundle_path: Path | None = None
_index_db_path: Path | None = None


def _get_bundle(repo_path: str | None = None, wiki_path: Path | None = None) -> Bundle:
    """Get or create the bundle for the current repo."""
    global _bundle, _bundle_path

    # If wiki_path is explicitly provided, use it
    if wiki_path:
        if _bundle is None or _bundle_path != wiki_path:
            _bundle_path = wiki_path
            _bundle = Bundle(root_path=wiki_path)
            try:
                _bundle.load()
            except FileNotFoundError:
                # Wiki doesn't exist yet - create empty bundle
                _bundle = Bundle(root_path=wiki_path)
        return _bundle

    # Determine repo path
    if repo_path:
        repo_path_obj = Path(repo_path)
    else:
        repo_path_obj = Path(os.getenv("MARKRAG_REPO", "."))

    # Derive wiki path (sibling convention)
    # If repo is at /path/to/<name>, wiki is at /path/to/<name>.wiki
    wiki_path = repo_path_obj.with_suffix(".wiki")

    # Check if we need to reload
    if _bundle is None or _bundle_path != wiki_path:
        _bundle_path = wiki_path
        _bundle = Bundle(root_path=wiki_path)
        try:
            _bundle.load()
        except FileNotFoundError:
            # Wiki doesn't exist yet - create empty bundle
            _bundle = Bundle(root_path=wiki_path)

    return _bundle


def _get_index_db_path() -> Path:
    """Get the path to the SQLite index database."""
    global _index_db_path

    if _index_db_path is None:
        # Use ~/.markrag/<bundle-name>.db
        repo_path = Path(os.getenv("MARKRAG_REPO", "."))
        bundle_name = repo_path.name
        markrag_dir = Path.home() / ".markrag"
        markrag_dir.mkdir(parents=True, exist_ok=True)
        _index_db_path = markrag_dir / f"{bundle_name}.db"

    return _index_db_path


def _tokenize_text(text: str, max_tokens: int = 40) -> str:
    """Tokenize text and truncate to max_tokens.

    Simple whitespace-based tokenization for excerpt generation.
    """
    if not text:
        return ""

    words = text.split()
    if len(words) <= max_tokens:
        return text

    return " ".join(words[:max_tokens]) + "..."


def _generate_trust_line(concept: Concept) -> str:
    """Generate trust level line for output."""
    trust = derive_trust_level(concept)
    stale_days = get_stale_days(concept)

    if stale_days is not None:
        return f"[Trust: {trust} | Stale: {stale_days} days past stale_after]"
    return f"[Trust: {trust}]"


def _generate_stale_warning(concept: Concept) -> str | None:
    """Generate staleness warning if applicable."""
    if is_stale(concept):
        stale_days = get_stale_days(concept)
        if stale_days:
            return f"WARNING: Document is {stale_days} days past stale_after"
        return "WARNING: Document is stale"
    return None


# Create MCP server
mcp = FastMCP(
    "markrag",
    instructions="""
You have access to a curated knowledge wiki for the current repository via markrag.

## Usage Rules

1. **Tier 0 (okf_index)**: Get bundle map - concept IDs, titles, descriptions, status. No bodies.
2. **Tier 1 (okf_search)**: Hybrid search with ~40-token excerpts. Use for discovery.
3. **Tier 2 (okf_read)**: Full document content. Use when you need the complete information.

## Trust Model

Every document has a trust level:
- `unverified`: No verified field present
- `machine-confirmed`: Only agent/process verifiers (producer/version, process:<id>)
- `human-reviewed`: At least one human verifier (human:<id>)

Documents may also be flagged as stale if past their stale_after date.

## When to use what

- Start with `okf_index` to understand the wiki structure
- Use `okf_search` to find relevant concepts
- Use `okf_read` only when you need the full content
- Always check trust level and staleness before relying on information

## Important

- Generated documents are always stamped with `generated.by` and `status: draft`
- Verification is a human act only - never mark content as verified yourself
- Never fabricate sources, verification events, or timestamps
""",
)


@mcp.tool()
def okf_index(path: str | None = None) -> dict[str, Any]:
    """Tier 0. Bundle map for root or a subdirectory path.

    Returns each entry as: concept_id, title, description (one line), status.
    No bodies. Server-managed files (index.md, log.md, home.md) are excluded.

    Args:
        path: Optional subdirectory path within the bundle (e.g. 'processes')

    Returns:
        Dictionary with 'concepts' list and 'path' info
    """
    bundle = _get_bundle()

    if path:
        # Filter to subdirectory
        prefix = path.rstrip("/") + "/"
        concepts = [
            (cid, c) for cid, c in bundle.concepts.items()
            if cid.startswith(prefix) or cid == path
        ]
    else:
        concepts = list(bundle.concepts.items())

    result = []
    for concept_id, concept in concepts:
        # Skip reserved files (shouldn't be here, but just in case)
        if concept.filename in RESERVED_FILES:
            continue

        trust = derive_trust_level(concept)
        stale_flag = is_stale(concept)

        result.append({
            "concept_id": concept_id,
            "title": concept.frontmatter.title or concept_id,
            "description": concept.frontmatter.description or "",
            "type": concept.frontmatter.type,
            "status": concept.frontmatter.status,
            "trust_level": trust,
            "stale": stale_flag,
        })

    return {
        "path": path or ".",
        "concepts": result,
        "count": len(result),
    }


@mcp.tool()
def okf_search(
    query: str,
    type_: str | None = None,
    tags: list[str] | None = None,
    status: str | None = None,
    top_k: int = 8,
) -> dict[str, Any]:
    """Tier 1. Hybrid FTS+vector search with metadata pre-filter and 1-hop link expansion.

    Per hit return: concept_id, title, description, trust_level, stale flag,
    best-matching section anchor, and a short excerpt (~40 tokens max).
    Plus 'related': [concept_id + description] from 1-hop links.
    Never full bodies.

    Args:
        query: Search query string
        type_: Filter by concept type (optional)
        tags: Filter by tags (AND logic) (optional)
        status: Filter by status: "any", "draft", "stable", "deprecated" (default: "any")
        top_k: Number of results to return (default: 8)

    Returns:
        Dictionary with 'results' list and 'stats'
    """
    bundle = _get_bundle()

    # Build candidate set with metadata filtering
    candidates = list(bundle.concepts.values())

    # Apply type filter
    if type_:
        candidates = [c for c in candidates if c.frontmatter.type == type_]

    # Apply status filter
    if status and status != "any":
        candidates = [c for c in candidates if c.frontmatter.status == status]

    # Apply tags filter (AND logic)
    if tags:
        for tag in tags:
            candidates = [c for c in candidates if tag in c.frontmatter.tags]

    # Simple keyword-based search for now (FTS5 will be used when index is built)
    # This is a fallback implementation
    results = []

    for concept in candidates:
        # Skip reserved files
        if concept.filename in RESERVED_FILES:
            continue

        # Match against various fields
        match_score = 0

        # Title match (high weight)
        if query.lower() in (concept.frontmatter.title or "").lower():
            match_score += 10

        # Description match
        if query.lower() in (concept.frontmatter.description or "").lower():
            match_score += 5

        # Body match
        if query.lower() in concept.body.lower():
            match_score += 2

        # Tag match
        if any(query.lower() in tag.lower() for tag in concept.frontmatter.tags):
            match_score += 3

        if match_score > 0:
            trust = derive_trust_level(concept)
            stale_flag = is_stale(concept)

            # Find best matching section
            best_section = None
            if "## " in concept.body:
                for match in re.finditer(r'^##\s+(.+)$', concept.body, re.MULTILINE):
                    if query.lower() in match.group(1).lower():
                        best_section = match.group(1)
                        break

            # Generate excerpt (~40 tokens)
            body_for_excerpt = concept.body
            if best_section:
                # Find the section content
                lines = concept.body.split("\n")
                found = False
                for i, line in enumerate(lines):
                    if line.startswith(f"## {best_section}"):
                        found = True
                        body_for_excerpt = "\n".join(lines[i:i+10])  # Next 10 lines
                        break

            excerpt = _tokenize_text(
                concept.frontmatter.description or body_for_excerpt or "",
                max_tokens=40
            )

            # Get 1-hop links
            related = []
            linked_ids = concept.extract_links()
            for linked_id in linked_ids:
                linked_concept = bundle.get_concept(linked_id)
                if linked_concept:
                    related.append({
                        "concept_id": linked_id,
                        "description": linked_concept.frontmatter.description or "",
                    })

            results.append({
                "concept_id": concept.concept_id,
                "title": concept.frontmatter.title or concept.concept_id,
                "description": concept.frontmatter.description or "",
                "type": concept.frontmatter.type,
                "status": concept.frontmatter.status,
                "trust_level": trust,
                "stale": stale_flag,
                "best_section": best_section,
                "excerpt": excerpt,
                "related": related,
                "score": match_score,
            })

    # Sort by score (descending)
    results.sort(key=lambda x: x["score"], reverse=True)

    return {
        "query": query,
        "results": results[:top_k],
        "count": len(results),
        "filters": {
            "type": type_,
            "tags": tags,
            "status": status,
        },
    }


@mcp.tool()
def okf_read(
    concept_id: str,
    section: str | None = None,
) -> dict[str, Any]:
    """Tier 2. Full document, or a single ## section when section is given.

    Output: trust_level line, staleness warning if any, frontmatter summary
    (sources list), then the Markdown content.

    Args:
        concept_id: The concept ID to read
        section: Optional section heading to extract (match by heading text)

    Returns:
        Dictionary with full document content
    """
    bundle = _get_bundle()

    concept = bundle.get_concept(concept_id)

    if concept is None:
        # Check if this is a reserved file
        reserved_stems = {f.replace(".md", "") for f in RESERVED_FILES}
        if concept_id in reserved_stems:
            return {
                "error": f"{concept_id} is a server-managed file, not a concept",
                "concept_id": concept_id,
            }
        return {
            "error": f"Concept not found: {concept_id}",
            "concept_id": concept_id,
        }

    # Skip reserved files (should return empty or error)
    if concept.filename in RESERVED_FILES:
        return {
            "error": f"{concept.filename} is a server-managed file, not a concept",
            "concept_id": concept_id,
        }

    trust_line = _generate_trust_line(concept)
    stale_warning = _generate_stale_warning(concept)

    # Build result
    result: dict[str, Any] = {
        "concept_id": concept_id,
        "trust_level": derive_trust_level(concept),
        "trust_line": trust_line,
        "stale": is_stale(concept),
        "stale_warning": stale_warning,
        "frontmatter": concept.frontmatter.to_dict(),
    }

    # Add sources summary
    if concept.frontmatter.sources:
        result["sources"] = [
            {
                "id": s.id,
                "resource": s.resource,
                "title": s.title,
            }
            for s in concept.frontmatter.sources
        ]

    # Get content (full or section)
    if section:
        # Find the section
        lines = concept.body.split("\n")
        found = False
        content_lines = []

        for line in lines:
            if line.startswith(f"## {section}"):
                found = True
                content_lines.append(line)
            elif found:
                if line.startswith("## "):
                    # Hit next section, stop
                    break
                content_lines.append(line)

        if found:
            result["content"] = "\n".join(content_lines)
            result["section"] = section
        else:
            # Section not found, return empty content
            result["content"] = ""
            result["section"] = section
            result["error"] = f"Section '{section}' not found"
    else:
        result["content"] = concept.body

    return result


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

    ALWAYS stamps generated{by, at} and status=draft.
    Updates preserve existing verified history; never add/forge verification.
    Appends a line to log.md.
    Regenerates affected index.md and home.md from frontmatter.

    Args:
        concept_id: The concept ID (path without .md)
        title: Concept title
        description: One-line description
        type_: Concept type
        body_markdown: Markdown body content
        tags: Optional list of tags
        sources: Optional list of source dictionaries

    Returns:
        Dictionary with written concept_id and path
    """
    bundle = _get_bundle()
    wiki_path = bundle.root_path

    # Create wiki directory if it doesn't exist
    wiki_path.mkdir(parents=True, exist_ok=True)

    # Determine file path from concept_id
    # concept_id uses / as separator, need to convert to OS path
    concept_path_str = concept_id.replace("/", os.sep)
    file_path = wiki_path / f"{concept_path_str}.md"

    # Create parent directories if needed
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare frontmatter
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    version = __import__('markrag').__version__

    # Build sources if provided
    from markrag.okf import Source, Generated, Verified
    source_objs = []
    if sources:
        for s in sources:
            source_objs.append(Source(
                id=s.get("id", ""),
                resource=s.get("resource", ""),
                title=s.get("title"),
            ))

    # Check if concept exists and has verified history
    existing_concept = bundle.get_concept(concept_id)
    preserved_verified = []
    if existing_concept:
        preserved_verified = existing_concept.frontmatter.verified

    # Create frontmatter
    frontmatter = Frontmatter(
        type=type_,
        title=title,
        description=description,
        tags=tags or [],
        sources=source_objs,
        generated=Generated(by=f"markrag/{version}", at=timestamp),
        verified=preserved_verified,  # Preserve existing verification
        status="draft",
        unknown_fields={},
    )

    # Create concept
    concept = Concept(
        path=file_path,
        concept_id=concept_id,
        frontmatter=frontmatter,
        body=body_markdown,
    )

    # Write the file
    markdown_content = concept.to_markdown()
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    # Append to log.md
    _append_to_log(wiki_path, concept_id, title, "created" if not existing_concept else "updated")

    # Regenerate index.md and home.md
    _regenerate_index_and_home(bundle)

    # Reload bundle
    bundle.load()

    return {
        "concept_id": concept_id,
        "path": str(file_path),
        "status": "written",
    }


@mcp.tool()
def okf_status() -> dict[str, Any]:
    """Bundle stats for the agent's orientation.

    Returns: doc count by type/status/trust level, stale docs list (id + days past),
    last rebuild time. Cheap; no bodies.

    Returns:
        Dictionary with bundle statistics
    """
    bundle = _get_bundle()

    # Count by type
    type_counts: dict[str, int] = {}
    for concept in bundle.concepts.values():
        if concept.filename in RESERVED_FILES:
            continue
        type_ = concept.frontmatter.type
        type_counts[type_] = type_counts.get(type_, 0) + 1

    # Count by status
    status_counts: dict[str, int] = {}
    for concept in bundle.concepts.values():
        if concept.filename in RESERVED_FILES:
            continue
        status = concept.frontmatter.status
        status_counts[status] = status_counts.get(status, 0) + 1

    # Count by trust level
    trust_counts: dict[str, int] = {}
    for concept in bundle.concepts.values():
        if concept.filename in RESERVED_FILES:
            continue
        trust = derive_trust_level(concept)
        trust_counts[trust] = trust_counts.get(trust, 0) + 1

    # Find stale docs
    stale_docs = []
    for concept_id, concept in bundle.concepts.items():
        if concept.filename in RESERVED_FILES:
            continue
        stale_days = get_stale_days(concept)
        if stale_days is not None:
            stale_docs.append({
                "concept_id": concept_id,
                "title": concept.frontmatter.title or concept_id,
                "days_past": stale_days,
            })

    # Sort stale docs by days past (descending)
    stale_docs.sort(key=lambda x: x["days_past"], reverse=True)

    # Last rebuild time (file modification time of index)
    index_path = bundle.root_path / "index.md"
    last_rebuild = None
    if index_path.exists():
        last_rebuild = datetime.fromtimestamp(
            index_path.stat().st_mtime, tz=timezone.utc
        ).strftime("%Y-%m-%dT%H:%M:%SZ")

    return {
        "total_docs": len(bundle.concepts),
        "by_type": type_counts,
        "by_status": status_counts,
        "by_trust_level": trust_counts,
        "stale_docs": stale_docs,
        "stale_count": len(stale_docs),
        "last_rebuild": last_rebuild,
    }


# Helper functions for write operations

def _append_to_log(wiki_path: Path, concept_id: str, title: str, action: str) -> None:
    """Append an entry to log.md."""
    log_path = wiki_path / "log.md"
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    entry = f"- [{timestamp}] {action}: [{title}](/{concept_id}.md)\n"

    if log_path.exists():
        with open(log_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check if we need to add header
        if not content.strip().startswith("#"):
            content = "# Change Log\n\n" + content

        # Check if we need to add a newline
        if not content.endswith("\n"):
            content += "\n"

        content += entry

        with open(log_path, "w", encoding="utf-8") as f:
            f.write(content)
    else:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("# Change Log\n\n")
            f.write(entry)


def _regenerate_index_and_home(bundle: Bundle) -> None:
    """Regenerate index.md and home.md from the bundle."""
    wiki_path = bundle.root_path

    # Reload to get latest state
    bundle.load()

    # Generate index.md content
    index_content = """---
okf_version: "0.2"
---

# Bundle Index

"""

    # Group concepts by directory
    from collections import defaultdict
    dirs: dict[str, list[tuple[str, Concept]]] = defaultdict(list)

    for concept_id, concept in bundle.concepts.items():
        if concept.filename in RESERVED_FILES:
            continue

        # Get directory path
        if "/" in concept_id:
            dir_path = concept_id.rsplit("/", 1)[0]
        else:
            dir_path = "."

        dirs[dir_path].append((concept_id, concept))

    # Write root-level concepts
    root_key = "."
    if root_key in dirs:
        index_content += "## Root\n\n"
        for concept_id, concept in sorted(dirs[root_key], key=lambda x: x[1].frontmatter.title or x[0]):
            trust = derive_trust_level(concept)
            stale = is_stale(concept)
            stale_marker = " (stale)" if stale else ""
            index_content += f"- [{concept.frontmatter.title or concept_id}](/{concept_id}.md) | {concept.frontmatter.type} | {concept.frontmatter.status} | {trust}{stale_marker}\n"
        index_content += "\n"
        del dirs[root_key]

    # Write subdirectories
    for dir_path in sorted(dirs.keys()):
        index_content += f"## {dir_path}\n\n"
        for concept_id, concept in sorted(dirs[dir_path], key=lambda x: x[1].frontmatter.title or x[0]):
            trust = derive_trust_level(concept)
            stale = is_stale(concept)
            stale_marker = " (stale)" if stale else ""
            index_content += f"- [{concept.frontmatter.title or concept_id}](/{concept_id}.md) | {concept.frontmatter.type} | {concept.frontmatter.status} | {trust}{stale_marker}\n"
        index_content += "\n"

    # Write index.md
    with open(wiki_path / "index.md", "w", encoding="utf-8") as f:
        f.write(index_content)

    # Generate home.md content (GitHub Wiki compatible)
    home_content = "# Wiki Home\n\n"
    home_content += "This is the GitHub Wiki for the repository.\n\n"
    home_content += "## Navigation\n\n"

    # Write all concepts as a flat list for home.md
    for concept_id, concept in sorted(
        bundle.concepts.items(),
        key=lambda x: x[1].frontmatter.title or x[0]
    ):
        if concept.filename in RESERVED_FILES:
            continue
        trust = derive_trust_level(concept)
        stale = is_stale(concept)
        stale_marker = " (stale)" if stale else ""
        # Use relative links without leading /
        link = concept_id.replace("/", "/") + ".md"
        home_content += f"- [{concept.frontmatter.title or concept_id}]({link}) | {concept.frontmatter.type} | {concept.frontmatter.status} | {trust}{stale_marker}\n"

    home_content += "\n"

    # Write home.md
    with open(wiki_path / "home.md", "w", encoding="utf-8") as f:
        f.write(home_content)


def main() -> None:
    """Run the MCP server."""
    # Get repo path from env or default
    repo_path = os.getenv("MARKRAG_REPO", ".")

    # Initialize bundle (will create empty if doesn't exist)
    _get_bundle(repo_path)

    # Run the server
    mcp.run()


if __name__ == "__main__":
    main()

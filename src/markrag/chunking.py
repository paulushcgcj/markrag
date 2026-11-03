"""Chunking logic for OKF concepts.

Concepts are split at ``##`` heading boundaries so every chunk stays within a
single document. Two rules keep retrieval honest:

- content before the first ``##`` is never dropped, and
- a section that exceeds :data:`MAX_CHUNK_CHARS` is split further at
  paragraph boundaries.

Every chunk carries its concept ID plus the document's full frontmatter so
provenance survives chunking.
"""

import re
from dataclasses import dataclass, field
from typing import Any

from markrag.okf import Concept, derive_trust_level

# Upper bound on chunk size in characters. Roughly 300 tokens of English text.
MAX_CHUNK_CHARS = 1200

# Chunk kinds.
KIND_TITLE = "title"
KIND_DESCRIPTION = "description"
KIND_HEADING = "heading"
KIND_BODY = "body"

_HEADING_RE = re.compile(r"^##\s+(.+)$")


@dataclass(frozen=True)
class Chunk:
    """A chunk of a concept for embedding and retrieval."""

    concept_id: str
    chunk_index: int
    kind: str
    text: str
    heading: str | None
    frontmatter: dict[str, Any] = field(default_factory=dict)
    trust_level: str = "unverified"
    status: str = "stable"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "concept_id": self.concept_id,
            "chunk_index": self.chunk_index,
            "kind": self.kind,
            "text": self.text,
            "heading": self.heading,
            "frontmatter": self.frontmatter,
            "trust_level": self.trust_level,
            "status": self.status,
        }


def _split_long(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split oversized text at paragraph, then line, then character boundaries."""
    if len(text) <= max_chars:
        return [text]

    paragraphs = re.split(r"\n\s*\n", text)
    pieces: list[str] = []
    current = ""

    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            pieces.append(current)
            current = ""
        if len(paragraph) <= max_chars:
            current = paragraph
            continue
        # A single oversized paragraph: fall back to line boundaries.
        buffer = ""
        for line in paragraph.splitlines():
            candidate = f"{buffer}\n{line}" if buffer else line
            if len(candidate) <= max_chars:
                buffer = candidate
                continue
            if buffer:
                pieces.append(buffer)
                buffer = ""
            while len(line) > max_chars:
                pieces.append(line[:max_chars])
                line = line[max_chars:]
            buffer = line
        if buffer:
            current = buffer

    if current:
        pieces.append(current)
    return pieces


def _is_heading_only(text: str) -> bool:
    """Check whether a block contains any content beyond headings."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return False
    return True


def chunk_concept(concept: Concept, max_chars: int = MAX_CHUNK_CHARS) -> list[Chunk]:
    """Split a concept body into chunks at ``##`` heading boundaries.

    The content preceding the first ``##`` is preserved: it becomes its own
    chunk when it carries real content, and is folded into the first section
    when it only contains headings.

    Args:
        concept: The concept to chunk.
        max_chars: Maximum characters per chunk.

    Returns:
        List of body chunks in document order.
    """
    frontmatter = concept.frontmatter.to_dict()
    trust_level = derive_trust_level(concept)
    status = concept.frontmatter.status
    body = concept.body

    def make(index: int, text: str, heading: str | None) -> Chunk:
        return Chunk(
            concept_id=concept.concept_id,
            chunk_index=index,
            kind=KIND_BODY,
            text=text,
            heading=heading,
            frontmatter=frontmatter,
            trust_level=trust_level,
            status=status,
        )

    # No `##` headings: the whole body is a single chunk.
    if "## " not in body:
        text = body.strip()
        return [make(0, text, None)] if text else []

    lines = body.split("\n")
    preamble: list[str] = []
    sections: list[tuple[str, list[str]]] = []
    current_heading: str | None = None
    current_lines: list[str] = []

    for line in lines:
        match = _HEADING_RE.match(line)
        if match:
            if current_heading is None:
                current_heading = match.group(1).strip()
                current_lines = [line]
            else:
                sections.append((current_heading, current_lines))
                current_heading = match.group(1).strip()
                current_lines = [line]
        elif current_heading is None:
            preamble.append(line)
        else:
            current_lines.append(line)

    if current_heading is not None:
        sections.append((current_heading, current_lines))

    chunks: list[Chunk] = []
    preamble_text = "\n".join(preamble).strip()

    if preamble_text and not _is_heading_only(preamble_text):
        for piece in _split_long(preamble_text, max_chars):
            chunks.append(make(len(chunks), piece, None))

    for heading, section_lines in sections:
        section_text = "\n".join(section_lines).strip()
        if not section_text:
            continue
        pieces = _split_long(section_text, max_chars)
        for piece_index, piece in enumerate(pieces):
            if piece_index == 0 and preamble_text and _is_heading_only(preamble_text):
                # Fold a heading-only preamble into the first section.
                piece = f"{preamble_text}\n\n{piece}"
            chunks.append(make(len(chunks), piece, heading))

    return chunks


def chunk_description(concept: Concept) -> list[Chunk]:
    """Create the description and heading chunks used for vector search.

    The retrieval design embeds three things per concept: its description,
    each of its ``##`` headings, and its body chunks.

    Args:
        concept: The concept to chunk.

    Returns:
        A description chunk followed by one chunk per ``##`` heading.
    """
    frontmatter = concept.frontmatter.to_dict()
    trust_level = derive_trust_level(concept)
    status = concept.frontmatter.status
    chunks: list[Chunk] = []

    description = concept.frontmatter.description
    if description:
        chunks.append(
            Chunk(
                concept_id=concept.concept_id,
                chunk_index=len(chunks),
                kind=KIND_DESCRIPTION,
                text=description,
                heading="description",
                frontmatter=frontmatter,
                trust_level=trust_level,
                status=status,
            )
        )

    for match in re.finditer(r"^##\s+(.+)$", concept.body, re.MULTILINE):
        heading_text = match.group(1).strip()
        chunks.append(
            Chunk(
                concept_id=concept.concept_id,
                chunk_index=len(chunks),
                kind=KIND_HEADING,
                text=heading_text,
                heading=heading_text,
                frontmatter=frontmatter,
                trust_level=trust_level,
                status=status,
            )
        )

    return chunks


def chunk_title(concept: Concept) -> list[Chunk]:
    """Create the title chunk used for title lookups.

    Titles are the strongest signal a reader has for a document, so they are
    indexed for both keyword and semantic retrieval.

    Args:
        concept: The concept to chunk.

    Returns:
        A single chunk holding the concept title, or an empty list when the
        concept has no title.
    """
    title = concept.title
    if not title:
        return []
    return [
        Chunk(
            concept_id=concept.concept_id,
            chunk_index=0,
            kind=KIND_TITLE,
            text=title,
            heading=title,
            frontmatter=concept.frontmatter.to_dict(),
            trust_level=derive_trust_level(concept),
            status=concept.frontmatter.status,
        )
    ]


def chunk_all(concept: Concept, max_chars: int = MAX_CHUNK_CHARS) -> list[Chunk]:
    """Chunk a concept completely: title, description, headings, and body.

    Args:
        concept: The concept to chunk.
        max_chars: Maximum characters per body chunk.

    Returns:
        Every chunk for the concept, with sequential chunk indexes.
    """
    chunks = [
        *chunk_title(concept),
        *chunk_description(concept),
        *chunk_concept(concept, max_chars),
    ]
    return [
        Chunk(
            concept_id=chunk.concept_id,
            chunk_index=index,
            kind=chunk.kind,
            text=chunk.text,
            heading=chunk.heading,
            frontmatter=chunk.frontmatter,
            trust_level=chunk.trust_level,
            status=chunk.status,
        )
        for index, chunk in enumerate(chunks)
    ]

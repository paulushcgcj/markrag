"""Chunking logic for OKF concepts.

Splits concepts at ## heading boundaries for RAG retrieval.
"""

import re
from dataclasses import dataclass
from typing import Any

from markrag.okf import Concept, derive_trust_level


@dataclass
class Chunk:
    """A chunk of a concept for embedding and retrieval."""

    concept_id: str
    chunk_index: int
    text: str
    heading: str | None
    frontmatter: dict[str, Any]
    trust_level: str
    status: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "concept_id": self.concept_id,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "heading": self.heading,
            "frontmatter": self.frontmatter,
            "trust_level": self.trust_level,
            "status": self.status,
        }


def chunk_concept(concept: Concept) -> list[Chunk]:
    """Split a concept into chunks at ## heading boundaries.

    Each chunk includes:
    - The heading text (if any)
    - The content under that heading (including content before first ##)
    - Full frontmatter from the concept
    - Concept ID and chunk index

    Args:
        concept: The concept to chunk

    Returns:
        List of chunks
    """
    chunks = []
    body = concept.body

    # If no ## headings, treat entire body as one chunk
    if "## " not in body:
        chunk = Chunk(
            concept_id=concept.concept_id,
            chunk_index=0,
            text=body.strip(),
            heading=None,
            frontmatter=concept.frontmatter.to_dict(),
            trust_level=derive_trust_level(concept),
            status=concept.frontmatter.status,
        )
        chunks.append(chunk)
        return chunks

    # Split at ## headings
    lines = body.split("\n")
    current_chunk_lines: list[str] = []
    current_heading: str | None = None
    chunk_index = 0
    has_first_heading = False

    for line in lines:
        # Check for ## heading
        match = re.match(r'^##\s+(.+)$', line)
        if match:
            if has_first_heading:
                # Save previous chunk if it has content
                if current_chunk_lines:
                    chunk_text = "\n".join(current_chunk_lines).strip()
                    if chunk_text:
                        chunk = Chunk(
                            concept_id=concept.concept_id,
                            chunk_index=chunk_index,
                            text=chunk_text,
                            heading=current_heading,
                            frontmatter=concept.frontmatter.to_dict(),
                            trust_level=derive_trust_level(concept),
                            status=concept.frontmatter.status,
                        )
                        chunks.append(chunk)
                        chunk_index += 1
            else:
                # First heading - include accumulated content
                has_first_heading = True

            # Start new chunk with heading
            current_chunk_lines = [line]
            current_heading = match.group(1)
        else:
            # Accumulate content
            if has_first_heading or current_heading is not None:
                current_chunk_lines.append(line)
            else:
                # Content before first heading - accumulate but don't create chunk yet
                current_chunk_lines.append(line)

    # Don't forget the last chunk
    if current_chunk_lines:
        chunk_text = "\n".join(current_chunk_lines).strip()
        if chunk_text:
            chunk = Chunk(
                concept_id=concept.concept_id,
                chunk_index=chunk_index,
                text=chunk_text,
                heading=current_heading,
                frontmatter=concept.frontmatter.to_dict(),
                trust_level=derive_trust_level(concept),
                status=concept.frontmatter.status,
            )
            chunks.append(chunk)

    return chunks


def chunk_description(concept: Concept) -> list[Chunk]:
    """Create chunks for description and headings only (for vector search).

    According to spec, we embed:
    (a) each description
    (b) each heading
    (c) body chunks

    This function creates chunks for (a) and (b), while chunk_concept handles (c).
    """
    chunks = []
    fm = concept.frontmatter

    # Description chunk
    if fm.description:
        desc_chunk = Chunk(
            concept_id=concept.concept_id,
            chunk_index=-2,  # Special index for description
            text=fm.description,
            heading="description",
            frontmatter=fm.to_dict(),
            trust_level=derive_trust_level(concept),
            status=fm.status,
        )
        chunks.append(desc_chunk)

    # Heading chunks
    if "## " in concept.body:
        # Extract all ## headings
        for match in re.finditer(r'^##\s+(.+)$', concept.body, re.MULTILINE):
            heading_text = match.group(1)
            heading_chunk = Chunk(
                concept_id=concept.concept_id,
                chunk_index=-1,  # Special index for headings
                text=heading_text,
                heading=heading_text,
                frontmatter=fm.to_dict(),
                trust_level=derive_trust_level(concept),
                status=fm.status,
            )
            chunks.append(heading_chunk)

    return chunks

"""Tests for chunking logic."""

import pytest

from markrag.chunking import Chunk, chunk_concept, chunk_description
from markrag.okf import Concept, Frontmatter, derive_trust_level


class TestChunking:
    """Test concept chunking."""

    def test_single_chunk_no_headings(self, temp_wiki_dir):
        """Test chunking a concept with no headings."""
        content = """---
type: Test
title: Simple
description: Simple concept
---

This is a simple concept with no headings. Just plain text.
"""
        file_path = temp_wiki_dir / "simple.md"
        with open(file_path, "w") as f:
            f.write(content)

        concept = Concept.from_file(file_path)
        chunks = chunk_concept(concept)

        assert len(chunks) == 1
        assert chunks[0].chunk_index == 0
        assert chunks[0].heading is None
        assert "simple concept with no headings" in chunks[0].text

    def test_multiple_chunks_with_headings(self, temp_wiki_dir):
        """Test chunking a concept with multiple ## headings."""
        content = """---
type: Test
title: Multi-section
description: Concept with multiple sections
---

# Main Content

## Section 1

Content of section 1.

## Section 2

Content of section 2.

## Section 3

Content of section 3.
"""
        file_path = temp_wiki_dir / "multisec.md"
        with open(file_path, "w") as f:
            f.write(content)

        concept = Concept.from_file(file_path)
        chunks = chunk_concept(concept)

        # Should have 3 chunks (one for each ## section)
        assert len(chunks) == 3

        # Check headings
        assert chunks[0].heading == "Section 1"
        assert chunks[1].heading == "Section 2"
        assert chunks[2].heading == "Section 3"

        # Check content
        assert "Content of section 1" in chunks[0].text
        assert "Content of section 2" in chunks[1].text
        assert "Content of section 3" in chunks[2].text

    def test_chunk_metadata(self, temp_wiki_dir):
        """Test that chunks carry correct metadata."""
        content = """---
type: Technology
title: Test Tech
status: draft
generated: {by: markrag/0.1, at: 2026-07-09T14:30:00Z}
---

## Section

Content.
"""
        file_path = temp_wiki_dir / "meta.md"
        with open(file_path, "w") as f:
            f.write(content)

        concept = Concept.from_file(file_path)
        chunks = chunk_concept(concept)

        assert len(chunks) == 1
        chunk = chunks[0]

        assert chunk.concept_id == "meta"
        assert chunk.frontmatter["type"] == "Technology"
        assert chunk.frontmatter["status"] == "draft"
        assert "generated" in chunk.frontmatter
        assert chunk.trust_level == "unverified"
        assert chunk.status == "draft"

    def test_description_chunk(self, temp_wiki_dir):
        """Test that description is chunked separately."""
        content = """---
type: Test
title: With Description
description: This is the description for vector search
---

# Main Content

Body text.

## Section

More text.
"""
        file_path = temp_wiki_dir / "desc.md"
        with open(file_path, "w") as f:
            f.write(content)

        concept = Concept.from_file(file_path)
        desc_chunks = chunk_description(concept)

        # Should have one chunk for description
        assert len(desc_chunks) == 1
        assert desc_chunks[0].heading == "description"
        assert "This is the description" in desc_chunks[0].text

    def test_heading_chunks(self, temp_wiki_dir):
        """Test that headings are chunked separately."""
        content = """---
type: Test
title: With Headings
description: Desc
---

# Main

## Heading 1

Content 1.

## Heading 2

Content 2.
"""
        file_path = temp_wiki_dir / "headings.md"
        with open(file_path, "w") as f:
            f.write(content)

        concept = Concept.from_file(file_path)
        heading_chunks = chunk_description(concept)

        # Should have description + 2 headings
        assert len(heading_chunks) == 3
        assert heading_chunks[0].heading == "description"
        assert heading_chunks[1].heading == "Heading 1"
        assert heading_chunks[2].heading == "Heading 2"


class TestChunkMetadataStamping:
    """Test that chunks are properly stamped with metadata."""

    def test_trust_level_stamping(self, temp_wiki_dir):
        """Test that trust level is stamped on chunks."""
        content = """---
type: Test
title: Trust Test
verified:
  - by: human:john
    at: 2026-07-09T14:30:00Z
---

## Section

Content.
"""
        file_path = temp_wiki_dir / "trust.md"
        with open(file_path, "w") as f:
            f.write(content)

        concept = Concept.from_file(file_path)
        chunks = chunk_concept(concept)

        for chunk in chunks:
            assert chunk.trust_level == "human-reviewed"

    def test_status_stamping(self, temp_wiki_dir):
        """Test that status is stamped on chunks."""
        content = """---
type: Test
title: Status Test
status: deprecated
---

## Section

Content.
"""
        file_path = temp_wiki_dir / "status.md"
        with open(file_path, "w") as f:
            f.write(content)

        concept = Concept.from_file(file_path)
        chunks = chunk_concept(concept)

        for chunk in chunks:
            assert chunk.status == "deprecated"

    def test_frontmatter_stamping(self, temp_wiki_dir):
        """Test that full frontmatter is stamped on chunks."""
        content = """---
type: Custom
title: Custom Type
custom_field: custom_value
tags: [a, b]
---

## Section

Content.
"""
        file_path = temp_wiki_dir / "custom.md"
        with open(file_path, "w") as f:
            f.write(content)

        concept = Concept.from_file(file_path)
        chunks = chunk_concept(concept)

        for chunk in chunks:
            assert chunk.frontmatter["type"] == "Custom"
            assert chunk.frontmatter["title"] == "Custom Type"
            assert "custom_field" in chunk.frontmatter
            assert chunk.frontmatter["tags"] == ["a", "b"]

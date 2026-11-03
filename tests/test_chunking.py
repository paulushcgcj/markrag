"""Tests for chunking logic."""


from markrag.chunking import (
    KIND_BODY,
    KIND_DESCRIPTION,
    KIND_HEADING,
    KIND_TITLE,
    MAX_CHUNK_CHARS,
    chunk_all,
    chunk_concept,
    chunk_description,
    chunk_title,
)
from markrag.okf import Concept


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

        # Should have chunks for description and headings
        assert len(desc_chunks) == 2
        # First chunk should be description
        assert desc_chunks[0].heading == "description"
        assert "This is the description" in desc_chunks[0].text
        # Second chunk should be heading
        assert desc_chunks[1].heading == "Section"

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


class TestChunkSize:
    """Test the maximum chunk size guarantee."""

    def test_oversized_section_is_split(self, temp_wiki_dir):
        """A section longer than the limit is broken into several chunks."""
        paragraph = "word " * 300
        content = f"""---
type: Test
title: Oversized
description: Long section
---

## Long Section

{paragraph}

{paragraph}
"""
        file_path = temp_wiki_dir / "long.md"
        file_path.write_text(content, encoding="utf-8")

        concept = Concept.from_file(file_path)
        chunks = chunk_concept(concept)

        assert len(chunks) > 1
        assert all(len(chunk.text) <= MAX_CHUNK_CHARS for chunk in chunks)

    def test_single_unbroken_line_is_split(self, temp_wiki_dir):
        """A line with no spaces is still cut to the limit."""
        content = f"""---
type: Test
title: Unbroken
description: One long line
---

## Long Line

{"x" * (MAX_CHUNK_CHARS * 2)}
"""
        file_path = temp_wiki_dir / "unbroken.md"
        file_path.write_text(content, encoding="utf-8")

        concept = Concept.from_file(file_path)
        chunks = chunk_concept(concept)

        assert len(chunks) > 1
        assert all(len(chunk.text) <= MAX_CHUNK_CHARS for chunk in chunks)

    def test_normal_sections_stay_in_one_chunk(self, temp_wiki_dir):
        """Ordinary content is not needlessly fragmented."""
        content = """---
type: Test
title: Normal
description: Ordinary
---

## Short

A short section.
"""
        file_path = temp_wiki_dir / "normal.md"
        file_path.write_text(content, encoding="utf-8")

        concept = Concept.from_file(file_path)
        assert len(chunk_concept(concept)) == 1

    def test_custom_limit_is_respected(self, temp_wiki_dir):
        """The limit is a parameter, not a constant baked into the logic."""
        content = f"""---
type: Test
title: Custom
description: Custom limit
---

## Section

{"word " * 100}
"""
        file_path = temp_wiki_dir / "custom.md"
        file_path.write_text(content, encoding="utf-8")

        concept = Concept.from_file(file_path)
        chunks = chunk_concept(concept, max_chars=200)

        assert len(chunks) > 1
        assert all(len(chunk.text) <= 200 for chunk in chunks)


class TestPreamble:
    """Test that content before the first heading is never lost."""

    def test_preamble_becomes_its_own_chunk(self, temp_wiki_dir):
        """Introductory prose is kept, not dropped."""
        content = """---
type: Test
title: Preamble
description: Has a preamble
---

# Overview

Some introductory prose that matters.

## Section

Section content.
"""
        file_path = temp_wiki_dir / "preamble.md"
        file_path.write_text(content, encoding="utf-8")

        concept = Concept.from_file(file_path)
        chunks = chunk_concept(concept)

        assert len(chunks) == 2
        assert chunks[0].heading is None
        assert "introductory prose" in chunks[0].text
        assert chunks[1].heading == "Section"

    def test_heading_only_preamble_is_folded_into_the_first_section(self, temp_wiki_dir):
        """A bare '#' heading adds no content of its own."""
        content = """---
type: Test
title: Heading Only
description: Just a heading
---

# Overview

## Section

Section content.
"""
        file_path = temp_wiki_dir / "headingonly.md"
        file_path.write_text(content, encoding="utf-8")

        concept = Concept.from_file(file_path)
        chunks = chunk_concept(concept)

        assert len(chunks) == 1
        assert chunks[0].heading == "Section"
        assert "# Overview" in chunks[0].text

    def test_empty_body_produces_no_chunks(self, temp_wiki_dir):
        """An empty body is not an error."""
        content = """---
type: Test
title: Empty
description: Nothing here
---
"""
        file_path = temp_wiki_dir / "empty.md"
        file_path.write_text(content, encoding="utf-8")

        concept = Concept.from_file(file_path)
        assert chunk_concept(concept) == []


class TestChunkKinds:
    """Test the complete chunk set for a concept."""

    def test_chunk_all_covers_every_kind(self, temp_wiki_dir):
        """Title, description, heading, and body chunks are all produced."""
        content = """---
type: Test
title: Complete
description: A complete concept
---

## Section

Body content.
"""
        file_path = temp_wiki_dir / "complete.md"
        file_path.write_text(content, encoding="utf-8")

        concept = Concept.from_file(file_path)
        chunks = chunk_all(concept)

        kinds = [chunk.kind for chunk in chunks]
        assert KIND_TITLE in kinds
        assert KIND_DESCRIPTION in kinds
        assert KIND_HEADING in kinds
        assert KIND_BODY in kinds

    def test_chunk_indexes_are_sequential(self, temp_wiki_dir):
        """Chunk indexes are dense and ordered."""
        content = """---
type: Test
title: Sequential
description: Sequential indexes
---

## One

First.

## Two

Second.
"""
        file_path = temp_wiki_dir / "sequential.md"
        file_path.write_text(content, encoding="utf-8")

        concept = Concept.from_file(file_path)
        chunks = chunk_all(concept)

        assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))

    def test_title_chunk_is_absent_without_a_title(self, temp_wiki_dir):
        """A concept without a title still chunks the rest."""
        content = """---
type: Test
description: No title
---

## Section

Body.
"""
        file_path = temp_wiki_dir / "notitle.md"
        file_path.write_text(content, encoding="utf-8")

        concept = Concept.from_file(file_path)
        assert chunk_title(concept) == []
        assert chunk_all(concept)


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

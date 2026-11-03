"""Tests for MCP server tools."""

import os
from pathlib import Path

import pytest

from markrag.server import (
    okf_index,
    okf_read,
    okf_search,
    okf_status,
    okf_write,
)


class TestOkfIndex:
    """Test okf_index tool."""

    def test_index_all_concepts(self, sample_bundle):
        """Test indexing all concepts."""
        # Temporarily set the global bundle
        import markrag.server as server_module

        server_module._bundle = sample_bundle
        server_module._bundle_path = sample_bundle.root_path

        result = okf_index()

        assert "concepts" in result
        assert len(result["concepts"]) == 3
        assert "path" in result
        assert result["count"] == 3

        # Check concept data
        concept_ids = [c["concept_id"] for c in result["concepts"]]
        assert "technologies/spring-boot" in concept_ids
        assert "technologies/database" in concept_ids
        assert "processes/deployment" in concept_ids

    def test_index_filter_by_path(self, sample_bundle):
        """Test filtering index by path."""
        import markrag.server as server_module

        server_module._bundle = sample_bundle
        server_module._bundle_path = sample_bundle.root_path

        result = okf_index(path="technologies")

        assert len(result["concepts"]) == 2
        for concept in result["concepts"]:
            assert concept["concept_id"].startswith("technologies/")

    def test_index_trust_level(self, sample_bundle):
        """Test that index includes trust level."""
        import markrag.server as server_module

        server_module._bundle = sample_bundle
        server_module._bundle_path = sample_bundle.root_path

        result = okf_index()

        for concept in result["concepts"]:
            assert "trust_level" in concept
            assert concept["trust_level"] in ["unverified", "machine-confirmed", "human-reviewed"]

    def test_index_stale_flag(self, sample_bundle):
        """Test that index includes stale flag."""
        import markrag.server as server_module

        server_module._bundle = sample_bundle
        server_module._bundle_path = sample_bundle.root_path

        result = okf_index()

        for concept in result["concepts"]:
            assert "stale" in concept


class TestOkfSearch:
    """Test okf_search tool."""

    def test_search_basic(self, sample_bundle):
        """Test basic search."""
        import markrag.server as server_module

        server_module._bundle = sample_bundle
        server_module._bundle_path = sample_bundle.root_path

        result = okf_search(query="Spring Boot")

        assert "results" in result
        assert len(result["results"]) > 0
        assert "query" in result
        assert result["query"] == "Spring Boot"

    def test_search_type_filter(self, sample_bundle):
        """Test search with type filter."""
        import markrag.server as server_module

        server_module._bundle = sample_bundle
        server_module._bundle_path = sample_bundle.root_path

        result = okf_search(query="", type_="Technology")

        for hit in result["results"]:
            assert hit["type"] == "Technology"

    def test_search_status_filter(self, sample_bundle):
        """Test search with status filter."""
        import markrag.server as server_module

        server_module._bundle = sample_bundle
        server_module._bundle_path = sample_bundle.root_path

        result = okf_search(query="", status="stable")

        for hit in result["results"]:
            assert hit["status"] == "stable"

    def test_search_top_k(self, sample_bundle):
        """Test search with top_k limit."""
        import markrag.server as server_module

        server_module._bundle = sample_bundle
        server_module._bundle_path = sample_bundle.root_path

        result = okf_search(query="", top_k=2)

        assert len(result["results"]) <= 2

    def test_search_excerpt_length(self, sample_bundle):
        """Test that excerpts are limited."""
        import markrag.server as server_module

        server_module._bundle = sample_bundle
        server_module._bundle_path = sample_bundle.root_path

        result = okf_search(query="")

        for hit in result["results"]:
            excerpt = hit.get("excerpt", "")
            # Excerpt should be limited to ~40 tokens
            word_count = len(excerpt.split())
            assert word_count <= 45  # Allow some margin

    def test_search_related_links(self, sample_bundle):
        """Test that related links are included."""
        # Add a concept with links
        content = """---
type: Reference
title: Links Test
---

See [Spring Boot](/technologies/spring-boot.md).
"""
        file_path = sample_bundle.root_path / "links.md"
        with open(file_path, "w") as f:
            f.write(content)

        sample_bundle.load()

        import markrag.server as server_module

        server_module._bundle = sample_bundle
        server_module._bundle_path = sample_bundle.root_path

        result = okf_search(query="Links")

        for hit in result["results"]:
            if hit["concept_id"] == "links":
                assert "related" in hit
                assert len(hit["related"]) > 0


class TestOkfRead:
    """Test okf_read tool."""

    def test_read_concept(self, sample_bundle):
        """Test reading a concept."""
        import markrag.server as server_module

        server_module._bundle = sample_bundle
        server_module._bundle_path = sample_bundle.root_path

        result = okf_read(concept_id="technologies/spring-boot")

        assert "concept_id" in result
        assert result["concept_id"] == "technologies/spring-boot"
        assert "content" in result
        assert "frontmatter" in result
        assert "trust_level" in result

    def test_read_nonexistent_concept(self, sample_bundle):
        """Test reading a non-existent concept."""
        import markrag.server as server_module

        server_module._bundle = sample_bundle
        server_module._bundle_path = sample_bundle.root_path

        result = okf_read(concept_id="nonexistent")

        assert "error" in result

    def test_read_reserved_file(self, sample_bundle):
        """Test reading a reserved file."""
        import markrag.server as server_module

        server_module._bundle = sample_bundle
        server_module._bundle_path = sample_bundle.root_path

        result = okf_read(concept_id="index")

        assert "error" in result
        assert "reserved" in result["error"]

    def test_read_section(self, sample_bundle):
        """Test reading a specific section."""
        import markrag.server as server_module

        server_module._bundle = sample_bundle
        server_module._bundle_path = sample_bundle.root_path

        result = okf_read(concept_id="technologies/database", section="SQL")

        assert "section" in result
        assert result["section"] == "SQL"
        assert "content" in result
        assert "Structured Query Language" in result["content"]

    def test_read_trust_line(self, sample_bundle):
        """Test that trust line is included."""
        import markrag.server as server_module

        server_module._bundle = sample_bundle
        server_module._bundle_path = sample_bundle.root_path

        result = okf_read(concept_id="technologies/spring-boot")

        assert "trust_line" in result
        assert "Trust:" in result["trust_line"]

    def test_read_sources(self, sample_bundle):
        """Test that sources are included."""
        # Add a concept with sources
        content = """---
type: Reference
title: With Sources
sources:
  - id: src1
    resource: https://example.com
    title: Example
---

Content.
"""
        file_path = sample_bundle.root_path / "sources.md"
        with open(file_path, "w") as f:
            f.write(content)

        sample_bundle.load()

        import markrag.server as server_module

        server_module._bundle = sample_bundle
        server_module._bundle_path = sample_bundle.root_path

        result = okf_read(concept_id="sources")

        assert "sources" in result
        assert len(result["sources"]) == 1
        assert result["sources"][0]["id"] == "src1"


class TestOkfWrite:
    """Test okf_write tool."""

    def test_write_new_concept(self, temp_wiki_dir):
        """Test writing a new concept."""
        import markrag.server as server_module

        server_module._bundle = None
        server_module._bundle_path = None

        os.environ["MARKRAG_REPO"] = str(temp_wiki_dir.parent / "test")

        result = okf_write(
            concept_id="new/concept",
            title="New Concept",
            description="A new concept",
            type_="Test",
            body_markdown="# New Concept\n\nThis is new.",
        )

        assert "concept_id" in result
        assert result["concept_id"] == "new/concept"
        assert "path" in result
        assert "written" in result["status"]

        # Verify file was created
        file_path = Path(result["path"])
        assert file_path.exists()

        # Verify content
        with open(file_path) as f:
            content = f.read()

        assert "type: Test" in content
        assert "title: New Concept" in content
        assert "status: draft" in content
        assert "generated:" in content
        assert "markrag/" in content

    def test_write_preserves_verified(self, temp_wiki_dir):
        """Test that existing verified history is preserved."""
        # First, create a concept with verification
        content = """---
type: Test
title: Existing
verified:
  - by: human:john
    at: 2026-07-09T14:30:00Z
---

Existing content.
"""
        file_path = temp_wiki_dir / "existing.md"
        with open(file_path, "w") as f:
            f.write(content)

        import markrag.server as server_module

        server_module._bundle = None
        server_module._bundle_path = None

        os.environ["MARKRAG_REPO"] = str(temp_wiki_dir.parent / "test")

        # Write to update the concept
        okf_write(
            concept_id="existing",
            title="Updated",
            description="Updated description",
            type_="Test",
            body_markdown="# Updated\n\nNew content.",
        )

        # Verify file was updated
        with open(file_path) as f:
            content = f.read()

        # Should preserve verified
        assert "verified:" in content
        assert "human:john" in content

    def test_write_stamps_generated(self, temp_wiki_dir):
        """Test that write always stamps generated metadata."""
        import markrag.server as server_module

        server_module._bundle = None
        server_module._bundle_path = None

        os.environ["MARKRAG_REPO"] = str(temp_wiki_dir.parent / "test")

        result = okf_write(
            concept_id="stamped",
            title="Stamped",
            description="Description",
            type_="Test",
            body_markdown="Content",
        )

        file_path = Path(result["path"])
        with open(file_path) as f:
            content = f.read()

        assert "generated:" in content
        assert "by: markrag/" in content
        assert "status: draft" in content

    def test_write_appends_to_log(self, temp_wiki_dir):
        """Test that write appends to log.md."""
        import markrag.server as server_module

        server_module._bundle = None
        server_module._bundle_path = None

        os.environ["MARKRAG_REPO"] = str(temp_wiki_dir.parent / "test")

        # First write
        okf_write(
            concept_id="first",
            title="First",
            description="First concept",
            type_="Test",
            body_markdown="Content",
        )

        log_path = temp_wiki_dir / "log.md"
        assert log_path.exists()

        with open(log_path) as f:
            log_content = f.read()

        assert "first" in log_content.lower() or "created" in log_content.lower()

    def test_write_updates_the_directory_log(self, temp_wiki_dir):
        """A write is recorded in the log of every directory it touches."""
        import markrag.server as server_module

        server_module._bundle = None
        server_module._bundle_path = None

        os.environ["MARKRAG_REPO"] = str(temp_wiki_dir.parent / "test")

        okf_write(
            concept_id="kb/frontend/guide",
            title="Guide",
            description="A frontend guide",
            type_="Test",
            body_markdown="Content",
        )

        for relative in ("log.md", "kb/log.md", "kb/frontend/log.md"):
            log_path = temp_wiki_dir / relative
            assert log_path.exists(), f"missing {relative}"
            assert "guide" in log_path.read_text(encoding="utf-8").lower()

    def test_write_regenerates_every_index(self, temp_wiki_dir):
        """A write refreshes the root index and the directory indexes."""
        import markrag.server as server_module

        server_module._bundle = None
        server_module._bundle_path = None

        os.environ["MARKRAG_REPO"] = str(temp_wiki_dir.parent / "test")

        okf_write(
            concept_id="kb/frontend/guide",
            title="Guide",
            description="A frontend guide",
            type_="Test",
            body_markdown="Content",
        )

        root_index = (temp_wiki_dir / "index.md").read_text(encoding="utf-8")
        directory_index = (temp_wiki_dir / "kb" / "frontend" / "index.md").read_text(
            encoding="utf-8"
        )

        assert "/kb/frontend/guide.md" in root_index
        assert "/kb/frontend/guide.md" in directory_index

    def test_write_never_touches_the_home_page(self, temp_wiki_dir):
        """home.md belongs to the user and is never rewritten."""
        import markrag.server as server_module

        server_module._bundle = None
        server_module._bundle_path = None

        os.environ["MARKRAG_REPO"] = str(temp_wiki_dir.parent / "test")

        home_path = temp_wiki_dir / "Home.md"
        home_path.write_text("# Hand written home\n", encoding="utf-8")

        okf_write(
            concept_id="first",
            title="First",
            description="First concept",
            type_="Test",
            body_markdown="Content",
        )

        assert home_path.read_text(encoding="utf-8") == "# Hand written home\n"


class TestOkfWriteHardening:
    """Test the write path's rejection and preservation rules."""

    @pytest.mark.parametrize(
        "concept_id",
        [
            "../escape",
            "nested/../../escape",
            "/absolute/path",
            "..",
            ".",
            "",
        ],
    )
    def test_write_rejects_path_traversal(self, temp_wiki_dir, concept_id):
        """A concept ID can never escape the wiki directory."""
        import markrag.server as server_module

        server_module._bundle = None
        server_module._bundle_path = None
        os.environ["MARKRAG_REPO"] = str(temp_wiki_dir.parent / "test")

        result = okf_write(
            concept_id=concept_id,
            title="Traversal",
            description="Attempt to escape",
            type_="Test",
            body_markdown="Content",
        )

        assert "error" in result
        assert not (temp_wiki_dir.parent / "escape.md").exists()

    def test_write_rejects_reserved_files(self, temp_wiki_dir):
        """Server-managed files are never written by a tool call."""
        import markrag.server as server_module

        server_module._bundle = None
        server_module._bundle_path = None
        os.environ["MARKRAG_REPO"] = str(temp_wiki_dir.parent / "test")

        for reserved in ("index", "home", "log"):
            result = okf_write(
                concept_id=reserved,
                title="Reserved",
                description="Attempt to overwrite",
                type_="Test",
                body_markdown="Content",
            )
            assert "error" in result

    def test_write_rejects_the_github_shell_pages(self, temp_wiki_dir):
        """The wiki's chrome cannot be clobbered by a tool call."""
        import markrag.server as server_module

        server_module._bundle = None
        server_module._bundle_path = None
        os.environ["MARKRAG_REPO"] = str(temp_wiki_dir.parent / "test")

        for shell in ("_Sidebar", "_Footer", "_Header"):
            result = okf_write(
                concept_id=shell,
                title="Shell",
                description="Attempt to overwrite",
                type_="Test",
                body_markdown="Content",
            )
            assert "error" in result
            assert "chrome" in result["error"]
            assert not (temp_wiki_dir / f"{shell}.md").exists()

    def test_write_rejects_reserved_names_at_any_depth(self, temp_wiki_dir):
        """A directory index cannot be overwritten through a nested path."""
        import markrag.server as server_module

        server_module._bundle = None
        server_module._bundle_path = None
        os.environ["MARKRAG_REPO"] = str(temp_wiki_dir.parent / "test")

        for reserved in ("kb/index", "kb/log", "kb/home"):
            result = okf_write(
                concept_id=reserved,
                title="Reserved",
                description="Attempt to overwrite",
                type_="Test",
                body_markdown="Content",
            )
            assert "error" in result
            assert not (temp_wiki_dir / f"{reserved}.md").exists()

    def test_write_rejects_reserved_names_case_insensitively(self, temp_wiki_dir):
        """On a case-insensitive filesystem, "Index" is "index.md"."""
        import markrag.server as server_module

        server_module._bundle = None
        server_module._bundle_path = None
        os.environ["MARKRAG_REPO"] = str(temp_wiki_dir.parent / "test")

        for reserved in ("Index", "LOG", "Home"):
            result = okf_write(
                concept_id=reserved,
                title="Reserved",
                description="Attempt to overwrite",
                type_="Test",
                body_markdown="Content",
            )
            assert "error" in result
            assert "cannot be written" in result["error"]

    def test_write_preserves_unknown_frontmatter_fields(self, temp_wiki_dir):
        """Fields markrag does not understand survive a write untouched."""
        import markrag.server as server_module

        server_module._bundle = None
        server_module._bundle_path = None
        os.environ["MARKRAG_REPO"] = str(temp_wiki_dir.parent / "test")

        file_path = temp_wiki_dir / "custom.md"
        file_path.write_text(
            """---
type: Test
title: Custom
custom_field: keep-me
another_field:
  nested: value
---

Original body.
""",
            encoding="utf-8",
        )

        okf_write(
            concept_id="custom",
            title="Custom Updated",
            description="Updated description",
            type_="Test",
            body_markdown="# Custom Updated\n\nNew body.",
        )

        import frontmatter

        metadata = dict(frontmatter.load(file_path).metadata)
        assert metadata["custom_field"] == "keep-me"
        assert metadata["another_field"] == {"nested": "value"}

    def test_write_preserves_sources_and_stale_after(self, temp_wiki_dir):
        """Provenance and expiry metadata survive a write."""
        import markrag.server as server_module

        server_module._bundle = None
        server_module._bundle_path = None
        os.environ["MARKRAG_REPO"] = str(temp_wiki_dir.parent / "test")

        file_path = temp_wiki_dir / "sourced.md"
        file_path.write_text(
            """---
type: Test
title: Sourced
stale_after: 2030-01-01T00:00:00Z
sources:
  - id: src1
    resource: https://example.com/doc
    title: Example
---

Original body.
""",
            encoding="utf-8",
        )

        okf_write(
            concept_id="sourced",
            title="Sourced Updated",
            description="Updated description",
            type_="Test",
            body_markdown="# Sourced Updated\n\nNew body.",
        )

        import frontmatter

        metadata: dict[str, object] = dict(frontmatter.load(file_path).metadata)
        assert metadata["stale_after"] == "2030-01-01T00:00:00Z"
        sources = metadata["sources"]
        assert isinstance(sources, list)
        assert sources[0]["resource"] == "https://example.com/doc"

    def test_write_never_creates_verification(self, temp_wiki_dir):
        """A write must never claim the content was verified."""
        import markrag.server as server_module

        server_module._bundle = None
        server_module._bundle_path = None
        os.environ["MARKRAG_REPO"] = str(temp_wiki_dir.parent / "test")

        result = okf_write(
            concept_id="unverified",
            title="Unverified",
            description="Written by a tool",
            type_="Test",
            body_markdown="Content",
        )

        import frontmatter

        metadata = dict(frontmatter.load(Path(result["path"])).metadata)
        assert not metadata.get("verified")

    def test_write_resets_status_to_draft(self, temp_wiki_dir):
        """Rewriting a stable document sends it back to draft."""
        import markrag.server as server_module

        server_module._bundle = None
        server_module._bundle_path = None
        os.environ["MARKRAG_REPO"] = str(temp_wiki_dir.parent / "test")

        file_path = temp_wiki_dir / "stable.md"
        file_path.write_text(
            "---\ntype: Test\ntitle: Stable\nstatus: stable\n---\n\nBody.\n",
            encoding="utf-8",
        )

        okf_write(
            concept_id="stable",
            title="Stable",
            description="Rewritten",
            type_="Test",
            body_markdown="# Stable\n\nRewritten body.",
        )

        import frontmatter

        assert dict(frontmatter.load(file_path).metadata)["status"] == "draft"

    def test_write_rejects_a_source_without_a_resource(self, temp_wiki_dir):
        """A source that points nowhere is rejected."""
        import markrag.server as server_module

        server_module._bundle = None
        server_module._bundle_path = None
        os.environ["MARKRAG_REPO"] = str(temp_wiki_dir.parent / "test")

        result = okf_write(
            concept_id="badsource",
            title="Bad Source",
            description="Source has no resource",
            type_="Test",
            body_markdown="Content",
            sources=[{"id": "src1", "title": "No resource"}],
        )

        assert "error" in result


class TestSectionRead:
    """Test exact section matching."""

    def test_read_section_is_exact(self, sample_bundle):
        """A section name must match the heading exactly."""
        import markrag.server as server_module

        server_module._bundle = sample_bundle
        server_module._bundle_path = sample_bundle.root_path

        result = okf_read(concept_id="technologies/database", section="SQL")
        assert "error" not in result
        assert "Structured Query Language" in result["content"]

    def test_read_unknown_section_is_reported(self, sample_bundle):
        """A missing section is an error, not an empty document."""
        import markrag.server as server_module

        server_module._bundle = sample_bundle
        server_module._bundle_path = sample_bundle.root_path

        result = okf_read(concept_id="technologies/database", section="No Such Section")
        assert "error" in result

    def test_read_section_does_not_leak_other_sections(self, sample_bundle):
        """Reading one section returns only that section."""
        import markrag.server as server_module

        (sample_bundle.root_path / "technologies" / "two-sections.md").write_text(
            """---
type: Technology
title: Two Sections
description: A concept with two sections
---

## First

The first section body.

## Second

The second section body.
""",
            encoding="utf-8",
        )
        sample_bundle.load()

        server_module._bundle = sample_bundle
        server_module._bundle_path = sample_bundle.root_path

        result = okf_read(concept_id="technologies/two-sections", section="First")
        assert "The first section body." in result["content"]
        assert "The second section body." not in result["content"]


class TestOkfStatus:
    """Test okf_status tool."""

    def test_status_basic(self, sample_bundle):
        """Test basic status."""
        import markrag.server as server_module

        server_module._bundle = sample_bundle
        server_module._bundle_path = sample_bundle.root_path

        result = okf_status()

        assert "total_docs" in result
        assert result["total_docs"] == 3
        assert "by_type" in result
        assert "by_status" in result
        assert "by_trust_level" in result

    def test_status_by_type(self, sample_bundle):
        """Test status by type."""
        import markrag.server as server_module

        server_module._bundle = sample_bundle
        server_module._bundle_path = sample_bundle.root_path

        result = okf_status()

        assert "Technology" in result["by_type"]
        assert result["by_type"]["Technology"] == 2
        assert "Process" in result["by_type"]
        assert result["by_type"]["Process"] == 1

    def test_status_by_trust_level(self, sample_bundle):
        """Test status by trust level."""
        import markrag.server as server_module

        server_module._bundle = sample_bundle
        server_module._bundle_path = sample_bundle.root_path

        result = okf_status()

        assert "unverified" in result["by_trust_level"]
        assert "human-reviewed" in result["by_trust_level"]

    def test_status_stale_docs(self, sample_bundle):
        """Test stale docs listing."""
        # Add a stale concept to the bundle
        content = """---
type: Test
title: Stale Concept
stale_after: 2020-01-01T00:00:00Z
---

Old content.
"""
        file_path = sample_bundle.root_path / "stale.md"
        with open(file_path, "w") as f:
            f.write(content)

        # Reload bundle
        sample_bundle.load()

        import markrag.server as server_module

        server_module._bundle = sample_bundle
        server_module._bundle_path = sample_bundle.root_path

        result = okf_status()

        assert "stale_docs" in result
        assert len(result["stale_docs"]) == 1
        assert result["stale_docs"][0]["concept_id"] == "stale"
        assert result["stale_docs"][0]["days_past"] > 0


class TestDisclosureLadder:
    """Test the 3-tier disclosure ladder."""

    def test_tier0_no_bodies(self, sample_bundle):
        """Test that tier 0 (index) returns no bodies."""
        import markrag.server as server_module

        server_module._bundle = sample_bundle
        server_module._bundle_path = sample_bundle.root_path

        result = okf_index()

        for concept in result["concepts"]:
            assert "content" not in concept
            assert "body" not in concept

    def test_tier1_no_full_bodies(self, sample_bundle):
        """Test that tier 1 (search) returns no full bodies."""
        import markrag.server as server_module

        server_module._bundle = sample_bundle
        server_module._bundle_path = sample_bundle.root_path

        result = okf_search(query="")

        for hit in result["results"]:
            assert "content" not in hit
            # Excerpt should be limited
            if "excerpt" in hit:
                word_count = len(hit["excerpt"].split())
                assert word_count <= 45

    def test_tier2_full_content(self, sample_bundle):
        """Test that tier 2 (read) can return full content."""
        import markrag.server as server_module

        server_module._bundle = sample_bundle
        server_module._bundle_path = sample_bundle.root_path

        result = okf_read(concept_id="technologies/spring-boot")

        assert "content" in result
        # Full content should be returned
        assert len(result["content"]) > 0

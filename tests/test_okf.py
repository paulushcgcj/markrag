"""Tests for OKF format handling."""

import pytest

from markrag.okf import (
    Bundle,
    Concept,
    Frontmatter,
    RESERVED_FILES,
    Source,
    Verified,
    Generated,
    derive_trust_level,
    is_stale,
    get_stale_days,
)


class TestFrontmatter:
    """Test frontmatter parsing and serialization."""

    def test_parse_basic(self, temp_wiki_dir):
        """Test parsing basic frontmatter."""
        content = """---
type: Technology
title: Test Tech
description: A test technology
---

Body content.
"""
        file_path = temp_wiki_dir / "test.md"
        with open(file_path, "w") as f:
            f.write(content)

        concept = Concept.from_file(file_path)
        assert concept.frontmatter.type == "Technology"
        assert concept.frontmatter.title == "Test Tech"
        assert concept.frontmatter.description == "A test technology"
        assert concept.body == "Body content."

    def test_parse_with_sources(self, temp_wiki_dir):
        """Test parsing frontmatter with sources."""
        content = """---
type: Reference
title: API Docs
sources:
  - id: api-ref
    resource: https://api.example.com/docs
    title: API Reference
  - id: tutorial
    resource: https://api.example.com/tutorial
---

Content.
"""
        file_path = temp_wiki_dir / "api.md"
        with open(file_path, "w") as f:
            f.write(content)

        concept = Concept.from_file(file_path)
        assert len(concept.frontmatter.sources) == 2
        assert concept.frontmatter.sources[0].id == "api-ref"
        assert concept.frontmatter.sources[0].resource == "https://api.example.com/docs"
        assert concept.frontmatter.sources[0].title == "API Reference"
        assert concept.frontmatter.sources[1].id == "tutorial"
        assert concept.frontmatter.sources[1].title is None

    def test_parse_generated(self, temp_wiki_dir):
        """Test parsing generated metadata."""
        content = """---
type: Generated
title: AutoDoc
generated: {by: markrag/0.1, at: 2026-07-09T14:30:00Z}
---

Generated content.
"""
        file_path = temp_wiki_dir / "generated.md"
        with open(file_path, "w") as f:
            f.write(content)

        concept = Concept.from_file(file_path)
        assert concept.frontmatter.generated is not None
        assert concept.frontmatter.generated.by == "markrag/0.1"
        assert concept.frontmatter.generated.at == "2026-07-09T14:30:00Z"

    def test_parse_verified(self, temp_wiki_dir):
        """Test parsing verified metadata."""
        content = """---
type: Process
title: Deployment
verified:
  - by: human:john
    at: 2026-07-10T10:00:00Z
  - by: process:ci-check
    at: 2026-07-10T11:00:00Z
---

Process content.
"""
        file_path = temp_wiki_dir / "process.md"
        with open(file_path, "w") as f:
            f.write(content)

        concept = Concept.from_file(file_path)
        assert len(concept.frontmatter.verified) == 2
        assert concept.frontmatter.verified[0].by == "human:john"
        assert concept.frontmatter.verified[1].by == "process:ci-check"

    def test_preserve_unknown_fields(self, temp_wiki_dir):
        """Test that unknown fields are preserved."""
        content = """---
type: Custom
custom_field: custom_value
another_field: 42
---

Content.
"""
        file_path = temp_wiki_dir / "custom.md"
        with open(file_path, "w") as f:
            f.write(content)

        concept = Concept.from_file(file_path)
        assert "custom_field" in concept.frontmatter.unknown_fields
        assert concept.frontmatter.unknown_fields["custom_field"] == "custom_value"
        assert concept.frontmatter.unknown_fields["another_field"] == 42

    def test_roundtrip(self, temp_wiki_dir):
        """Test that parsing and serialization preserves all data."""
        original_content = """---
type: Technology
title: Test Tech
description: A test
custom_field: value
tags: [a, b]
---

Body.
"""
        file_path = temp_wiki_dir / "roundtrip.md"
        with open(file_path, "w") as f:
            f.write(original_content)

        concept = Concept.from_file(file_path)
        markdown = concept.to_markdown()

        # Re-parse
        concept2 = Concept.from_content(file_path, markdown)

        assert concept2.frontmatter.type == concept.frontmatter.type
        assert concept2.frontmatter.title == concept.frontmatter.title
        assert concept2.frontmatter.description == concept.frontmatter.description
        assert concept2.frontmatter.tags == concept.frontmatter.tags
        assert "custom_field" in concept2.frontmatter.unknown_fields

    def test_missing_optional_fields(self, temp_wiki_dir):
        """Test that missing optional fields don't cause errors."""
        content = """---
type: Minimal
---

Body.
"""
        file_path = temp_wiki_dir / "minimal.md"
        with open(file_path, "w") as f:
            f.write(content)

        concept = Concept.from_file(file_path)
        assert concept.frontmatter.type == "Minimal"
        assert concept.frontmatter.title is None
        assert concept.frontmatter.description is None
        assert concept.frontmatter.tags == []
        assert concept.frontmatter.status == "stable"  # Default


class TestBundle:
    """Test bundle loading and querying."""

    def test_load_bundle(self, sample_bundle):
        """Test loading a bundle."""
        assert len(sample_bundle.concepts) == 3
        assert "technologies/spring-boot" in sample_bundle.concepts
        assert "technologies/database" in sample_bundle.concepts
        assert "processes/deployment" in sample_bundle.concepts

    def test_reserved_files_excluded(self, temp_wiki_dir):
        """Test that reserved files are excluded from concepts."""
        # Create reserved files
        for filename in RESERVED_FILES:
            file_path = temp_wiki_dir / filename
            with open(file_path, "w") as f:
                f.write(f"# {filename}\n")

        # Create a regular concept
        file_path = temp_wiki_dir / "concept.md"
        with open(file_path, "w") as f:
            f.write("---\ntype: Concept\n---\nContent.")

        bundle = Bundle(root_path=temp_wiki_dir)
        bundle.load()

        # Reserved files should not be in concepts
        for filename in RESERVED_FILES:
            assert filename not in bundle.reserved_files or filename not in [
                c.filename for c in bundle.concepts.values()
            ]
        assert "concept" in bundle.concepts

    def test_get_concept(self, sample_bundle):
        """Test getting a concept by ID."""
        concept = sample_bundle.get_concept("technologies/spring-boot")
        assert concept is not None
        assert concept.concept_id == "technologies/spring-boot"
        assert concept.frontmatter.type == "Technology"

        # Non-existent concept
        assert sample_bundle.get_concept("nonexistent") is None

    def test_get_concepts_by_type(self, sample_bundle):
        """Test filtering concepts by type."""
        tech_concepts = sample_bundle.get_concepts_by_type("Technology")
        assert len(tech_concepts) == 2

        process_concepts = sample_bundle.get_concepts_by_type("Process")
        assert len(process_concepts) == 1

    def test_get_concepts_by_tag(self, sample_bundle):
        """Test filtering concepts by tag."""
        java_concepts = sample_bundle.get_concepts_by_tag("java")
        assert len(java_concepts) == 1

        cd_concepts = sample_bundle.get_concepts_by_tag("cd")
        assert len(cd_concepts) == 1

    def test_get_concepts_by_status(self, sample_bundle):
        """Test filtering concepts by status."""
        stable_concepts = sample_bundle.get_concepts_by_status("stable")
        assert len(stable_concepts) == 2

        draft_concepts = sample_bundle.get_concepts_by_status("draft")
        assert len(draft_concepts) == 1

    def test_extract_links(self, sample_bundle):
        """Test extracting links from concepts."""
        # Add a concept with links
        content = """---
type: Reference
title: Links Test
---

# Links

See [Spring Boot](/technologies/spring-boot.md) and [Database](/technologies/database.md).
"""
        file_path = sample_bundle.root_path / "links.md"
        with open(file_path, "w") as f:
            f.write(content)

        # Reload bundle
        sample_bundle.load()

        links_concept = sample_bundle.get_concept("links")
        assert links_concept is not None

        links = links_concept.extract_links()
        assert "technologies/spring-boot" in links
        assert "technologies/database" in links


class TestTrustLevel:
    """Test trust level derivation."""

    def test_unverified(self, temp_wiki_dir):
        """Test unverified trust level."""
        content = """---
type: Test
title: Unverified
---

Content.
"""
        file_path = temp_wiki_dir / "unverified.md"
        with open(file_path, "w") as f:
            f.write(content)

        concept = Concept.from_file(file_path)
        assert derive_trust_level(concept) == "unverified"

    def test_machine_confirmed(self, temp_wiki_dir):
        """Test machine-confirmed trust level."""
        content = """---
type: Test
title: Machine Confirmed
verified:
  - by: producer/markrag/0.1
    at: 2026-07-09T14:30:00Z
---

Content.
"""
        file_path = temp_wiki_dir / "machine.md"
        with open(file_path, "w") as f:
            f.write(content)

        concept = Concept.from_file(file_path)
        assert derive_trust_level(concept) == "machine-confirmed"

    def test_machine_confirmed_process(self, temp_wiki_dir):
        """Test machine-confirmed with process verifier."""
        content = """---
type: Test
title: Process Verified
verified:
  - by: process:ci-check
    at: 2026-07-09T14:30:00Z
---

Content.
"""
        file_path = temp_wiki_dir / "process.md"
        with open(file_path, "w") as f:
            f.write(content)

        concept = Concept.from_file(file_path)
        assert derive_trust_level(concept) == "machine-confirmed"

    def test_human_reviewed(self, temp_wiki_dir):
        """Test human-reviewed trust level."""
        content = """---
type: Test
title: Human Reviewed
verified:
  - by: human:john
    at: 2026-07-09T14:30:00Z
---

Content.
"""
        file_path = temp_wiki_dir / "human.md"
        with open(file_path, "w") as f:
            f.write(content)

        concept = Concept.from_file(file_path)
        assert derive_trust_level(concept) == "human-reviewed"

    def test_human_reviewed_with_machine(self, temp_wiki_dir):
        """Test human-reviewed when both human and machine verifiers exist."""
        content = """---
type: Test
title: Mixed Verifiers
verified:
  - by: process:ci-check
    at: 2026-07-09T14:30:00Z
  - by: human:john
    at: 2026-07-10T10:00:00Z
---

Content.
"""
        file_path = temp_wiki_dir / "mixed.md"
        with open(file_path, "w") as f:
            f.write(content)

        concept = Concept.from_file(file_path)
        # Should be human-reviewed because human verifier exists
        assert derive_trust_level(concept) == "human-reviewed"


class TestStaleness:
    """Test staleness checking."""

    def test_not_stale(self, temp_wiki_dir):
        """Test document that is not stale."""
        content = """---
type: Test
title: Not Stale
stale_after: 2099-01-01T00:00:00Z
---

Content.
"""
        file_path = temp_wiki_dir / "notstale.md"
        with open(file_path, "w") as f:
            f.write(content)

        concept = Concept.from_file(file_path)
        assert not is_stale(concept)
        assert get_stale_days(concept) is None

    def test_stale(self, temp_wiki_dir):
        """Test document that is stale."""
        content = """---
type: Test
title: Stale
stale_after: 2020-01-01T00:00:00Z
---

Content.
"""
        file_path = temp_wiki_dir / "stale.md"
        with open(file_path, "w") as f:
            f.write(content)

        concept = Concept.from_file(file_path)
        assert is_stale(concept)
        # Should be stale by several years
        assert get_stale_days(concept) is not None
        assert get_stale_days(concept) > 0

    def test_no_stale_after(self, temp_wiki_dir):
        """Test document without stale_after field."""
        content = """---
type: Test
title: No Stale After
---

Content.
"""
        file_path = temp_wiki_dir / "nostale.md"
        with open(file_path, "w") as f:
            f.write(content)

        concept = Concept.from_file(file_path)
        assert not is_stale(concept)
        assert get_stale_days(concept) is None


class TestConformance:
    """Test OKF conformance requirements."""

    def test_missing_optional_fields_valid(self, temp_wiki_dir):
        """Test that missing optional fields don't fail reads."""
        content = """---
type: Minimal
---

Body.
"""
        file_path = temp_wiki_dir / "minimal.md"
        with open(file_path, "w") as f:
            f.write(content)

        # Should not raise
        concept = Concept.from_file(file_path)
        assert concept.frontmatter.type == "Minimal"

    def test_unknown_type_valid(self, temp_wiki_dir):
        """Test that unknown types don't fail reads."""
        content = """---
type: UnknownCustomType
---

Body.
"""
        file_path = temp_wiki_dir / "unknown.md"
        with open(file_path, "w") as f:
            f.write(content)

        concept = Concept.from_file(file_path)
        assert concept.frontmatter.type == "UnknownCustomType"

    def test_extra_fields_preserved(self, temp_wiki_dir):
        """Test that extra fields are preserved."""
        content = """---
type: Test
extra1: value1
extra2: value2
---

Body.
"""
        file_path = temp_wiki_dir / "extra.md"
        with open(file_path, "w") as f:
            f.write(content)

        concept = Concept.from_file(file_path)
        assert "extra1" in concept.frontmatter.unknown_fields
        assert concept.frontmatter.unknown_fields["extra1"] == "value1"

    def test_broken_links_tolerated(self, temp_wiki_dir):
        """Test that broken links don't fail reads."""
        content = """---
type: Test
title: Broken Links
---

See [Broken Link](/nonexistent.md) and [Another Broken](https://example.com/404).
"""
        file_path = temp_wiki_dir / "broken.md"
        with open(file_path, "w") as f:
            f.write(content)

        # Should not raise
        concept = Concept.from_file(file_path)
        links = concept.extract_links()
        assert "nonexistent" in links


class TestConceptID:
    """Test concept ID derivation."""

    def test_root_level_concept(self, temp_wiki_dir):
        """Test concept ID for root-level files."""
        content = """---
type: Test
---

Body.
"""
        file_path = temp_wiki_dir / "root.md"
        with open(file_path, "w") as f:
            f.write(content)

        concept = Concept.from_file(file_path)
        assert concept.concept_id == "root"

    def test_nested_concept(self, temp_wiki_dir):
        """Test concept ID for nested files."""
        content = """---
type: Test
---

Body.
"""
        nested_dir = temp_wiki_dir / "nested" / "deep"
        nested_dir.mkdir(parents=True)
        file_path = nested_dir / "concept.md"
        with open(file_path, "w") as f:
            f.write(content)

        concept = Concept.from_file(file_path, temp_wiki_dir)
        # Concept ID should be relative path without .md
        assert concept.concept_id == "nested/deep/concept"

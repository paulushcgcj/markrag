"""Pytest configuration for markrag tests."""

import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_wiki_dir():
    """Create a temporary wiki directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        wiki_path = Path(tmpdir) / "test.wiki"
        wiki_path.mkdir()
        yield wiki_path


@pytest.fixture
def sample_concept_file(temp_wiki_dir):
    """Create a sample concept file for testing."""
    concept_content = """---
type: Technology
title: Spring Boot Configuration
description: How configuration profiles are organized in our applications.
tags: [java, spring-boot]
status: stable

generated: {by: markrag/0.1, at: 2026-07-09T14:30:00Z}
stale_after: 2027-01-31T00:00:00Z

sources:
  - {id: official-docs, resource: https://docs.spring.io/spring-boot/, title: Spring Boot Docs}
---

# Spring Boot Configuration

This is the body content.

## Profiles

Spring Boot uses profiles for environment-specific configuration.

## Properties

Configuration properties can be defined in application.properties.
"""
    file_path = temp_wiki_dir / "technologies/spring-boot.md"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w") as f:
        f.write(concept_content)
    return file_path


@pytest.fixture
def sample_bundle(temp_wiki_dir):
    """Create a sample bundle with multiple concepts."""
    from markrag.okf import Bundle

    # Create index.md
    index_path = temp_wiki_dir / "index.md"
    with open(index_path, "w") as f:
        f.write("---\nokf_version: "0.2"\n---\n\n# Bundle Index\n")

    # Create log.md
    log_path = temp_wiki_dir / "log.md"
    with open(log_path, "w") as f:
        f.write("# Change Log\n\n- [2026-07-09T14:30:00Z] Initial commit\n")

    # Create home.md
    home_path = temp_wiki_dir / "home.md"
    with open(home_path, "w") as f:
        f.write("# Wiki Home\n\n")

    # Create sample concepts
    concepts = [
        ("technologies/spring-boot", """---
type: Technology
title: Spring Boot
description: Java framework for building applications
tags: [java, framework]
status: stable
generated: {by: markrag/0.1, at: 2026-07-09T14:30:00Z}
---

# Spring Boot

Great framework.
"""),
        ("technologies/database", """---
type: Technology
title: Database
description: Data storage and retrieval
tags: [data, storage]
status: draft
generated: {by: markrag/0.1, at: 2026-07-09T14:30:00Z}
---

# Database

Stores data.

## SQL

Structured Query Language.
"""),
        ("processes/deployment", """---
type: Process
title: Deployment
description: How to deploy applications
tags: [ci, cd]
status: stable
verified: [{by: human:john, at: 2026-07-10T10:00:00Z}]
generated: {by: markrag/0.1, at: 2026-07-09T14:30:00Z}
---

# Deployment

Deploy to production.
"""),
    ]

    for concept_id, content in concepts:
        file_path = temp_wiki_dir / f"{concept_id}.md"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as f:
            f.write(content)

    bundle = Bundle(root_path=temp_wiki_dir)
    bundle.load()
    return bundle

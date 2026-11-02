"""Command-line interface for markrag.

Provides commands:
- serve: Run the MCP server
- init: Bootstrap the wiki
- search: Search the wiki
- read: Read a concept
"""

import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click

from markrag.okf import (
    Bundle,
    Concept,
    Frontmatter,
    Source,
    RESERVED_FILES,
    derive_trust_level,
    get_stale_days,
    is_stale,
)


def _get_repo_path(ctx: click.Context) -> Path:
    """Get the repo path from context, env, or default."""
    repo_path = ctx.params.get("repo")
    if not repo_path:
        repo_path = os.getenv("MARKRAG_REPO", ".")
    return Path(repo_path).resolve()


def _get_wiki_path(repo_path: Path) -> Path:
    """Get the wiki path from repo path (sibling convention)."""
    return repo_path.parent / f"{repo_path.name}.wiki"


def _format_output(data: dict[str, Any], format_type: str) -> str:
    """Format output as text or JSON."""
    if format_type == "json":
        import json
        return json.dumps(data, indent=2, default=str)
    return _format_text(data)


def _format_text(data: dict[str, Any]) -> str:
    """Format data as human-readable text."""
    if "error" in data:
        return f"Error: {data['error']}"

    if "concepts" in data:
        lines = [f"Bundle: {data.get('path', '.')}", f"Total concepts: {data['count']}", ""]
        for concept in data["concepts"]:
            trust = concept.get("trust_level", "unverified")
            stale = " [STALE]" if concept.get("stale") else ""
            lines.append(
                f"  {concept['concept_id']}: {concept.get('title', 'Untitled')} "
                f"({concept['type']}) - {concept.get('description', '')[:60]}...{stale}"
            )
        return "\n".join(lines)

    if "results" in data:
        lines = [f"Query: {data.get('query', '')}", f"Results: {len(data['results'])}", ""]
        for i, result in enumerate(data["results"], 1):
            trust = result.get("trust_level", "unverified")
            stale = " [STALE]" if result.get("stale") else ""
            excerpt = result.get("excerpt", "")[:80]
            lines.append(f"  {i}. {result['concept_id']}: {result.get('title', 'Untitled')}{stale}")
            lines.append(f"     Trust: {trust} | Type: {result.get('type', '')}")
            lines.append(f"     Excerpt: {excerpt}...")
            if result.get("related"):
                related = ", ".join([r["concept_id"] for r in result["related"][:3]])
                lines.append(f"     Related: {related}")
            lines.append("")
        return "\n".join(lines)

    if "content" in data:
        lines = [
            f"Concept: {data.get('concept_id', 'Unknown')}",
            f"Trust: {data.get('trust_line', 'unverified')}",
        ]
        if data.get("stale_warning"):
            lines.append(f"WARNING: {data['stale_warning']}")
        lines.append("")

        if data.get("frontmatter"):
            lines.append("Frontmatter:")
            for key, value in data["frontmatter"].items():
                lines.append(f"  {key}: {value}")
            lines.append("")

        if data.get("sources"):
            lines.append("Sources:")
            for src in data["sources"]:
                lines.append(f"  - {src.get('id', '')}: {src.get('resource', '')}")
            lines.append("")

        lines.append("Content:")
        lines.append("-" * 40)
        lines.append(data["content"])
        return "\n".join(lines)

    if "total_docs" in data:
        lines = [
            f"Bundle Statistics:",
            f"  Total documents: {data['total_docs']}",
            f"  Stale documents: {data.get('stale_count', 0)}",
            f"  Last rebuild: {data.get('last_rebuild', 'Never')}",
            "",
            "By Type:",
        ]
        for type_, count in data.get("by_type", {}).items():
            lines.append(f"  {type_}: {count}")
        lines.append("")
        lines.append("By Status:")
        for status, count in data.get("by_status", {}).items():
            lines.append(f"  {status}: {count}")
        lines.append("")
        lines.append("By Trust Level:")
        for trust, count in data.get("by_trust_level", {}).items():
            lines.append(f"  {trust}: {count}")
        if data.get("stale_docs"):
            lines.append("")
            lines.append("Stale Documents:")
            for doc in data["stale_docs"]:
                lines.append(f"  {doc['concept_id']}: {doc['days_past']} days past")
        return "\n".join(lines)

    return str(data)


@click.group()
@click.option("--repo", "-r", type=click.Path(), help="Path to the repository")
@click.option("--format", "-f", type=click.Choice(["text", "json"]), default="text", help="Output format")
@click.pass_context
def cli(ctx: click.Context, repo: str | None, format: str) -> None:
    """markrag - OKF-backed RAG MCP server for per-repo wikis."""
    ctx.ensure_object(dict)
    ctx.obj["repo"] = repo
    ctx.obj["format"] = format


@cli.command()
@click.option("--repo", "-r", type=click.Path(), help="Path to the repository")
@click.option("--wiki-remote", "-w", envvar="MARKRAG_WIKI_REMOTE", help="Git remote URL for wiki")
@click.pass_context
def init(ctx: click.Context, repo: str | None, wiki_remote: str | None) -> None:
    """Bootstrap the wiki for a repository.

    Creates the wiki folder, initializes git, and generates seed documents
    from the repository structure and README.
    """
    repo_path = _get_repo_path(ctx)
    if repo:
        repo_path = Path(repo).resolve()

    wiki_path = _get_wiki_path(repo_path)
    format_type = ctx.obj.get("format", "text")

    click.echo(f"Initializing wiki at: {wiki_path}")

    # Create wiki directory
    wiki_path.mkdir(parents=True, exist_ok=True)

    # Initialize git repo or clone
    if wiki_remote and not (wiki_path / ".git").exists():
        click.echo(f"Cloning wiki from: {wiki_remote}")
        subprocess.run(["git", "clone", wiki_remote, str(wiki_path)], check=True)
    elif not (wiki_path / ".git").exists():
        click.echo("Initializing git repository")
        subprocess.run(["git", "init"], cwd=wiki_path, check=True)
        subprocess.run(["git", "config", "user.name", "markrag"], cwd=wiki_path, check=True)
        subprocess.run(["git", "config", "user.email", "markrag@localhost"], cwd=wiki_path, check=True)

    # Create initial files
    click.echo("Creating initial files...")

    # Create index.md
    index_content = """---
okf_version: "0.2"
---

# Bundle Index

"""
    with open(wiki_path / "index.md", "w") as f:
        f.write(index_content)

    # Create log.md
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    log_content = f"# Change Log\n\n- [{timestamp}] Bootstrap: Initial wiki created by markrag\n"
    with open(wiki_path / "log.md", "w") as f:
        f.write(log_content)

    # Create home.md
    home_content = """# Wiki Home

This is the GitHub Wiki for the repository.

"""
    with open(wiki_path / "home.md", "w") as f:
        f.write(home_content)

    # Scan the repo and generate seed documents
    click.echo("Generating seed documents from repository...")
    seed_docs = _generate_seed_docs(repo_path, wiki_path)

    for concept_id, content in seed_docs.items():
        file_path = wiki_path / f"{concept_id}.md"
        with open(file_path, "w") as f:
            f.write(content)
        click.echo(f"  Created: {concept_id}.md")

    # Add seed docs to log
    for concept_id in seed_docs:
        log_entry = f"- [{timestamp}] created: [{_extract_title(seed_docs[concept_id])}](/{concept_id}.md) by markrag\n"
        with open(wiki_path / "log.md", "a") as f:
            f.write(log_entry)

    # Update index.md with seed docs
    _update_index_for_seed(wiki_path, seed_docs)

    # Add all files to git
    click.echo("Adding files to git...")
    subprocess.run(["git", "add", "."], cwd=wiki_path, check=True)

    # Commit
    click.echo("Creating initial commit...")
    subprocess.run(["git", "commit", "-m", "Initial commit: Bootstrap wiki with seed documents"], cwd=wiki_path, check=True)

    click.echo(f"\nWiki initialized at: {wiki_path}")
    click.echo("You can now start the server with: markrag serve")


@cli.command()
@click.option("--repo", "-r", type=click.Path(), help="Path to the repository")
@click.option("--host", "-H", default="0.0.0.0", help="Host to bind to")
@click.option("--port", "-p", type=int, default=None, help="Port to bind to")
@click.pass_context
def serve(ctx: click.Context, repo: str | None, host: str, port: int | None) -> None:
    """Run the MCP server.

    The server reads the MARKRAG_REPO environment variable or uses the
    --repo option to determine the repository path.
    """
    repo_path = _get_repo_path(ctx)
    if repo:
        repo_path = Path(repo).resolve()

    # Set environment variable
    os.environ["MARKRAG_REPO"] = str(repo_path)

    click.echo(f"Starting markrag MCP server for: {repo_path}")
    click.echo(f"Wiki path: {_get_wiki_path(repo_path)}")

    # Import and run the server
    from markrag.server import mcp, main as server_main

    # Override host/port if specified
    if host or port:
        # We need to pass these to the server
        # For now, we'll just run with defaults
        pass

    server_main()


@cli.command()
@click.argument("query")
@click.option("--repo", "-r", type=click.Path(), help="Path to the repository")
@click.option("--type", "-t", type=str, default=None, help="Filter by concept type")
@click.option("--tags", type=str, default=None, help="Filter by tags (comma-separated)")
@click.option("--status", type=str, default=None, help="Filter by status")
@click.option("--top-k", "-n", type=int, default=8, help="Number of results")
@click.option("--format", "-f", type=click.Choice(["text", "json"]), default=None, help="Output format")
@click.pass_context
def search(
    ctx: click.Context,
    query: str,
    repo: str | None,
    type: str | None,
    tags: str | None,
    status: str | None,
    top_k: int,
    format: str | None,
) -> None:
    """Search the wiki.

    Performs hybrid search (FTS + vector) with metadata filtering.
    """
    repo_path = _get_repo_path(ctx)
    if repo:
        repo_path = Path(repo).resolve()

    os.environ["MARKRAG_REPO"] = str(repo_path)

    out_format = format or ctx.obj.get("format", "text")

    # Parse tags
    tags_list = None
    if tags:
        tags_list = [t.strip() for t in tags.split(",")]

    # Import and use the server tools
    from markrag.server import okf_search

    result = okf_search(
        query=query,
        type_=type,
        tags=tags_list,
        status=status,
        top_k=top_k,
    )

    click.echo(_format_output(result, out_format))


@cli.command()
@click.argument("concept_id")
@click.option("--section", "-s", type=str, default=None, help="Section to read")
@click.option("--repo", "-r", type=click.Path(), help="Path to the repository")
@click.option("--format", "-f", type=click.Choice(["text", "json"]), default=None, help="Output format")
@click.pass_context
def read(
    ctx: click.Context,
    concept_id: str,
    section: str | None,
    repo: str | None,
    format: str | None,
) -> None:
    """Read a concept from the wiki.

    Retrieves the full document or a specific section.
    """
    repo_path = _get_repo_path(ctx)
    if repo:
        repo_path = Path(repo).resolve()

    os.environ["MARKRAG_REPO"] = str(repo_path)

    out_format = format or ctx.obj.get("format", "text")

    from markrag.server import okf_read

    result = okf_read(concept_id=concept_id, section=section)

    click.echo(_format_output(result, out_format))


@cli.command()
@click.option("--repo", "-r", type=click.Path(), help="Path to the repository")
@click.option("--format", "-f", type=click.Choice(["text", "json"]), default=None, help="Output format")
@click.pass_context
def status(ctx: click.Context, repo: str | None, format: str | None) -> None:
    """Show bundle statistics.

    Displays doc count by type/status/trust level and stale docs list.
    """
    repo_path = _get_repo_path(ctx)
    if repo:
        repo_path = Path(repo).resolve()

    os.environ["MARKRAG_REPO"] = str(repo_path)

    out_format = format or ctx.obj.get("format", "text")

    from markrag.server import okf_status

    result = okf_status()

    click.echo(_format_output(result, out_format))


@cli.command()
@click.option("--repo", "-r", type=click.Path(), help="Path to the repository")
@click.option("--format", "-f", type=click.Choice(["text", "json"]), default=None, help="Output format")
@click.option("--path", "-p", type=str, default=None, help="Subdirectory path")
@click.pass_context
def index(ctx: click.Context, repo: str | None, format: str | None, path: str | None) -> None:
    """Show bundle index.

    Lists all concepts with their metadata (tier 0).
    """
    repo_path = _get_repo_path(ctx)
    if repo:
        repo_path = Path(repo).resolve()

    os.environ["MARKRAG_REPO"] = str(repo_path)

    out_format = format or ctx.obj.get("format", "text")

    from markrag.server import okf_index

    result = okf_index(path=path)

    click.echo(_format_output(result, out_format))


# Helper functions

def _generate_seed_docs(repo_path: Path, wiki_path: Path) -> dict[str, str]:
    """Generate seed documents from the repository.

    Creates ~5-10 draft documents from:
    - README analysis
    - Top-level structure (src/, etc.)
    - Config files

    Args:
        repo_path: Path to the repository
        wiki_path: Path to the wiki

    Returns:
        Dictionary mapping concept_id to markdown content
    """
    seed_docs = {}
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    # 1. README
    readme_path = repo_path / "README.md"
    if readme_path.exists():
        with open(readme_path, "r", encoding="utf-8") as f:
            readme_content = f.read()

        # Extract title from first heading
        title = "Project Overview"
        first_line = readme_content.split("\n")[0]
        if first_line.startswith("# "):
            title = first_line[2:]

        # Extract description from first paragraph
        lines = readme_content.split("\n")
        description = ""
        for line in lines:
            if line.strip() and not line.startswith("#") and not line.startswith("```"):
                description = line.strip()
                break

        version = __import__('markrag').__version__
        frontmatter = f"""---
type: Project
title: {title}
description: {description}
status: draft
generated: {{by: markrag/{version}, at: {timestamp}}}
sources:
  - {{id: readme, resource: README.md, title: Project README}}
---

{readme_content}
"""
        seed_docs["readme"] = frontmatter

    # 2. Repository structure
    version = __import__('markrag').__version__
    structure_content = f"""---
type: Architecture
title: Repository Structure
description: Overview of the repository directory structure
status: draft
generated: {{by: markrag/{version}, at: {timestamp}}}
sources:
  - {{id: filesystem, resource: {repo_path.name}/, title: Repository filesystem}}
---

# Repository Structure

This document describes the top-level structure of the repository.

## Directories
"""

    dirs_to_doc = ["src", "tests", "docs", "config", "scripts"]
    for dir_name in dirs_to_doc:
        dir_path = repo_path / dir_name
        if dir_path.exists() and dir_path.is_dir():
            # Count files
            file_count = len(list(dir_path.rglob("*.py"))) if dir_name == "src" else len(list(dir_path.rglob("*")))
            structure_content += f"- `{dir_name}/`: {file_count} files\n"

    structure_content += "\n## Configuration Files\n"

    # 3. Config files
    config_files = ["pyproject.toml", "setup.py", "requirements.txt", "package.json", "*.yml", "*.yaml"]
    for pattern in config_files:
        for config_path in repo_path.glob(pattern):
            structure_content += f"- `{config_path.name}`: {config_path.name}\n"

    seed_docs["structure"] = structure_content

    # 3. Main source files (if src/ exists)
    src_path = repo_path / "src" / repo_path.name
    if src_path.exists():
        main_files = list(src_path.glob("*.py"))[:5]
        version = __import__('markrag').__version__
        for py_file in main_files:
            module_name = py_file.stem
            content = f"""---
type: Module
title: {module_name.capitalize()}
description: Auto-generated documentation for {module_name}.py
status: draft
generated: {{by: markrag/{version}, at: {timestamp}}}
sources:
  - {{id: {module_name}, resource: src/{repo_path.name}/{module_name}.py, title: Source file}}
---

# {module_name.capitalize()} Module

Auto-generated from source code. Review and enhance this documentation.

## Overview

TODO: Add module overview

## Functions/Classes

TODO: Document key functions and classes
"""
            seed_docs[f"modules/{module_name}"] = content

    return seed_docs


def _extract_title(content: str) -> str:
    """Extract title from concept content."""
    lines = content.split("\n")
    for line in lines:
        if line.startswith("title: "):
            return line[7:]
    return "Untitled"


def _update_index_for_seed(wiki_path: Path, seed_docs: dict[str, str]) -> None:
    """Update index.md with seed documents."""
    index_path = wiki_path / "index.md"
    if not index_path.exists():
        with open(index_path, "w") as f:
            f.write("""---
okf_version: "0.2"
---

# Bundle Index

""")

    with open(index_path, "r") as f:
        content = f.read()

    # Add seed docs to index
    for concept_id in seed_docs:
        title = _extract_title(seed_docs[concept_id])
        content += f"- [{title}](/{concept_id}.md)\n"

    with open(index_path, "w") as f:
        f.write(content)


def main() -> None:
    """Entry point for the CLI."""
    cli(obj={})


if __name__ == "__main__":
    main()

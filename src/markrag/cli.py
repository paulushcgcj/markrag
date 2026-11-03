"""Command-line interface for markrag.

Commands:

- ``serve``  - run the MCP server over stdio
- ``init``   - bootstrap the wiki for a repository
- ``search`` - search the wiki
- ``read``   - read a concept or one of its sections
- ``index``  - show the bundle map
- ``status`` - show bundle statistics

Everything is written to stdout except the startup banner, which goes to
stderr so that ``markrag serve`` never contaminates the MCP JSON-RPC stream.
"""

import json
import os
from pathlib import Path
from typing import Any

import click

from markrag import __version__
from markrag.okf import wiki_path_for_repo


def _repo_path(ctx: click.Context, repo: str | None) -> Path:
    """Resolve the repository path from the option, the environment, or cwd."""
    raw: str = repo or os.getenv("MARKRAG_REPO") or "."
    return Path(raw).expanduser().resolve()


def _wiki_path(repo_path: Path) -> Path:
    """Return the sibling wiki path for a repository."""
    return wiki_path_for_repo(repo_path)


def _format_output(data: dict[str, Any], format_type: str) -> str:
    """Format tool output as text or JSON."""
    if format_type == "json":
        return json.dumps(data, indent=2, default=str)
    return _format_text(data)


def _format_text(data: dict[str, Any]) -> str:
    """Format tool output as human-readable text."""
    if "error" in data and "results" not in data and "concepts" not in data:
        return f"Error: {data['error']}"

    if "concepts" in data:
        lines = [f"Bundle: {data.get('path', '.')}", f"Total concepts: {data['count']}", ""]
        for concept in data["concepts"]:
            stale = " [STALE]" if concept.get("stale") else ""
            description = concept.get("description", "")
            lines.append(
                f"  {concept['concept_id']}: {concept.get('title', 'Untitled')} "
                f"({concept.get('type', '')}) - {description[:60]}{stale}"
            )
            lines.append(f"     {concept.get('trust_line', '')}")
        return "\n".join(lines)

    if "results" in data:
        lines = [f"Query: {data.get('query', '')}", f"Results: {len(data['results'])}", ""]
        for position, result in enumerate(data["results"], 1):
            stale = " [STALE]" if result.get("stale") else ""
            lines.append(
                f"  {position}. {result['concept_id']}: {result.get('title', 'Untitled')}{stale}"
            )
            lines.append(f"     {result.get('trust_line', '')}")
            if result.get("section"):
                lines.append(f"     Section: {result['section']}")
            if result.get("excerpt"):
                lines.append(f"     Excerpt: {result['excerpt']}")
            if result.get("related"):
                related = ", ".join(item["concept_id"] for item in result["related"][:3])
                lines.append(f"     Related: {related}")
            lines.append("")
        retrieval = data.get("retrieval", {})
        if retrieval:
            lines.append(
                f"Retrieval: keyword={retrieval.get('keyword')} "
                f"vector={retrieval.get('vector')} "
                f"candidates={retrieval.get('candidates')}"
            )
        return "\n".join(lines)

    if "content" in data:
        lines = [f"Concept: {data.get('concept_id', 'Unknown')}", data.get("trust_line", "")]
        if data.get("stale_warning"):
            lines.append(data["stale_warning"])
        lines.append("")

        if data.get("sources"):
            lines.append("Sources:")
            for source in data["sources"]:
                lines.append(f"  - {source.get('id', '')}: {source.get('resource', '')}")
            lines.append("")

        lines.append("Content:")
        lines.append("-" * 40)
        lines.append(data["content"])
        return "\n".join(lines)

    if "total_docs" in data:
        lines = [
            "Bundle Statistics:",
            f"  Total documents: {data['total_docs']}",
            f"  Stale documents: {data.get('stale_count', 0)}",
            f"  Last rebuild: {data.get('last_rebuild', 'Never')}",
            "",
            "By Type:",
        ]
        for type_name, count in data.get("by_type", {}).items():
            lines.append(f"  {type_name}: {count}")
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
@click.version_option(version=__version__, prog_name="markrag")
@click.option("--repo", "-r", type=click.Path(), help="Path to the repository")
@click.option(
    "--format",
    "-f",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format",
)
@click.pass_context
def cli(ctx: click.Context, repo: str | None, format: str) -> None:
    """markrag - OKF-backed RAG MCP server for per-repo wikis."""
    ctx.ensure_object(dict)
    ctx.obj["repo"] = repo
    ctx.obj["format"] = format


@cli.command()
@click.option("--repo", "-r", type=click.Path(), help="Path to the repository")
@click.option(
    "--wiki-remote",
    "-w",
    envvar="MARKRAG_WIKI_REMOTE",
    help="Git remote URL to clone the wiki from",
)
@click.pass_context
def init(ctx: click.Context, repo: str | None, wiki_remote: str | None) -> None:
    """Bootstrap the wiki for a repository.

    Creates the sibling wiki directory, initialises it as a git repository,
    and writes draft seed concepts derived from the repository itself. Safe to
    re-run: existing concepts and log entries are preserved.

    A wiki that already has content is never seeded, but its index.md and
    log.md files are refreshed: those are OKF-curated and markrag owns them.
    The user-facing home.md is left untouched.
    """
    from markrag.bootstrap import bootstrap_wiki

    repo_path = _repo_path(ctx, repo)
    wiki_path = _wiki_path(repo_path)

    click.echo(f"Initializing wiki at: {wiki_path}")
    result = bootstrap_wiki(repo_path, wiki_path, wiki_remote, version=__version__)

    if result.cloned:
        click.echo(f"  Cloned from {wiki_remote}")
    for concept_id in result.created:
        click.echo(f"  Created: {concept_id}.md")
    if result.committed:
        click.echo("  Committed to git")
    for note in result.notes:
        click.echo(f"\n{note}")
    click.echo(f"\nWiki ready at: {result.wiki_path}")
    click.echo("Start the server with: markrag serve")


@cli.command()
@click.option("--repo", "-r", type=click.Path(), help="Path to the repository")
@click.pass_context
def serve(ctx: click.Context, repo: str | None) -> None:
    """Run the MCP server over stdio.

    The repository path comes from --repo or from the MARKRAG_REPO
    environment variable. When the wiki does not exist yet it is bootstrapped
    automatically.
    """
    repo_path = _repo_path(ctx, repo)
    os.environ["MARKRAG_REPO"] = str(repo_path)

    # The banner goes to stderr: stdout carries MCP JSON-RPC frames only.
    click.echo(f"Starting markrag MCP server for: {repo_path}", err=True)
    click.echo(f"Wiki path: {_wiki_path(repo_path)}", err=True)

    from markrag.server import main as server_main

    server_main()


@cli.command()
@click.argument("query")
@click.option("--repo", "-r", type=click.Path(), help="Path to the repository")
@click.option("--type", "-t", type=str, default=None, help="Filter by concept type")
@click.option("--tags", type=str, default=None, help="Filter by tags (comma-separated)")
@click.option("--status", type=str, default="any", help="Filter by status")
@click.option("--top-k", "-n", type=int, default=8, help="Number of results")
@click.option(
    "--format",
    "-f",
    type=click.Choice(["text", "json"]),
    default=None,
    help="Output format",
)
@click.pass_context
def search(
    ctx: click.Context,
    query: str,
    repo: str | None,
    type: str | None,
    tags: str | None,
    status: str,
    top_k: int,
    format: str | None,
) -> None:
    """Search the wiki with hybrid keyword and vector retrieval."""
    repo_path = _repo_path(ctx, repo)
    os.environ["MARKRAG_REPO"] = str(repo_path)

    out_format = format or ctx.obj.get("format", "text")
    tags_list = [tag.strip() for tag in tags.split(",")] if tags else None

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
@click.option(
    "--format",
    "-f",
    type=click.Choice(["text", "json"]),
    default=None,
    help="Output format",
)
@click.pass_context
def read(
    ctx: click.Context,
    concept_id: str,
    section: str | None,
    repo: str | None,
    format: str | None,
) -> None:
    """Read a concept from the wiki, or a single section of it."""
    repo_path = _repo_path(ctx, repo)
    os.environ["MARKRAG_REPO"] = str(repo_path)

    out_format = format or ctx.obj.get("format", "text")

    from markrag.server import okf_read

    result = okf_read(concept_id=concept_id, section=section)
    click.echo(_format_output(result, out_format))


@cli.command()
@click.option("--repo", "-r", type=click.Path(), help="Path to the repository")
@click.option(
    "--format",
    "-f",
    type=click.Choice(["text", "json"]),
    default=None,
    help="Output format",
)
@click.option("--path", "-p", type=str, default=None, help="Subdirectory path")
@click.pass_context
def index(ctx: click.Context, repo: str | None, format: str | None, path: str | None) -> None:
    """Show the bundle map (tier 0)."""
    repo_path = _repo_path(ctx, repo)
    os.environ["MARKRAG_REPO"] = str(repo_path)

    out_format = format or ctx.obj.get("format", "text")

    from markrag.server import okf_index

    result = okf_index(path=path)
    click.echo(_format_output(result, out_format))


@cli.command()
@click.option("--repo", "-r", type=click.Path(), help="Path to the repository")
@click.option(
    "--format",
    "-f",
    type=click.Choice(["text", "json"]),
    default=None,
    help="Output format",
)
@click.pass_context
def status(ctx: click.Context, repo: str | None, format: str | None) -> None:
    """Show bundle statistics."""
    repo_path = _repo_path(ctx, repo)
    os.environ["MARKRAG_REPO"] = str(repo_path)

    out_format = format or ctx.obj.get("format", "text")

    from markrag.server import okf_status

    result = okf_status()
    click.echo(_format_output(result, out_format))


def main() -> None:
    """Entry point for the CLI."""
    cli(obj={})


if __name__ == "__main__":
    main()

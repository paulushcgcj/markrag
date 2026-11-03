"""Wiki bootstrap.

Creates the sibling wiki repository and seeds it with draft concepts derived
from the repository it documents. Every seeded document:

- carries ``generated.by`` and a real ISO 8601 UTC timestamp,
- starts at ``status: draft``,
- cites only files that actually exist in the repository, and
- contains real, extracted content - never placeholders.

The bootstrap is non-destructive: re-running it never overwrites a concept a
human has already written, and it appends to ``log.md`` instead of replacing it.
"""

import ast
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import yaml

from markrag.bundle_map import write_bundle_map
from markrag.okf import (
    Bundle,
    Frontmatter,
    Generated,
    Source,
    append_log_entry,
    wiki_path_for_repo,
)

# Directories that are never interesting when scanning a repository.
_IGNORED_DIRS = frozenset(
    {
        ".git",
        ".github",
        ".idea",
        ".mvn",
        ".next",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "build",
        "coverage",
        "dist",
        "node_modules",
        "target",
        "venv",
    }
)

# Directory names that usually mark a distinct component of a repository.
_COMPONENT_HINTS = (
    "src",
    "app",
    "apps",
    "backend",
    "common",
    "database",
    "frontend",
    "libs",
    "packages",
    "services",
    "shared",
    "tools",
)

# Manifest files inspected for dependency information.
_MANIFESTS = (
    "pyproject.toml",
    "package.json",
    "pom.xml",
    "build.gradle",
    "build.gradle.kts",
    "requirements.txt",
    "docker-compose.yml",
    "docker-compose.yaml",
    "compose.yml",
    "compose.yaml",
)

_MAX_COMPONENTS = 3
_MAX_PYTHON_MODULES = 3


@dataclass(frozen=True)
class SeedDoc:
    """A draft concept produced by the bootstrap."""

    concept_id: str
    title: str
    description: str
    type: str
    body: str
    tags: tuple[str, ...] = ()
    sources: tuple[Source, ...] = ()

    def to_markdown(self, version: str, timestamp: str) -> str:
        """Render the seed as a full markdown document with frontmatter."""
        frontmatter = Frontmatter(
            type=self.type,
            title=self.title,
            description=self.description,
            tags=list(self.tags),
            sources=list(self.sources),
            generated=Generated(by=f"markrag/{version}", at=timestamp),
            status="draft",
        )
        return f"---\n{frontmatter.to_yaml()}\n---\n\n{self.body.rstrip()}\n"


@dataclass
class BootstrapResult:
    """Outcome of a bootstrap run."""

    wiki_path: Path
    created: list[str] = field(default_factory=list)
    committed: bool = False
    cloned: bool = False
    notes: list[str] = field(default_factory=list)


def _utc_now() -> str:
    """Current UTC time as an ISO 8601 string with explicit UTC offset."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _count_files(directory: Path) -> int:
    """Count files under a directory, skipping ignored directories."""
    total = 0
    for path in directory.rglob("*"):
        if any(part in _IGNORED_DIRS for part in path.relative_to(directory).parts):
            continue
        if path.is_file():
            total += 1
    return total


def _relative_listing(directory: Path, limit: int = 20) -> list[str]:
    """List files under a directory as repo-relative paths."""
    listing: list[str] = []
    for path in sorted(directory.rglob("*")):
        if any(part in _IGNORED_DIRS for part in path.relative_to(directory).parts):
            continue
        if path.is_file():
            listing.append(str(path.relative_to(directory)))
        if len(listing) >= limit:
            break
    return listing


def _first_paragraph(text: str, limit: int = 240) -> str:
    """Return the first meaningful paragraph of a markdown document."""
    paragraph: list[str] = []
    in_code = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        stripped = line.strip()
        if not stripped:
            if paragraph:
                break
            continue
        if stripped.startswith(("#", ">", "|", "!", "[", "*", "-")):
            continue
        paragraph.append(stripped)
        if sum(len(part) for part in paragraph) > limit:
            break
    result = " ".join(paragraph)
    if len(result) > limit:
        result = result[: limit - 1].rsplit(" ", 1)[0] + "…"
    return result


def _readme_title(readme: str, fallback: str) -> str:
    """Extract the README title, falling back to the repository name."""
    for line in readme.splitlines():
        if line.startswith("# "):
            return line[2:].strip() or fallback
    return fallback


def _readme_sections(readme: str) -> list[tuple[str, str]]:
    """Extract ``(heading, first paragraph)`` pairs for each H2 section."""
    sections: list[tuple[str, str]] = []
    current: str | None = None
    buffer: list[str] = []

    for line in readme.splitlines():
        if line.startswith("## "):
            if current is not None:
                sections.append((current, _first_paragraph("\n".join(buffer))))
            current = line[3:].strip()
            buffer = []
        elif current is not None:
            buffer.append(line)

    if current is not None:
        sections.append((current, _first_paragraph("\n".join(buffer))))
    return sections


def _seed_readme(repo: Path) -> SeedDoc | None:
    """Seed a project overview derived from the repository README."""
    readme_path = repo / "README.md"
    if not readme_path.is_file():
        return None

    readme = readme_path.read_text(encoding="utf-8", errors="replace")
    title = _readme_title(readme, repo.name)
    description = _first_paragraph(readme) or f"Overview of the {repo.name} repository."

    body = [f"# {title}", "", description, ""]
    sections = _readme_sections(readme)
    if sections:
        body.append("## Sections")
        body.append("")
        for heading, paragraph in sections:
            body.append(f"### {heading}")
            body.append("")
            body.append(paragraph or "_See the repository README for this section._")
            body.append("")
    else:
        body.append("## Details")
        body.append("")
        body.append("_This repository has a README without section headings._")
        body.append("")

    return SeedDoc(
        concept_id="readme",
        title=title,
        description=description,
        type="Project",
        body="\n".join(body),
        tags=("overview",),
        sources=(Source(id="readme", resource="README.md", title="Repository README"),),
    )


def _seed_structure(repo: Path) -> SeedDoc:
    """Seed a document describing the repository's top-level layout."""
    directories = [
        entry
        for entry in sorted(repo.iterdir())
        if entry.is_dir() and entry.name not in _IGNORED_DIRS
    ]
    root_files = [
        entry.name
        for entry in sorted(repo.iterdir())
        if entry.is_file() and entry.name not in {".DS_Store"}
    ]

    body = [
        "# Repository Structure",
        "",
        f"The `{repo.name}` repository is organized as follows.",
        "",
        "## Top-level directories",
        "",
    ]
    if directories:
        for directory in directories:
            body.append(f"- `{directory.name}/` — {_count_files(directory)} files")
    else:
        body.append("_No top-level directories._")
    body.append("")

    body.append("## Root files")
    body.append("")
    if root_files:
        for name in root_files:
            body.append(f"- `{name}`")
    else:
        body.append("_No root files._")
    body.append("")

    return SeedDoc(
        concept_id="structure",
        title="Repository Structure",
        description=f"Top-level layout of the {repo.name} repository.",
        type="Architecture",
        body="\n".join(body),
        tags=("architecture", "structure"),
        sources=(Source(id="filesystem", resource=".", title="Repository root"),),
    )


def _detect_components(repo: Path) -> list[Path]:
    """Detect the repository's distinct component directories."""
    detected: list[Path] = []
    for name in _COMPONENT_HINTS:
        candidate = repo / name
        if candidate.is_dir() and candidate not in detected:
            detected.append(candidate)
    return detected[:_MAX_COMPONENTS]


def _seed_components(repo: Path) -> list[SeedDoc]:
    """Seed one document per detected component directory."""
    docs: list[SeedDoc] = []
    for directory in _detect_components(repo):
        listing = _relative_listing(directory)
        body = [
            f"# {directory.name}",
            "",
            f"The `{directory.name}/` directory contains {_count_files(directory)} files.",
            "",
            "## Contents",
            "",
        ]
        if listing:
            for item in listing:
                body.append(f"- `{item}`")
        else:
            body.append("_No files found._")
        body.append("")

        docs.append(
            SeedDoc(
                concept_id=f"components/{directory.name}",
                title=f"{directory.name} component",
                description=f"Files and layout of the {directory.name} component.",
                type="Architecture",
                body="\n".join(body),
                tags=("architecture", directory.name),
                sources=(
                    Source(
                        id=directory.name,
                        resource=f"{directory.name}/",
                        title=f"{directory.name} directory",
                    ),
                ),
            )
        )
    return docs


def _parse_pyproject(path: Path) -> dict[str, list[str]]:
    """Extract dependency groups from a ``pyproject.toml``."""
    try:
        import tomllib

        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}

    groups: dict[str, list[str]] = {}
    project = data.get("project", {})
    runtime = project.get("dependencies", [])
    if runtime:
        groups["runtime (project.dependencies)"] = [str(item) for item in runtime]
    optional = project.get("optional-dependencies", {})
    for name, items in optional.items():
        groups[f"optional: {name}"] = [str(item) for item in items]
    for name, items in data.get("dependency-groups", {}).items():
        groups[f"group: {name}"] = [str(item) for item in items]
    return groups


def _parse_package_json(path: Path) -> dict[str, list[str]]:
    """Extract dependency groups from a ``package.json``."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}

    groups: dict[str, list[str]] = {}
    for key, label in (
        ("dependencies", "runtime (dependencies)"),
        ("devDependencies", "development (devDependencies)"),
    ):
        items = data.get(key, {})
        if isinstance(items, dict) and items:
            groups[label] = [f"{name} {version}" for name, version in sorted(items.items())]
    return groups


def _parse_pom(path: Path) -> dict[str, list[str]]:
    """Extract dependency names from a Maven ``pom.xml``."""
    try:
        tree = ET.parse(path)  # noqa: S314 - trusted local build file
    except (OSError, ET.ParseError):
        return {}

    namespace = {"m": "http://maven.apache.org/POM/4.0.0"}
    artifacts: list[str] = []
    for dependency in tree.getroot().findall(".//m:dependencies/m:dependency", namespace):
        group = dependency.findtext("m:groupId", default="", namespaces=namespace)
        artifact = dependency.findtext("m:artifactId", default="", namespaces=namespace)
        if artifact:
            artifacts.append(f"{group}:{artifact}" if group else artifact)
    return {"maven (dependencies)": artifacts} if artifacts else {}


def _parse_compose(path: Path) -> dict[str, list[str]]:
    """Extract service names from a docker compose file."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}

    if not isinstance(data, dict):
        return {}
    services = data.get("services", {})
    if not isinstance(services, dict) or not services:
        return {}
    return {
        "compose services": [
            f"{name} (image: {svc.get('image', 'build')})"
            for name, svc in sorted(services.items())
            if isinstance(svc, dict)
        ]
    }


def _parse_requirements(path: Path) -> dict[str, list[str]]:
    """Extract requirement lines from a ``requirements.txt``."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}

    requirements = [
        line.strip() for line in lines if line.strip() and not line.strip().startswith(("#", "-"))
    ]
    return {"requirements": requirements} if requirements else {}


def _seed_dependencies(repo: Path) -> SeedDoc | None:
    """Seed a document describing the repository's declared dependencies."""
    groups: dict[str, list[str]] = {}
    cited: list[str] = []

    for name in _MANIFESTS:
        path = repo / name
        if not path.is_file():
            continue
        cited.append(name)
        if name == "pyproject.toml":
            groups.update(_parse_pyproject(path))
        elif name == "package.json":
            groups.update(_parse_package_json(path))
        elif name == "pom.xml":
            groups.update(_parse_pom(path))
        elif name == "requirements.txt":
            groups.update(_parse_requirements(path))
        elif name.startswith(("docker-compose", "compose.")):
            groups.update(_parse_compose(path))

    if not cited:
        return None

    body = ["# Dependencies", "", "Dependencies declared by the repository's build manifests.", ""]
    if groups:
        for label, items in groups.items():
            body.append(f"## {label}")
            body.append("")
            for item in items:
                body.append(f"- `{item}`")
            body.append("")
    else:
        body.append("## Declared")
        body.append("")
        body.append("_No dependency declarations were found in the manifests._")
        body.append("")

    return SeedDoc(
        concept_id="dependencies",
        title="Dependencies",
        description="Build-time and runtime dependencies declared by the repository.",
        type="Dependency",
        body="\n".join(body),
        tags=("dependencies",),
        sources=tuple(Source(id=Path(name).stem, resource=name, title=name) for name in cited),
    )


def _makefile_targets(path: Path) -> list[str]:
    """Extract target names from a Makefile."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []

    targets: list[str] = []
    for line in lines:
        match = re.match(r"^([A-Za-z0-9_.-]+)\s*:(?!=)", line)
        if match and not match.group(1).startswith("."):
            targets.append(match.group(1))
    return targets


def _seed_workflow(repo: Path) -> SeedDoc | None:
    """Seed a document describing how the repository is built and tested."""
    cited: list[str] = []
    body = [
        "# Development Workflow",
        "",
        "How this repository is built, tested, and run.",
        "",
    ]

    makefile = repo / "Makefile"
    if makefile.is_file():
        cited.append("Makefile")
        targets = _makefile_targets(makefile)
        body.append("## Make targets")
        body.append("")
        if targets:
            for target in targets:
                body.append(f"- `make {target}`")
        else:
            body.append("_No targets found._")
        body.append("")

    package_json = repo / "package.json"
    if package_json.is_file():
        cited.append("package.json")
        try:
            data = json.loads(package_json.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        scripts = data.get("scripts", {}) if isinstance(data, dict) else {}
        if scripts:
            body.append("## npm scripts")
            body.append("")
            for name, command in sorted(scripts.items()):
                body.append(f"- `npm run {name}` — {command}")
            body.append("")

    workflows_dir = repo / ".github" / "workflows"
    if workflows_dir.is_dir():
        cited.append(".github/workflows/")
        workflow_files = sorted(path.name for path in workflows_dir.glob("*.y*ml"))
        body.append("## CI workflows")
        body.append("")
        if workflow_files:
            for name in workflow_files:
                body.append(f"- `.github/workflows/{name}`")
        else:
            body.append("_No workflow files found._")
        body.append("")

    if not cited:
        return None

    return SeedDoc(
        concept_id="processes/development-workflow",
        title="Development Workflow",
        description="How to build, test, and run this repository.",
        type="Process",
        body="\n".join(body),
        tags=("process", "workflow"),
        sources=tuple(
            Source(id=Path(name.rstrip("/")).stem or name, resource=name, title=name)
            for name in cited
        ),
    )


def _detect_test_layout(repo: Path) -> list[tuple[str, int]]:
    """Detect test directories in the repository with their file counts."""
    candidates = (
        "tests",
        "test",
        "src/test",
        "backend/src/test",
        "frontend/src",
        "cypress",
        "spec",
    )
    found: list[tuple[str, int]] = []
    for name in candidates:
        path = repo / name
        if not path.is_dir():
            continue
        count = sum(
            1
            for file in path.rglob("*")
            if file.is_file()
            and file.suffix in {".py", ".ts", ".tsx", ".js", ".jsx", ".java"}
            and not any(part in _IGNORED_DIRS for part in file.relative_to(path).parts)
        )
        if count:
            found.append((name, count))
    return found


def _seed_testing(repo: Path) -> SeedDoc | None:
    """Seed a document describing the repository's test layout."""
    layout = _detect_test_layout(repo)
    if not layout:
        return None

    body = [
        "# Testing",
        "",
        "Where the tests live and how they are organised.",
        "",
        "## Test locations",
        "",
    ]
    for name, count in layout:
        body.append(f"- `{name}/` — {count} test source files")
    body.append("")

    frameworks: list[str] = []
    for marker, label in (
        ("pytest.ini", "pytest"),
        ("conftest.py", "pytest"),
        ("vitest.config.ts", "Vitest"),
        ("vite.config.ts", "Vite"),
        ("playwright.config.ts", "Playwright"),
        ("cypress.config.ts", "Cypress"),
        ("jest.config.js", "Jest"),
    ):
        if (repo / marker).exists():
            frameworks.append(label)
    if frameworks:
        body.append("## Frameworks")
        body.append("")
        for framework in sorted(set(frameworks)):
            body.append(f"- {framework}")
        body.append("")

    return SeedDoc(
        concept_id="processes/testing",
        title="Testing",
        description="Test layout and tooling for this repository.",
        type="Process",
        body="\n".join(body),
        tags=("process", "testing"),
        sources=(Source(id="filesystem", resource=".", title="Repository root"),),
    )


def _python_module_summary(path: Path) -> tuple[str, list[str]]:
    """Extract a module docstring and its public API surface."""
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (OSError, SyntaxError, ValueError):
        return "", []

    docstring = ast.get_docstring(tree) or ""
    summary = _first_paragraph(docstring) if docstring else ""

    api: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith(
            "_"
        ):
            api.append(f"`{node.name}()`")
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            methods = [
                child.name
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                and not child.name.startswith("_")
            ]
            suffix = f" — methods: {', '.join(f'`{m}()`' for m in methods)}" if methods else ""
            api.append(f"`{node.name}`{suffix}")
    return summary, api


def _seed_python_modules(repo: Path) -> list[SeedDoc]:
    """Seed one document per top-level module of a Python package."""
    package_dir = repo / "src"
    if not package_dir.is_dir():
        return []

    candidates: list[Path] = []
    for package in sorted(package_dir.iterdir()):
        if not package.is_dir() or package.name.startswith("."):
            continue
        for module in sorted(package.glob("*.py")):
            if module.name.startswith("_"):
                continue
            candidates.append(module)
        if candidates:
            break

    docs: list[SeedDoc] = []
    for module in candidates[:_MAX_PYTHON_MODULES]:
        summary, api = _python_module_summary(module)
        relative = str(module.relative_to(repo))

        body = [
            f"# {module.stem}",
            "",
            summary or f"The `{relative}` module.",
            "",
        ]
        if api:
            body.append("## Public API")
            body.append("")
            for item in api:
                body.append(f"- {item}")
            body.append("")

        docs.append(
            SeedDoc(
                concept_id=f"modules/{module.stem}",
                title=f"{module.stem} module",
                description=summary or f"Public API of the {module.stem} module.",
                type="Module",
                body="\n".join(body),
                tags=("module", module.stem),
                sources=(
                    Source(id=module.stem, resource=relative, title=f"{relative} source file"),
                ),
            )
        )
    return docs


def build_seed_docs(repo: Path) -> list[SeedDoc]:
    """Build the set of draft seed concepts for a repository.

    Only documents backed by files that actually exist are produced.

    Args:
        repo: Path to the repository being documented.

    Returns:
        The seed concepts, ordered by concept ID.
    """
    docs: list[SeedDoc] = []
    producers: list[Callable[[], SeedDoc | None]] = [
        lambda: _seed_readme(repo),
        lambda: _seed_structure(repo),
    ]
    for producer in producers:
        doc = producer()
        if doc is not None:
            docs.append(doc)

    docs.extend(_seed_components(repo))

    producers = [
        lambda: _seed_dependencies(repo),
        lambda: _seed_workflow(repo),
        lambda: _seed_testing(repo),
    ]
    for producer in producers:
        doc = producer()
        if doc is not None:
            docs.append(doc)

    docs.extend(_seed_python_modules(repo))
    return docs


def _run_git(args: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a git command in a directory.

    Args:
        args: Arguments to pass to git.
        cwd: Directory to run the command in.
        check: Raise when the command fails. Disable for probes whose exit
            status carries the answer.
    """
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=check,
        capture_output=True,
        text=True,
    )


def _ensure_git_repo(wiki: Path, wiki_remote: str | None) -> bool:
    """Initialise or clone the wiki git repository.

    Returns:
        ``True`` when the wiki was cloned from a remote, ``False`` otherwise.
    """
    if (wiki / ".git").exists():
        return False

    if wiki_remote:
        empty = not any(wiki.iterdir())
        if empty:
            subprocess.run(
                ["git", "clone", wiki_remote, str(wiki)],
                check=True,
                capture_output=True,
                text=True,
            )
            return True
        _run_git(["init"], wiki)
    else:
        _run_git(["init"], wiki)

    _run_git(["config", "user.name", "markrag"], wiki)
    _run_git(["config", "user.email", "markrag@localhost"], wiki)
    return False


def _append_log(wiki: Path, entry: str, *, date: str | None = None) -> None:
    """Append an entry to the wiki's root ``log.md``."""
    append_log_entry(wiki, entry, date=date)


def bootstrap_wiki(
    repo_path: str | Path,
    wiki_path: str | Path | None = None,
    wiki_remote: str | None = None,
    version: str = "0.1.0",
) -> BootstrapResult:
    """Bootstrap the wiki for a repository.

    A wiki that does not exist yet is created: it becomes its own git
    repository, gains draft seed concepts derived from the repository, and gets
    its ``index.md`` and ``log.md`` files generated.

    A wiki that already holds documents is never seeded - it was written by
    someone, and markrag does not add concepts to it. Its ``index.md`` and
    ``log.md`` files are still refreshed, because those are OKF-curated:
    markrag owns them wherever they live.

    Args:
        repo_path: Path to the repository being documented.
        wiki_path: Explicit wiki path. Defaults to the sibling ``.wiki`` folder.
        wiki_remote: Git remote to clone from. Defaults to ``MARKRAG_WIKI_REMOTE``.
        version: markrag version recorded in ``generated.by``.

    Returns:
        A :class:`BootstrapResult` describing what was created.
    """
    repo = Path(repo_path).expanduser().resolve()
    wiki = Path(wiki_path).expanduser().resolve() if wiki_path else wiki_path_for_repo(repo)

    result = BootstrapResult(wiki_path=wiki)

    wiki.mkdir(parents=True, exist_ok=True)

    result.cloned = _ensure_git_repo(wiki, wiki_remote)

    # A wiki with no documents at all is fresh, whatever else it holds.
    has_documents = any(wiki.rglob("*.md"))

    timestamp = _utc_now()

    if has_documents:
        result.notes.append(
            "This wiki already contains documents, so no seed documents were added. "
            "Its index.md and log.md files were refreshed from the existing content."
        )
        _append_log(wiki, f"- [{timestamp}] bootstrap: refreshed the bundle indexes by markrag")
    else:
        for doc in build_seed_docs(repo):
            target = wiki / f"{doc.concept_id}.md"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(doc.to_markdown(version, timestamp), encoding="utf-8")
            result.created.append(doc.concept_id)

        if result.created:
            names = ", ".join(result.created)
            _append_log(wiki, f"- [{timestamp}] created: {names} by markrag")
        else:
            _append_log(wiki, f"- [{timestamp}] bootstrap: wiki checked by markrag")

    bundle = Bundle(root_path=wiki)
    bundle.load()
    write_bundle_map(bundle)

    # Only a wiki markrag seeded is committed. A wiki that already had content
    # may hold work in progress that is none of markrag's business to commit.
    if result.created and (wiki / ".git").exists():
        _run_git(["add", "-A"], wiki)
        # `git diff --cached --quiet` exits non-zero when there is something to
        # commit, so this probe must not raise.
        staged = _run_git(["diff", "--cached", "--quiet"], wiki, check=False)
        result.committed = staged.returncode != 0
        if result.committed:
            _run_git(
                ["commit", "-m", "Initial commit: bootstrap wiki with seed documents"],
                wiki,
            )

    return result

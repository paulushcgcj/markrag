"""Server-managed bundle map generation.

Every ``index.md`` in a wiki - the one at the root and the one in each
directory that holds concepts - is a rendering of the *same* bundle map. They
are produced from a single traversal of the bundle, so they cannot drift apart.

``home.md`` is the one file markrag does not own. It is the user-facing GitHub
Wiki homepage and belongs to whoever curates the wiki; markrag serves it and
never writes it.

The map is derived entirely from concept frontmatter. Call
:func:`write_bundle_map` after any change to the bundle.
"""

import re
from dataclasses import dataclass

from markrag.okf import (
    INDEX_FILENAME,
    OKF_VERSION,
    Bundle,
    derive_trust_level,
    get_stale_days,
    is_stale,
)

_ROOT_GROUP = "Root"


@dataclass(frozen=True)
class MapEntry:
    """A single concept as it appears in the bundle map."""

    concept_id: str
    title: str
    description: str
    type: str
    status: str
    trust_level: str
    stale: bool
    stale_days: int | None

    @property
    def directory(self) -> str:
        """Directory the concept lives in, relative to the bundle root."""
        return self.concept_id.rsplit("/", 1)[0] if "/" in self.concept_id else ""

    @property
    def wiki_link(self) -> str:
        """Wiki-root-relative link target for the concept."""
        return f"/{self.concept_id}.md"

    @property
    def trust_line(self) -> str:
        """One-line trust prefix, shared with tool output."""
        if self.stale_days is not None:
            return f"[Trust: {self.trust_level} | Stale: {self.stale_days} days past stale_after]"
        return f"[Trust: {self.trust_level}]"


@dataclass(frozen=True)
class BundleMap:
    """The ordered contents of a bundle, grouped by directory."""

    entries: tuple[MapEntry, ...]

    @property
    def groups(self) -> tuple[tuple[str, tuple[MapEntry, ...]], ...]:
        """Entries grouped by directory, root first, then alphabetical."""
        grouped: dict[str, list[MapEntry]] = {}
        for entry in self.entries:
            grouped.setdefault(entry.directory, []).append(entry)

        ordered: list[tuple[str, tuple[MapEntry, ...]]] = []
        if "" in grouped:
            ordered.append((_ROOT_GROUP, tuple(grouped.pop(""))))
        for directory in sorted(grouped):
            ordered.append((directory, tuple(grouped[directory])))
        return tuple(ordered)

    @property
    def directories(self) -> tuple[str, ...]:
        """Every directory that needs an ``index.md``.

        A directory qualifies when it holds a concept of its own or when it is
        an ancestor of one, so the indexes form a tree that mirrors the wiki.
        """
        found: set[str] = set()
        for entry in self.entries:
            parts = entry.directory.split("/") if entry.directory else []
            for depth in range(1, len(parts) + 1):
                found.add("/".join(parts[:depth]))
        return tuple(sorted(found))

    def entries_in(self, directory: str) -> tuple[MapEntry, ...]:
        """Return the concepts that live directly in a directory."""
        return tuple(entry for entry in self.entries if entry.directory == directory)

    def subdirectories(self, directory: str) -> tuple[str, ...]:
        """Return the immediate child directories of a directory."""
        prefix = f"{directory}/" if directory else ""
        found = {
            candidate
            for candidate in self.directories
            if candidate != directory
            and candidate.startswith(prefix)
            and "/" not in candidate[len(prefix) :]
        }
        return tuple(sorted(found))

    def count_in(self, directory: str) -> int:
        """Return how many concepts live in a directory or below it."""
        if not directory:
            return len(self.entries)
        prefix = f"{directory}/"
        return sum(
            1
            for entry in self.entries
            if entry.directory == directory or entry.directory.startswith(prefix)
        )


def build_bundle_map(bundle: Bundle) -> BundleMap:
    """Build the bundle map from a loaded bundle.

    Server-managed files (``index.md``, ``log.md``, ``home.md``) are never
    concepts and are excluded from the map.

    Args:
        bundle: A loaded bundle.

    Returns:
        The bundle map, sorted by concept ID for stable output.
    """
    entries: list[MapEntry] = []
    for concept_id, concept in bundle.concepts.items():
        frontmatter = concept.frontmatter
        entries.append(
            MapEntry(
                concept_id=concept_id,
                title=concept.title or concept_id,
                description=frontmatter.description or "",
                type=frontmatter.type,
                status=frontmatter.status,
                trust_level=derive_trust_level(concept),
                stale=is_stale(concept),
                stale_days=get_stale_days(concept),
            )
        )

    entries.sort(key=lambda entry: entry.concept_id)
    return BundleMap(entries=tuple(entries))


def _render_entry(entry: MapEntry) -> str:
    """Render one concept as a bullet line."""
    return (
        f"- [{entry.title}]({entry.wiki_link}) | {entry.type} | {entry.status} | {entry.trust_line}"
    )


def _render_preamble(title: str) -> list[str]:
    """Render the heading block shared by every index file."""
    return [
        f"# {title}",
        "",
        "Generated by markrag from concept frontmatter. Do not hand-edit.",
        "",
    ]


def render_index(bundle_map: BundleMap) -> str:
    """Render the root ``index.md`` - the agent-facing bundle map.

    The root index is the one place an index file may carry frontmatter: OKF
    v0.2 allows a bundle-root ``index.md`` to declare the ``okf_version`` it
    targets, and forbids frontmatter in every other index file.

    Every group heading links to the index of that directory, so the root
    index doubles as the top of the wiki's navigation tree.

    Args:
        bundle_map: The map to render.

    Returns:
        Full markdown document for the root ``index.md``.
    """
    lines = [
        "---",
        f'okf_version: "{OKF_VERSION}"',
        "---",
        "",
        "# Bundle Index",
        "",
        "Generated by markrag from concept frontmatter. Do not hand-edit.",
        "",
    ]

    if not bundle_map.entries:
        lines.append("_This bundle contains no concepts yet._")
        lines.append("")
        return "\n".join(lines)

    for group, entries in bundle_map.groups:
        if group == _ROOT_GROUP:
            lines.append(f"## {group}")
        else:
            lines.append(f"## [{group}](/{group}/{INDEX_FILENAME})")
        lines.append("")
        for entry in entries:
            lines.append(_render_entry(entry))
        lines.append("")

    return "\n".join(lines)


def render_directory_index(bundle_map: BundleMap, directory: str) -> str:
    """Render the ``index.md`` of one directory.

    The index lists the concepts that live directly in the directory and links
    to the index of each of its subdirectories, so the indexes form a tree that
    mirrors the wiki.

    A directory index carries no frontmatter: OKF v0.2 permits it only in the
    bundle-root index.

    Args:
        bundle_map: The map to render.
        directory: The directory, relative to the bundle root.

    Returns:
        Full markdown document for the directory's ``index.md``.
    """
    lines = _render_preamble(directory)

    entries = bundle_map.entries_in(directory)
    subdirectories = bundle_map.subdirectories(directory)

    if entries:
        lines.append("## Concepts")
        lines.append("")
        for entry in entries:
            lines.append(_render_entry(entry))
        lines.append("")

    if subdirectories:
        lines.append("## Subdirectories")
        lines.append("")
        for subdirectory in subdirectories:
            count = bundle_map.count_in(subdirectory)
            noun = "concept" if count == 1 else "concepts"
            lines.append(f"- [{subdirectory}](/{subdirectory}/{INDEX_FILENAME}) — {count} {noun}")
        lines.append("")

    if not entries and not subdirectories:
        lines.append("_This directory contains no concepts yet._")
        lines.append("")

    return "\n".join(lines)


def write_bundle_map(bundle: Bundle) -> BundleMap:
    """Regenerate every ``index.md`` in the wiki from the bundle.

    The root index and each directory index are written from the same
    :class:`BundleMap`, which is what keeps them from drifting apart.

    Args:
        bundle: The bundle to render. It is reloaded before rendering.

    Returns:
        The bundle map that was written.
    """
    bundle.load()
    bundle_map = build_bundle_map(bundle)

    root = bundle.root_path
    root.mkdir(parents=True, exist_ok=True)

    (root / INDEX_FILENAME).write_text(render_index(bundle_map), encoding="utf-8")

    for directory in bundle_map.directories:
        index_path = root / directory / INDEX_FILENAME
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text(
            render_directory_index(bundle_map, directory),
            encoding="utf-8",
        )

    return bundle_map


def map_link_targets(text: str) -> set[str]:
    """Extract the concept IDs referenced by a rendered bundle map.

    Used by the parity test to prove an index describes the concepts in its
    scope.

    Args:
        text: Rendered markdown.

    Returns:
        Set of referenced concept IDs.
    """
    targets: set[str] = set()
    for match in re.finditer(r"\]\(([^)]+)\)", text):
        target = match.group(1).strip()
        if target.endswith(".md") and not target.endswith(f"/{INDEX_FILENAME}"):
            concept_id = target.removeprefix("/")[: -len(".md")]
            targets.add(concept_id)
    return targets

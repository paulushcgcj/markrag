"""OKF v0.2 format handling for markrag.

This module provides the core data structures and parsing logic for
Open Knowledge Format (OKF) v0.2 concepts.

Reference: https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md
"""

import hashlib
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self, cast

import frontmatter
import yaml

# Reserved filenames that are NOT concepts.
#
# ``index.md`` and ``log.md`` are OKF-curated: markrag generates every one of
# them, at the wiki root and in every directory that holds concepts.
#
# The rest are the wiki's user-facing shell. ``home.md`` is the GitHub Wiki
# homepage; ``_Sidebar.md``, ``_Footer.md`` and ``_Header.md`` are the pages
# GitHub renders around the wiki. They belong to whoever curates the wiki:
# markrag serves them and never writes them.
INDEX_FILENAME = "index.md"
LOG_FILENAME = "log.md"
HOME_FILENAME = "home.md"
GITHUB_SHELL_FILENAMES = frozenset({"_header.md", "_sidebar.md", "_footer.md"})
RESERVED_FILES = {INDEX_FILENAME, LOG_FILENAME, HOME_FILENAME} | GITHUB_SHELL_FILENAMES

# The reserved filenames that are the wiki's shell rather than OKF's own
# reserved files.
SHELL_FILENAMES = {HOME_FILENAME} | GITHUB_SHELL_FILENAMES

# OKF version we support
OKF_VERSION = "0.2"

# A concept with no ``type`` is treated as a generic concept. OKF v0.2 makes
# ``type`` required, but also tells consumers to tolerate unknown types
# gracefully, "typically by treating them as generic concepts". Naming that
# generic type keeps the bundle map, the tool output and the type filter
# agreeing with each other.
DEFAULT_CONCEPT_TYPE = "Concept"

# Link targets that are not internal concept references
_EXTERNAL_LINK_RE = re.compile(r"^(?:[a-zA-Z][a-zA-Z0-9+.-]*:|//|#)")

# First markdown heading of a document, used as a title fallback
_H1_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)

# Dated sections in a log file.
_LOG_DATE_RE = re.compile(r"^##\s+(\d{4}-\d{2}-\d{2})\s*$")


def is_reserved_filename(name: str) -> bool:
    """Return whether a filename is server-managed rather than a concept.

    The comparison is case-insensitive: a wiki that spells its homepage
    ``Home.md`` means the same thing as one that spells it ``home.md``, and on
    a case-insensitive filesystem they are the same file.

    Args:
        name: File name, with or without a directory.

    Returns:
        ``True`` when the file is reserved.
    """
    return name.rsplit("/", 1)[-1].lower() in RESERVED_FILES


def _insert_log_entry(content: str, entry: str, date: str) -> str:
    """Insert an entry into log content under a dated section.

    Existing content is never rewritten - the entry is inserted into the
    section for its date, and a new section is created when the date has none
    yet.

    Args:
        content: The current log file content.
        entry: The markdown bullet to insert.
        date: The ``YYYY-MM-DD`` heading the entry belongs to.

    Returns:
        The updated log file content.
    """
    lines = content.split("\n")
    dated = [
        (index, match.group(1))
        for index, line in enumerate(lines)
        if (match := _LOG_DATE_RE.match(line))
    ]

    # The date already has a section: the entry joins the top of it.
    for index, heading_date in dated:
        if heading_date != date:
            continue
        insert_at = index + 1
        while insert_at < len(lines) and not lines[insert_at].strip():
            insert_at += 1
        lines.insert(insert_at, entry)
        return "\n".join(lines)

    heading = [f"## {date}", "", entry]
    if dated:
        # Keep the newest section first, directly under the file title.
        title = next((i for i, line in enumerate(lines) if line.startswith("# ")), None)
        insert_at = title + 1 if title is not None else dated[0][0]
        lines[insert_at:insert_at] = ["", *heading]
    else:
        while lines and not lines[-1].strip():
            lines.pop()
        lines.extend(["", *heading, ""])
    return "\n".join(lines)


def append_log_entry(
    root: Path,
    entry: str,
    *,
    directory: str = "",
    date: str | None = None,
) -> Path:
    """Add an entry to a wiki's log file, creating the file when needed.

    Entries are grouped under a ``## YYYY-MM-DD`` heading, newest first, and
    the heading is reused when an entry lands on a date that already has one.

    Args:
        root: The wiki root.
        entry: The markdown bullet to add.
        directory: The directory within the wiki that owns the log. Defaults
            to the wiki root.
        date: The date heading to file the entry under. Defaults to today.

    Returns:
        The path of the log file that was written.
    """
    day = date or datetime.now(UTC).strftime("%Y-%m-%d")
    log_path = root / directory / LOG_FILENAME if directory else root / LOG_FILENAME
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if not log_path.exists():
        log_path.write_text(
            f"# Directory Update Log\n\n## {day}\n\n{entry}\n",
            encoding="utf-8",
        )
        return log_path

    content = log_path.read_text(encoding="utf-8")
    log_path.write_text(_insert_log_entry(content, entry, day), encoding="utf-8")
    return log_path


def wiki_path_for_repo(repo_path: str | Path) -> Path:
    """Derive the sibling wiki path for a repository path.

    Implements the sibling-folder convention: a repo at ``<org>/<name>`` has its
    wiki at ``<org>/<name>.wiki``. The extension is *appended* to the directory
    name so that repos whose names contain dots (``my.repo``) resolve to
    ``my.repo.wiki`` rather than replacing ``.repo``.

    Args:
        repo_path: Path to the repository.

    Returns:
        Absolute path to the sibling wiki directory.
    """
    repo = Path(repo_path).expanduser().resolve()
    return repo.parent / f"{repo.name}.wiki"


def path_fingerprint(root: Path) -> str:
    """Compute a fingerprint of the Markdown files under a directory.

    The fingerprint changes whenever a document is added, removed, or edited.
    It is used to detect that a cached bundle is out of date.

    Reserved files are excluded: they are navigation and history, not
    documents, and markrag regenerates them on every write.

    Args:
        root: Directory to fingerprint.

    Returns:
        A hex digest of the directory's Markdown contents.
    """
    digest = hashlib.sha256()
    if not root.exists():
        return digest.hexdigest()
    for path in sorted(root.rglob("*.md")):
        if is_reserved_filename(path.name):
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(str(stat.st_mtime_ns).encode("utf-8"))
        digest.update(str(stat.st_size).encode("utf-8"))
    return digest.hexdigest()


def resolve_link_target(concept_id: str, target: str) -> str | None:
    """Resolve a markdown link target to a concept ID within the bundle.

    Handles root-relative (``/technologies/db.md``) and sibling-relative
    (``../shared/util.md``) targets. External URLs, anchors, and mail links are
    ignored. Broken links are tolerated by returning ``None``.

    Args:
        concept_id: ID of the concept containing the link.
        target: Raw link target as written in the markdown.

    Returns:
        The resolved concept ID, or ``None`` when the target is not internal.
    """
    if not target:
        return None

    cleaned = target.split("#", 1)[0].split("?", 1)[0].strip()
    if not cleaned or _EXTERNAL_LINK_RE.match(cleaned):
        return None

    if cleaned.startswith("/"):
        candidate = cleaned.lstrip("/")
    else:
        base = concept_id.rsplit("/", 1)[0] if "/" in concept_id else ""
        candidate = f"{base}/{cleaned}" if base else cleaned

    parts: list[str] = []
    for part in candidate.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            if parts:
                parts.pop()
            continue
        parts.append(part)

    if not parts:
        return None

    resolved = "/".join(parts)
    if resolved.endswith(".md"):
        resolved = resolved[: -len(".md")]
    return resolved or None


@dataclass
class Source:
    """A source reference from OKF frontmatter."""

    id: str
    resource: str
    title: str | None = None
    author: str | None = None
    usage_count: int | None = None
    last_modified: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create a Source from a dictionary."""
        return cls(
            id=data.get("id", ""),
            resource=data.get("resource", ""),
            title=data.get("title"),
            author=data.get("author"),
            usage_count=data.get("usage_count"),
            last_modified=data.get("last_modified"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary, preserving all fields."""
        result: dict[str, Any] = {"id": self.id, "resource": self.resource}
        if self.title:
            result["title"] = self.title
        if self.author:
            result["author"] = self.author
        if self.usage_count is not None:
            result["usage_count"] = self.usage_count
        if self.last_modified:
            result["last_modified"] = self.last_modified
        return result


@dataclass
class Generated:
    """Generated metadata from OKF frontmatter."""

    by: str
    at: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create Generated from a dictionary."""
        by = data.get("by", "")
        at = data.get("at", "")
        # Convert datetime to string if needed
        if hasattr(at, "isoformat"):
            at = at.isoformat().replace("+00:00", "Z")
        return cls(by=str(by), at=str(at))

    def to_dict(self) -> dict[str, str]:
        """Convert to dictionary."""
        return {"by": self.by, "at": self.at}


@dataclass
class Verified:
    """A verification entry from OKF frontmatter."""

    by: str
    at: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Create Verified from a dictionary."""
        by = data.get("by", "")
        at = data.get("at", "")
        # Convert datetime to string if needed
        if hasattr(at, "isoformat"):
            at = at.isoformat().replace("+00:00", "Z")
        return cls(by=str(by), at=str(at))

    def to_dict(self) -> dict[str, str]:
        """Convert to dictionary."""
        return {"by": self.by, "at": self.at}


@dataclass
class Frontmatter:
    """Parsed OKF frontmatter for a concept."""

    # Required
    type: str

    # Recommended
    title: str | None = None
    description: str | None = None
    resource: str | None = None
    tags: list[str] = field(default_factory=list)

    # Provenance
    sources: list[Source] = field(default_factory=list)
    generated: Generated | None = None

    # Trust
    verified: list[Verified] = field(default_factory=list)

    # Lifecycle
    status: str = "stable"
    stale_after: str | None = None

    # OKF version marker (for index.md)
    okf_version: str | None = None

    # Unknown fields - preserve verbatim
    unknown_fields: dict[str, Any] = field(default_factory=dict)

    # Keys that were explicitly present in the source document. Used so that a
    # rewrite does not silently add or drop optional fields.
    present_fields: frozenset[str] = frozenset()

    @classmethod
    def from_post(cls, post: frontmatter.Post) -> Self:
        """Create Frontmatter from a frontmatter Post object."""
        metadata = cast("dict[str, Any]", post.metadata or {})

        # Extract known fields
        known_fields = {
            "type",
            "title",
            "description",
            "resource",
            "tags",
            "sources",
            "generated",
            "verified",
            "status",
            "stale_after",
            "okf_version",
        }

        # Collect unknown fields
        unknown: dict[str, Any] = {}
        for key, value in metadata.items():
            if key not in known_fields:
                unknown[key] = value

        # Parse sources
        sources_list = []
        if "sources" in metadata and metadata["sources"]:
            for src_data in metadata["sources"]:
                if isinstance(src_data, dict):
                    sources_list.append(Source.from_dict(src_data))

        # Parse generated
        generated_obj = None
        if "generated" in metadata and metadata["generated"]:
            gen_data = metadata["generated"]
            if isinstance(gen_data, dict):
                generated_obj = Generated.from_dict(gen_data)

        # Parse verified
        verified_list = []
        if "verified" in metadata and metadata["verified"]:
            for ver_data in metadata["verified"]:
                if isinstance(ver_data, dict):
                    verified_list.append(Verified.from_dict(ver_data))

        # Convert stale_after to string if it's a datetime
        stale_after = metadata.get("stale_after")
        if stale_after is not None and hasattr(stale_after, "isoformat"):
            stale_after = stale_after.isoformat().replace("+00:00", "Z")

        return cls(
            type=metadata.get("type") or DEFAULT_CONCEPT_TYPE,
            title=metadata.get("title"),
            description=metadata.get("description"),
            resource=metadata.get("resource"),
            tags=metadata.get("tags", []) or [],
            sources=sources_list,
            generated=generated_obj,
            verified=verified_list,
            status=metadata.get("status", "stable"),
            stale_after=stale_after,
            okf_version=metadata.get("okf_version"),
            unknown_fields=unknown,
            present_fields=frozenset(metadata.keys()),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for YAML serialization."""
        result: dict[str, Any] = {"type": self.type}

        if self.title:
            result["title"] = self.title
        if self.description:
            result["description"] = self.description
        if self.resource:
            result["resource"] = self.resource
        if self.tags:
            result["tags"] = self.tags
        if self.sources:
            result["sources"] = [s.to_dict() for s in self.sources]
        if self.generated:
            result["generated"] = self.generated.to_dict()
        if self.verified:
            result["verified"] = [v.to_dict() for v in self.verified]
        if self.status != "stable" or "status" in self.present_fields:
            result["status"] = self.status
        if self.stale_after:
            result["stale_after"] = self.stale_after
        if self.okf_version:
            result["okf_version"] = self.okf_version

        # Add unknown fields
        result.update(self.unknown_fields)

        return result

    def to_yaml(self) -> str:
        """Convert to YAML string."""
        return str(yaml.dump(self.to_dict(), sort_keys=False, allow_unicode=True)).rstrip()


@dataclass
class Concept:
    """An OKF concept - a single markdown file with frontmatter."""

    path: Path  # Absolute path to the file
    concept_id: str  # Relative path without .md extension
    frontmatter: Frontmatter
    body: str

    @property
    def is_reserved(self) -> bool:
        """Check if this is a reserved file (not a concept)."""
        return is_reserved_filename(self.path.name)

    @property
    def filename(self) -> str:
        """Get the filename."""
        return self.path.name

    @property
    def title(self) -> str:
        """The document's title.

        Most OKF documents carry an explicit ``title`` in their frontmatter.
        Documents written before that convention existed often do not, so the
        first level-one heading is used instead. The fallback is derived for
        display only and is never written back to the frontmatter.
        """
        if self.frontmatter.title:
            return self.frontmatter.title
        match = _H1_RE.search(self.body)
        return match.group(1).strip() if match else ""

    @classmethod
    def from_file(cls, file_path: Path, bundle_root: Path | None = None) -> Self:
        """Load a concept from a markdown file."""
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        return cls.from_content(file_path, content, bundle_root)

    @classmethod
    def from_content(cls, file_path: Path, content: str, bundle_root: Path | None = None) -> Self:
        """Parse a concept from file path and content.

        Args:
            file_path: Path to the markdown file
            content: File content
            bundle_root: Optional bundle root for relative path calculation
        """
        post = frontmatter.loads(content)
        fm = Frontmatter.from_post(post)

        # Derive concept_id: relative path from bundle root without .md
        if bundle_root:
            # Get relative path from bundle root
            rel_path = file_path.relative_to(bundle_root)
            concept_id = str(rel_path.with_suffix("")).replace("\\", "/")
        else:
            # Fallback: use stem if we don't have bundle root
            concept_id = file_path.stem

        return cls(
            path=file_path.resolve(),
            concept_id=concept_id,
            frontmatter=fm,
            body=post.content or "",
        )

    def to_markdown(self) -> str:
        """Render the concept as markdown with frontmatter."""
        fm_yaml = self.frontmatter.to_yaml()
        return f"---\n{fm_yaml}\n---\n\n{self.body}"

    def extract_link_targets(self) -> list[str]:
        """Extract raw markdown link targets from the body.

        Targets are returned verbatim; use :func:`resolve_link_target` to turn
        them into concept IDs relative to the bundle root.
        """
        pattern = r"\[([^\]]*)\]\(([^\)]+)\)"
        return [match.group(2).strip() for match in re.finditer(pattern, self.body)]

    def extract_links(self) -> list[str]:
        """Extract the concept IDs linked from this concept.

        Link targets are resolved relative to this concept, so both
        ``/technologies/db.md`` and ``../shared/util.md`` resolve correctly.
        External links and broken links are skipped rather than raising.
        """
        links: list[str] = []
        for target in self.extract_link_targets():
            link_id = resolve_link_target(self.concept_id, target)
            if link_id is not None and link_id not in links:
                links.append(link_id)
        return links


@dataclass
class Bundle:
    """An OKF bundle - a directory of interconnected concept files."""

    root_path: Path
    concepts: dict[str, Concept] = field(default_factory=dict)
    reserved_files: dict[str, Path] = field(default_factory=dict)
    _fingerprint: str = field(default="", repr=False)

    def load(self) -> None:
        """Load all markdown files from the bundle root."""
        if not self.root_path.exists():
            raise FileNotFoundError(f"Bundle root not found: {self.root_path}")

        self.concepts.clear()
        self.reserved_files.clear()

        for md_file in self.root_path.rglob("*.md"):
            rel_path = md_file.relative_to(self.root_path)
            filename = md_file.name

            if is_reserved_filename(filename):
                # Track reserved files but don't parse as concepts
                self.reserved_files[str(rel_path)] = md_file
                continue

            try:
                concept = Concept.from_file(md_file, self.root_path)
                self.concepts[concept.concept_id] = concept
            except Exception:
                # Tolerance principle: a malformed document must not make the
                # whole bundle unreadable. Skip it and keep serving the rest.
                continue

        self._fingerprint = path_fingerprint(self.root_path)

    def is_stale(self) -> bool:
        """Return whether the bundle's files changed since it was loaded.

        A human editing a document in the wiki must be visible to the next
        read, so the in-RAM copy is checked against the filesystem.
        """
        return self._fingerprint != path_fingerprint(self.root_path)

    def get_concept(self, concept_id: str) -> Concept | None:
        """Get a concept by ID."""
        return self.concepts.get(concept_id)

    def get_concepts_by_type(self, type_filter: str | None = None) -> list[Concept]:
        """Get concepts filtered by type."""
        if type_filter is None:
            return list(self.concepts.values())
        return [c for c in self.concepts.values() if c.frontmatter.type == type_filter]

    def get_concepts_by_tag(self, tag: str) -> list[Concept]:
        """Get concepts with a specific tag."""
        return [c for c in self.concepts.values() if tag in c.frontmatter.tags]

    def get_concepts_by_status(self, status: str) -> list[Concept]:
        """Get concepts with a specific status."""
        return [c for c in self.concepts.values() if c.frontmatter.status == status]

    def get_linked_concepts(self, concept_id: str) -> list[str]:
        """Get concept IDs linked from a given concept.

        Broken links are skipped rather than raising.
        """
        concept = self.get_concept(concept_id)
        if concept is None:
            return []
        return concept.extract_links()

    def get_1hop_links(self, concept_id: str) -> list[tuple[str, str]]:
        """Get 1-hop link expansion: linked concept IDs + descriptions."""
        result: list[tuple[str, str]] = []
        for linked_id in self.get_linked_concepts(concept_id):
            linked_concept = self.get_concept(linked_id)
            if linked_concept is not None:
                result.append((linked_id, linked_concept.frontmatter.description or ""))
        return result


# Trust level derivation
def derive_trust_level(concept: Concept) -> str:
    """Derive trust level from verified field.

    Returns one of: unverified, machine-confirmed, human-reviewed
    """
    verified_list = concept.frontmatter.verified

    if not verified_list:
        return "unverified"

    # Check for human verifiers
    for v in verified_list:
        if v.by.startswith("human:"):
            return "human-reviewed"

    # Check for machine verifiers (producer/version or process:<id>)
    for v in verified_list:
        if v.by.startswith("producer/") or v.by.startswith("process:"):
            return "machine-confirmed"

    # Unknown verifier format - still count as machine-confirmed
    # (conservative: if there's a verified field, it's at least machine-confirmed)
    return "machine-confirmed"


def is_stale(concept: Concept) -> bool:
    """Check if a concept is stale based on stale_after field."""
    stale_after = concept.frontmatter.stale_after
    if stale_after is None:
        return False

    try:
        # Parse ISO 8601 datetime with explicit UTC offset
        stale_dt = datetime.fromisoformat(stale_after.replace("Z", "+00:00"))
        # Normalize to UTC
        if stale_dt.tzinfo is None:
            stale_dt = stale_dt.replace(tzinfo=UTC)
        else:
            stale_dt = stale_dt.astimezone(UTC)

        now = datetime.now(UTC)
        return now > stale_dt
    except (ValueError, TypeError):
        # If we can't parse, assume not stale
        return False


def get_stale_days(concept: Concept) -> int | None:
    """Get number of days past stale_after, or None if not stale."""
    stale_after = concept.frontmatter.stale_after
    if stale_after is None:
        return None

    try:
        stale_dt = datetime.fromisoformat(stale_after.replace("Z", "+00:00"))
        if stale_dt.tzinfo is None:
            stale_dt = stale_dt.replace(tzinfo=UTC)
        else:
            stale_dt = stale_dt.astimezone(UTC)

        now = datetime.now(UTC)
        if now <= stale_dt:
            return None

        delta = now - stale_dt
        return delta.days
    except (ValueError, TypeError):
        return None


def trust_line(concept: Concept) -> str:
    """Render the one-line trust prefix for a concept.

    Used by every surface that exposes a document: tool output, the bundle
    map, and CLI output. Staleness is reported as a warning only and never
    filters the document out.
    """
    trust = derive_trust_level(concept)
    stale_days = get_stale_days(concept)
    if stale_days is not None:
        return f"[Trust: {trust} | Stale: {stale_days} days past stale_after]"
    return f"[Trust: {trust}]"

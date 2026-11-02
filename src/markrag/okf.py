"""OKF v0.2 format handling for markrag.

This module provides the core data structures and parsing logic for
Open Knowledge Format (OKF) v0.2 concepts.

Reference: https://github.com/GoogleCloudPlatform/knowledge-catalog/blob/main/okf/SPEC.md
"""

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Self

import frontmatter

# Reserved filenames that are NOT concepts
RESERVED_FILES = {"index.md", "log.md", "home.md"}

# OKF version we support
OKF_VERSION = "0.2"


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
        return cls(by=data.get("by", ""), at=data.get("at", ""))

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
        return cls(by=data.get("by", ""), at=data.get("at", ""))

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

    @classmethod
    def from_post(cls, post: frontmatter.Post) -> Self:
        """Create Frontmatter from a frontmatter Post object."""
        metadata = post.metadata or {}
        content = post.content or ""

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

        return cls(
            type=metadata.get("type", ""),
            title=metadata.get("title"),
            description=metadata.get("description"),
            resource=metadata.get("resource"),
            tags=metadata.get("tags", []) or [],
            sources=sources_list,
            generated=generated_obj,
            verified=verified_list,
            status=metadata.get("status", "stable"),
            stale_after=metadata.get("stale_after"),
            okf_version=metadata.get("okf_version"),
            unknown_fields=unknown,
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
        if self.status != "stable":
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
        import yaml

        return yaml.dump(self.to_dict(), sort_keys=False, allow_unicode=True).rstrip()


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
        return self.path.name in RESERVED_FILES

    @property
    def filename(self) -> str:
        """Get the filename."""
        return self.path.name

    @classmethod
    def from_file(cls, file_path: Path) -> Self:
        """Load a concept from a markdown file."""
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        return cls.from_content(file_path, content)

    @classmethod
    def from_content(cls, file_path: Path, content: str) -> Self:
        """Parse a concept from file path and content."""
        post = frontmatter.loads(content)
        fm = Frontmatter.from_post(post)

        # Derive concept_id: relative path without .md
        # The concept_id is relative to the bundle root
        concept_id = file_path.stem
        if file_path.parent != file_path:  # If not at root
            # Get relative path from parent directory
            rel_path = file_path.relative_to(file_path.parent)
            concept_id = str(rel_path.with_suffix("")).replace("\\", "/")

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

    def extract_links(self) -> list[str]:
        """Extract concept links from the body.
        
        Returns list of concept IDs linked from this concept.
        """
        # Match markdown links like [text](/path/to/concept.md)
        # or [text](path/to/concept.md)
        pattern = r'\[([^\]]+)\]\(([^\)]+)\)'
        links = []

        for match in re.finditer(pattern, self.body):
            link_target = match.group(2)
            # Convert .md links to concept IDs
            if link_target.endswith(".md"):
                # Remove .md extension
                concept_id = link_target[:-3]
                # Remove leading /
                if concept_id.startswith("/"):
                    concept_id = concept_id[1:]
                links.append(concept_id)
            elif link_target.endswith("/"):
                # Directory link - might be to index.md
                concept_id = link_target.rstrip("/")
                if concept_id:
                    links.append(concept_id)

        return links


@dataclass
class Bundle:
    """An OKF bundle - a directory of interconnected concept files."""

    root_path: Path
    concepts: dict[str, Concept] = field(default_factory=dict)
    reserved_files: dict[str, Path] = field(default_factory=dict)

    def load(self) -> None:
        """Load all markdown files from the bundle root."""
        if not self.root_path.exists():
            raise FileNotFoundError(f"Bundle root not found: {self.root_path}")

        self.concepts.clear()
        self.reserved_files.clear()

        for md_file in self.root_path.rglob("*.md"):
            rel_path = md_file.relative_to(self.root_path)
            filename = md_file.name

            if filename in RESERVED_FILES:
                # Track reserved files but don't parse as concepts
                self.reserved_files[str(rel_path)] = md_file
                continue

            try:
                concept = Concept.from_file(md_file)
                # Use relative path without .md as concept_id
                concept_id = str(rel_path.with_suffix("")).replace("\\", "/")
                concept.concept_id = concept_id
                self.concepts[concept_id] = concept
            except Exception as e:
                # Log but don't fail on individual file parse errors
                # (tolerance principle)
                continue

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
        return [
            c for c in self.concepts.values() if c.frontmatter.status == status
        ]

    def get_linked_concepts(self, concept_id: str) -> list[str]:
        """Get concept IDs linked from a given concept."""
        concept = self.get_concept(concept_id)
        if concept is None:
            return []
        return concept.extract_links()

    def get_1hop_links(self, concept_id: str) -> list[tuple[str, str]]:
        """Get 1-hop link expansion: linked concept IDs + descriptions."""
        linked_ids = self.get_linked_concepts(concept_id)
        result = []
        for linked_id in linked_ids:
            linked_concept = self.get_concept(linked_id)
            if linked_concept and not linked_concept.is_reserved:
                desc = linked_concept.frontmatter.description or ""
                result.append((linked_id, desc))
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
            stale_dt = stale_dt.replace(tzinfo=timezone.utc)
        else:
            stale_dt = stale_dt.astimezone(timezone.utc)

        now = datetime.now(timezone.utc)
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
            stale_dt = stale_dt.replace(tzinfo=timezone.utc)
        else:
            stale_dt = stale_dt.astimezone(timezone.utc)

        now = datetime.now(timezone.utc)
        if now <= stale_dt:
            return None

        delta = now - stale_dt
        return delta.days
    except (ValueError, TypeError):
        return None

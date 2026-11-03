"""Tests for the server-managed bundle map.

Every ``index.md`` in a wiki - the root one and the one in each directory
that holds concepts - is a rendering of one map, so they must always describe
the same set of concepts.
"""

import pytest

from markrag.bundle_map import (
    build_bundle_map,
    map_link_targets,
    render_directory_index,
    render_index,
    write_bundle_map,
)
from markrag.okf import HOME_FILENAME, INDEX_FILENAME, Bundle


def _write_concept(wiki, concept_id, body, frontmatter=""):
    """Write a concept file into a wiki directory."""
    path = wiki / f"{concept_id}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"---\n{frontmatter}\n---\n\n{body}", encoding="utf-8")
    return path


@pytest.fixture
def map_bundle(tmp_path):
    """Build a bundle with concepts in two directories."""
    wiki = tmp_path / "map.wiki"
    wiki.mkdir()

    _write_concept(
        wiki,
        "readme",
        "# Readme\n\nOverview.\n",
        "type: Project\ntitle: Readme\ndescription: Overview of the project.",
    )
    _write_concept(
        wiki,
        "tech/spring-boot",
        "# Spring Boot\n\n## Profiles\n\nProfile config.\n",
        "type: Technology\ntitle: Spring Boot\ndescription: Java framework.",
    )
    _write_concept(
        wiki,
        "tech/database",
        "# Database\n\n## SQL\n\nStructured Query Language.\n",
        "type: Technology\ntitle: Database\ndescription: Data storage.",
    )

    bundle = Bundle(root_path=wiki)
    bundle.load()
    return bundle


class TestBundleMap:
    """Test the shared bundle map."""

    def test_map_excludes_server_managed_files(self, map_bundle):
        """index.md, log.md and home.md are never concepts."""
        (map_bundle.root_path / INDEX_FILENAME).write_text(
            '---\nokf_version: "0.2"\n---\n', encoding="utf-8"
        )
        (map_bundle.root_path / HOME_FILENAME).write_text("# Home\n", encoding="utf-8")
        (map_bundle.root_path / "log.md").write_text("# Log\n", encoding="utf-8")
        map_bundle.load()

        bundle_map = build_bundle_map(map_bundle)
        concept_ids = {entry.concept_id for entry in bundle_map.entries}

        assert concept_ids == {"readme", "tech/spring-boot", "tech/database"}

    def test_the_root_index_references_every_concept(self, map_bundle):
        """The root index is the complete map."""
        write_bundle_map(map_bundle)

        index_text = (map_bundle.root_path / INDEX_FILENAME).read_text(encoding="utf-8")

        assert map_link_targets(index_text) == {
            "readme",
            "tech/spring-boot",
            "tech/database",
        }

    def test_every_directory_gets_an_index(self, map_bundle):
        """Each directory that holds a concept gets its own index.md."""
        write_bundle_map(map_bundle)

        root = map_bundle.root_path
        assert (root / INDEX_FILENAME).exists()
        assert (root / "tech" / INDEX_FILENAME).exists()

    def test_a_directory_index_lists_only_its_own_concepts(self, map_bundle):
        """A directory index is the map of that directory."""
        write_bundle_map(map_bundle)

        tech_index = (map_bundle.root_path / "tech" / INDEX_FILENAME).read_text(encoding="utf-8")

        assert map_link_targets(tech_index) == {"tech/spring-boot", "tech/database"}

    def test_a_directory_index_links_to_its_subdirectories(self, map_bundle):
        """Nested directories are reachable from their parent's index."""
        _write_concept(
            map_bundle.root_path,
            "tech/spring/data",
            "# Data\n\nBody.\n",
            "type: Technology\ntitle: Data\ndescription: Data access.",
        )
        write_bundle_map(map_bundle)

        tech_index = (map_bundle.root_path / "tech" / INDEX_FILENAME).read_text(encoding="utf-8")

        assert "(/tech/spring/index.md)" in tech_index
        assert "1 concept" in tech_index

    def test_the_root_index_links_to_the_directory_indexes(self, map_bundle):
        """Each group heading points at the index of its directory."""
        write_bundle_map(map_bundle)

        index_text = (map_bundle.root_path / INDEX_FILENAME).read_text(encoding="utf-8")

        assert "## [tech](/tech/index.md)" in index_text

    def test_index_carries_only_the_okf_version_exception(self, map_bundle):
        """index.md may carry nothing but the okf_version frontmatter."""
        write_bundle_map(map_bundle)
        index_text = (map_bundle.root_path / INDEX_FILENAME).read_text(encoding="utf-8")

        frontmatter = index_text.split("---")[1]
        assert frontmatter.strip() == 'okf_version: "0.2"'

    def test_a_directory_index_carries_no_frontmatter(self, map_bundle):
        """OKF allows frontmatter only in the bundle-root index."""
        write_bundle_map(map_bundle)
        index_text = (map_bundle.root_path / "tech" / INDEX_FILENAME).read_text(encoding="utf-8")

        assert not index_text.startswith("---")
        assert "okf_version" not in index_text

    def test_home_is_never_written(self, map_bundle):
        """home.md is the user-facing page and belongs to the user."""
        (map_bundle.root_path / HOME_FILENAME).write_text("# Hand written home\n", encoding="utf-8")

        write_bundle_map(map_bundle)

        assert (map_bundle.root_path / HOME_FILENAME).read_text(encoding="utf-8") == (
            "# Hand written home\n"
        )

    def test_the_github_shell_pages_are_not_concepts(self, map_bundle):
        """The sidebar, footer and header are chrome, not knowledge."""
        for name in ("_Sidebar.md", "_Footer.md", "_Header.md"):
            (map_bundle.root_path / name).write_text(f"# {name}\n", encoding="utf-8")
        map_bundle.load()

        write_bundle_map(map_bundle)

        index_text = (map_bundle.root_path / INDEX_FILENAME).read_text(encoding="utf-8")
        assert "_Sidebar" not in index_text
        assert "_Footer" not in index_text
        assert "_Header" not in index_text

    def test_a_concept_without_a_type_is_generic(self, map_bundle):
        """A concept with no type is treated as a generic concept."""
        _write_concept(
            map_bundle.root_path,
            "plain",
            "# Plain\n\nBody.\n",
            "title: Plain\ndescription: No type at all.",
        )
        map_bundle.load()

        write_bundle_map(map_bundle)

        index_text = (map_bundle.root_path / INDEX_FILENAME).read_text(encoding="utf-8")
        assert "Concept" in index_text

    def test_the_indexes_carry_trust_and_staleness(self, map_bundle):
        """A trust line appears on every entry in the indexes."""
        stale = map_bundle.root_path / "tech" / "database.md"
        stale.write_text(
            stale.read_text(encoding="utf-8").replace(
                "type: Technology",
                "type: Technology\nstale_after: 2020-01-01T00:00:00Z",
            ),
            encoding="utf-8",
        )
        map_bundle.load()
        write_bundle_map(map_bundle)

        index_text = (map_bundle.root_path / INDEX_FILENAME).read_text(encoding="utf-8")
        tech_text = (map_bundle.root_path / "tech" / INDEX_FILENAME).read_text(encoding="utf-8")

        assert "[Trust: unverified]" in index_text
        assert "[Trust: unverified]" in tech_text
        assert "days past stale_after" in index_text
        assert "days past stale_after" in tech_text

    def test_rendering_is_deterministic(self, map_bundle):
        """The same bundle always renders the same files."""
        first = render_index(build_bundle_map(map_bundle))
        second = render_index(build_bundle_map(map_bundle))
        assert first == second

    def test_directory_rendering_is_deterministic(self, map_bundle):
        """The same bundle always renders the same directory index."""
        first = render_directory_index(build_bundle_map(map_bundle), "tech")
        second = render_directory_index(build_bundle_map(map_bundle), "tech")
        assert first == second

    def test_entries_are_grouped_by_directory(self, map_bundle):
        """Directory grouping is stable and alphabetical."""
        bundle_map = build_bundle_map(map_bundle)
        groups = [name for name, _ in bundle_map.groups]

        assert groups[0] == "Root"
        assert groups[1:] == sorted(groups[1:])


class TestDirectoryTree:
    """Test the directory tree the indexes mirror."""

    def test_directories_include_every_ancestor(self, map_bundle):
        """A nested concept gives every ancestor directory an index."""
        _write_concept(
            map_bundle.root_path,
            "tech/spring/data",
            "# Data\n\nBody.\n",
            "type: Technology\ntitle: Data\ndescription: Data access.",
        )
        map_bundle.load()

        bundle_map = build_bundle_map(map_bundle)

        assert bundle_map.directories == ("tech", "tech/spring")

    def test_subdirectories_are_immediate_children_only(self, map_bundle):
        """A directory lists its children, not its grandchildren."""
        _write_concept(
            map_bundle.root_path,
            "tech/spring/data",
            "# Data\n\nBody.\n",
            "type: Technology\ntitle: Data\ndescription: Data access.",
        )
        map_bundle.load()

        bundle_map = build_bundle_map(map_bundle)

        assert bundle_map.subdirectories("") == ("tech",)
        assert bundle_map.subdirectories("tech") == ("tech/spring",)

    def test_count_includes_nested_concepts(self, map_bundle):
        """A directory count covers everything below it."""
        bundle_map = build_bundle_map(map_bundle)

        assert bundle_map.count_in("") == 3
        assert bundle_map.count_in("tech") == 2

    def test_entries_in_selects_direct_concepts_only(self, map_bundle):
        """A directory index holds the concepts that live in it."""
        bundle_map = build_bundle_map(map_bundle)

        assert [entry.concept_id for entry in bundle_map.entries_in("tech")] == [
            "tech/database",
            "tech/spring-boot",
        ]
        assert [entry.concept_id for entry in bundle_map.entries_in("")] == ["readme"]

    def test_an_empty_directory_says_so(self, map_bundle):
        """A directory with nothing in it renders a placeholder."""
        index = render_directory_index(build_bundle_map(map_bundle), "empty")

        assert "no concepts yet" in index

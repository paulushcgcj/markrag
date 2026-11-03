"""Tests for bundle loading, link resolution, and the server's state handling."""

import pytest

from markrag.okf import (
    Bundle,
    is_reserved_filename,
    path_fingerprint,
    resolve_link_target,
    wiki_path_for_repo,
)
from markrag.server import (
    _excerpt,
    _get_bundle,
    _invalidate_index,
    _stale_warning,
)


class TestPathFingerprint:
    """Test the bundle fingerprint."""

    def test_missing_directory_has_a_fingerprint(self, tmp_path):
        """A missing directory still produces a digest."""
        assert path_fingerprint(tmp_path / "absent")

    def test_fingerprint_changes_when_a_file_changes(self, tmp_path):
        """Editing a file changes the fingerprint."""
        path = tmp_path / "a.md"
        path.write_text("one", encoding="utf-8")
        before = path_fingerprint(tmp_path)

        path.write_text("two", encoding="utf-8")
        assert path_fingerprint(tmp_path) != before


class TestLinkResolution:
    """Test markdown link resolution."""

    @pytest.mark.parametrize(
        "target",
        ["", "   ", "#anchor", "https://example.com", "mailto:someone@example.com"],
    )
    def test_non_internal_targets_are_ignored(self, target):
        """External and empty links never resolve to a concept."""
        assert resolve_link_target("tech/db", target) is None

    def test_root_relative_link(self):
        """A root-relative link resolves against the bundle root."""
        assert (
            resolve_link_target("tech/db", "/technologies/postgres.md") == "technologies/postgres"
        )

    def test_sibling_relative_link(self):
        """A relative link resolves against the linking document."""
        assert resolve_link_target("tech/db", "../shared/util.md") == "shared/util"

    def test_link_without_an_extension(self):
        """A link may omit the .md extension."""
        assert resolve_link_target("tech/db", "/tech/cache") == "tech/cache"

    def test_anchor_and_query_are_stripped(self):
        """A fragment or query string does not change the target."""
        assert resolve_link_target("tech/db", "/tech/cache.md#section") == "tech/cache"
        assert resolve_link_target("tech/db", "/tech/cache.md?v=2") == "tech/cache"

    def test_current_directory_segments_are_dropped(self):
        """A './' segment does not change the target."""
        assert resolve_link_target("tech/db", "./cache.md") == "tech/cache"

    def test_a_link_that_only_goes_up_stays_inside_the_bundle(self):
        """A link pointing above the bundle root is clamped to the root."""
        resolved = resolve_link_target("tech/db", "../../escape.md")

        assert resolved == "escape"
        assert not resolved.startswith("..")


class TestBundleLoading:
    """Test how a bundle loads its documents."""

    def test_missing_root_raises(self, tmp_path):
        """A bundle without a directory is an error."""
        bundle = Bundle(root_path=tmp_path / "absent")
        with pytest.raises(FileNotFoundError):
            bundle.load()

    def test_a_malformed_document_is_skipped(self, tmp_path):
        """One unreadable document never breaks the whole bundle."""
        wiki = tmp_path / "broken.wiki"
        wiki.mkdir()
        (wiki / "good.md").write_text(
            "---\ntype: Test\ntitle: Good\n---\n\nBody.\n", encoding="utf-8"
        )
        (wiki / "bad.md").write_text("", encoding="utf-8")

        bundle = Bundle(root_path=wiki)
        bundle.load()

        assert "good" in bundle.concepts

    def test_staleness_is_detected(self, tmp_path):
        """Editing a document after loading marks the bundle stale."""
        wiki = tmp_path / "watch.wiki"
        wiki.mkdir()
        path = wiki / "doc.md"
        path.write_text("---\ntype: Test\ntitle: Doc\n---\n\nBody.\n", encoding="utf-8")

        bundle = Bundle(root_path=wiki)
        bundle.load()
        assert not bundle.is_stale()

        path.write_text("---\ntype: Test\ntitle: Doc\n---\n\nChanged body.\n", encoding="utf-8")
        assert bundle.is_stale()

    def test_filter_helpers(self, tmp_path):
        """The bundle's filter helpers work without a filter."""
        wiki = tmp_path / "filters.wiki"
        wiki.mkdir()
        (wiki / "a.md").write_text(
            "---\ntype: Technology\ntitle: A\ntags: [x]\nstatus: stable\n---\n\nBody.\n",
            encoding="utf-8",
        )

        bundle = Bundle(root_path=wiki)
        bundle.load()

        assert len(bundle.get_concepts_by_type()) == 1
        assert bundle.get_concepts_by_type("Technology")
        assert not bundle.get_concepts_by_type("Process")
        assert bundle.get_concepts_by_tag("x")
        assert bundle.get_concepts_by_status("stable")

    def test_links_for_an_unknown_concept(self, tmp_path):
        """Asking about an unknown document yields nothing."""
        bundle = Bundle(root_path=tmp_path / "empty.wiki")
        bundle.root_path.mkdir(parents=True)
        bundle.load()

        assert bundle.get_linked_concepts("nope") == []


class TestExcerpt:
    """Test the excerpt helper."""

    def test_empty_text(self):
        """Empty text yields an empty excerpt."""
        assert _excerpt("") == ""

    def test_short_text_is_untouched(self):
        """Text under the limit is returned verbatim."""
        assert _excerpt("one two three") == "one two three"

    def test_long_text_is_capped(self):
        """Long text is cut to the token budget."""
        excerpt = _excerpt(" ".join(["word"] * 100), max_tokens=10)

        assert excerpt.endswith("…")
        assert len(excerpt.split()) == 10


class TestStaleWarning:
    """Test the staleness warning."""

    def test_a_fresh_document_has_no_warning(self, tmp_path):
        """A document within its lifetime gets no warning."""
        wiki = tmp_path / "fresh.wiki"
        wiki.mkdir()
        (wiki / "doc.md").write_text(
            "---\ntype: Test\ntitle: Doc\nstale_after: 2999-01-01T00:00:00Z\n---\n\nBody.\n",
            encoding="utf-8",
        )

        bundle = Bundle(root_path=wiki)
        bundle.load()
        assert _stale_warning(bundle.concepts["doc"]) is None

    def test_a_stale_document_gets_a_warning(self, tmp_path):
        """A document past stale_after gets a warning with the day count."""
        wiki = tmp_path / "old.wiki"
        wiki.mkdir()
        (wiki / "doc.md").write_text(
            "---\ntype: Test\ntitle: Doc\nstale_after: 2020-01-01T00:00:00Z\n---\n\nBody.\n",
            encoding="utf-8",
        )

        bundle = Bundle(root_path=wiki)
        bundle.load()

        warning = _stale_warning(bundle.concepts["doc"])
        assert warning is not None
        assert "days past stale_after" in warning


class TestReservedFilenames:
    """Test how reserved filenames are recognised."""

    @pytest.mark.parametrize(
        "name",
        ["index.md", "Index.md", "INDEX.MD", "log.md", "Log.md", "home.md", "Home.md"],
    )
    def test_reserved_names_are_recognised(self, name):
        """The check ignores case."""
        assert is_reserved_filename(name)

    @pytest.mark.parametrize("name", ["doc.md", "indexer.md", "homepage.md", "logs.md"])
    def test_other_names_are_concepts(self, name):
        """A file that merely resembles a reserved one is a concept."""
        assert not is_reserved_filename(name)

    def test_a_capitalised_home_page_is_not_a_concept(self, tmp_path):
        """A wiki that spells its homepage Home.md keeps it out of the map."""
        wiki = tmp_path / "cap.wiki"
        wiki.mkdir()
        (wiki / "Home.md").write_text("# Home\n\nWelcome.\n", encoding="utf-8")
        (wiki / "doc.md").write_text(
            "---\ntype: Test\ntitle: Doc\n---\n\nBody.\n", encoding="utf-8"
        )

        bundle = Bundle(root_path=wiki)
        bundle.load()

        assert "Home" not in bundle.concepts
        assert "doc" in bundle.concepts


class TestTitleFallback:
    """Test the title fallback for documents without a title field."""

    def test_the_first_heading_is_used(self, tmp_path):
        """A document without a title takes its first level-one heading."""
        wiki = tmp_path / "titles.wiki"
        wiki.mkdir()
        (wiki / "doc.md").write_text("# Object Storage Contract\n\nBody text.\n", encoding="utf-8")

        bundle = Bundle(root_path=wiki)
        bundle.load()

        assert bundle.concepts["doc"].title == "Object Storage Contract"

    def test_an_explicit_title_wins(self, tmp_path):
        """The frontmatter title is preferred over the heading."""
        wiki = tmp_path / "titles.wiki"
        wiki.mkdir()
        (wiki / "doc.md").write_text(
            "---\ntype: Test\ntitle: Explicit\n---\n\n# Heading\n\nBody.\n", encoding="utf-8"
        )

        bundle = Bundle(root_path=wiki)
        bundle.load()

        assert bundle.concepts["doc"].title == "Explicit"

    def test_a_document_without_a_title_or_heading_has_no_title(self, tmp_path):
        """A document with neither has an empty title."""
        wiki = tmp_path / "titles.wiki"
        wiki.mkdir()
        (wiki / "doc.md").write_text("---\ntype: Test\n---\n\nBody only.\n", encoding="utf-8")

        bundle = Bundle(root_path=wiki)
        bundle.load()

        assert bundle.concepts["doc"].title == ""

    def test_the_fallback_never_reaches_the_frontmatter(self, tmp_path):
        """The fallback is for display; it is not written back."""
        wiki = tmp_path / "titles.wiki"
        wiki.mkdir()
        path = wiki / "doc.md"
        path.write_text("# Heading\n\nBody.\n", encoding="utf-8")

        bundle = Bundle(root_path=wiki)
        bundle.load()

        assert bundle.concepts["doc"].frontmatter.title is None
        assert "title:" not in path.read_text(encoding="utf-8")


class TestServerState:
    """Test how the server keeps its bundle and index in step."""

    def test_bundle_is_loaded_from_the_repository(self, tmp_path, monkeypatch):
        """The bundle is derived from the repository's sibling wiki."""
        repo = tmp_path / "demo"
        repo.mkdir()
        wiki = wiki_path_for_repo(repo)
        wiki.mkdir()
        (wiki / "doc.md").write_text(
            "---\ntype: Test\ntitle: Doc\n---\n\nBody.\n", encoding="utf-8"
        )

        monkeypatch.setenv("MARKRAG_REPO", str(repo))

        bundle = _get_bundle()
        assert "doc" in bundle.concepts

    def test_a_missing_wiki_is_bootstrapped(self, tmp_path, monkeypatch):
        """Asking for a bundle with allow_bootstrap creates the wiki."""
        repo = tmp_path / "demo"
        repo.mkdir()
        (repo / "README.md").write_text("# Demo\n\nText.\n", encoding="utf-8")

        monkeypatch.setenv("MARKRAG_REPO", str(repo))

        bundle = _get_bundle(allow_bootstrap=True)
        assert wiki_path_for_repo(repo).exists()
        assert bundle.concepts

    def test_a_document_edited_on_disk_is_picked_up(self, tmp_path, monkeypatch):
        """A human edit is visible to the next read."""
        repo = tmp_path / "demo"
        repo.mkdir()
        wiki = wiki_path_for_repo(repo)
        wiki.mkdir()
        path = wiki / "doc.md"
        path.write_text("---\ntype: Test\ntitle: Doc\n---\n\nOriginal.\n", encoding="utf-8")

        monkeypatch.setenv("MARKRAG_REPO", str(repo))

        bundle = _get_bundle()
        assert "Original." in bundle.concepts["doc"].body

        path.write_text(
            "---\ntype: Test\ntitle: Doc\n---\n\nEdited by a human.\n", encoding="utf-8"
        )

        bundle = _get_bundle()
        assert "Edited by a human." in bundle.concepts["doc"].body

    def test_invalidating_the_index_forces_a_rebuild(self, tmp_path, monkeypatch):
        """Invalidating the index does not break the next search."""
        from markrag.server import okf_search

        repo = tmp_path / "demo"
        repo.mkdir()
        wiki = wiki_path_for_repo(repo)
        wiki.mkdir()
        (wiki / "doc.md").write_text(
            "---\ntype: Test\ntitle: Doc\ndescription: Searchable content\n---\n\n"
            "## Section\n\nSearchable body.\n",
            encoding="utf-8",
        )

        monkeypatch.setenv("MARKRAG_REPO", str(repo))

        first = okf_search("searchable")
        assert first["results"]

        _invalidate_index()
        second = okf_search("searchable")
        assert second["results"]

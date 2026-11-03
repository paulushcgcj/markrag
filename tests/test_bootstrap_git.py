"""Tests for the wiki bootstrap's git handling and edge cases."""

import subprocess

import pytest

from markrag.bootstrap import _append_log, _ensure_git_repo, bootstrap_wiki
from markrag.bundle_map import (
    build_bundle_map,
    render_index,
    write_bundle_map,
)
from markrag.okf import Bundle, wiki_path_for_repo


def _git(args, cwd):
    """Run a git command and return its stdout."""
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
    ).stdout


@pytest.fixture
def source_repo(tmp_path):
    """Build a small repository."""
    repo = tmp_path / "demo"
    repo.mkdir()
    (repo / "README.md").write_text("# Demo\n\nA demo repository.\n", encoding="utf-8")
    return repo


class TestGitHandling:
    """Test the wiki's git repository handling."""

    def test_wiki_is_initialised_when_no_remote_is_given(self, tmp_path):
        """A wiki without a remote becomes its own git repository."""
        wiki = tmp_path / "demo.wiki"
        wiki.mkdir()

        cloned = _ensure_git_repo(wiki, None)

        assert cloned is False
        assert (wiki / ".git").exists()
        assert _git(["config", "user.name"], wiki).strip() == "markrag"

    def test_an_existing_repository_is_left_alone(self, tmp_path):
        """An already initialised wiki is not re-initialised."""
        wiki = tmp_path / "demo.wiki"
        wiki.mkdir()
        _git(["init"], wiki)

        cloned = _ensure_git_repo(wiki, None)

        assert cloned is False
        assert (wiki / ".git").exists()

    def test_the_wiki_is_cloned_from_a_remote(self, tmp_path, source_repo):
        """A configured remote is cloned into the wiki directory."""
        remote = tmp_path / "remote.git"
        _git(["init", "--bare", str(remote)], tmp_path)

        wiki = tmp_path / "demo.wiki"
        wiki.mkdir()

        cloned = _ensure_git_repo(wiki, str(remote))

        assert cloned is True
        assert (wiki / ".git").exists()

    def test_a_non_empty_wiki_is_not_cloned_over(self, tmp_path, source_repo):
        """Existing wiki content is never overwritten by a clone."""
        remote = tmp_path / "remote.git"
        _git(["init", "--bare", str(remote)], tmp_path)

        wiki = tmp_path / "demo.wiki"
        wiki.mkdir()
        (wiki / "readme.md").write_text("human content\n", encoding="utf-8")

        cloned = _ensure_git_repo(wiki, str(remote))

        assert cloned is False
        assert (wiki / "readme.md").read_text(encoding="utf-8") == "human content\n"

    def test_bootstrap_reports_a_clone(self, tmp_path, source_repo):
        """The bootstrap result records that the wiki was cloned."""
        remote = tmp_path / "remote.git"
        _git(["init", "--bare", str(remote)], tmp_path)

        wiki = wiki_path_for_repo(source_repo)
        result = bootstrap_wiki(source_repo, wiki, str(remote), version="0.1.0")

        assert result.cloned is True


class TestAppendLog:
    """Test the wiki change log."""

    def test_log_is_created_when_missing(self, tmp_path):
        """A missing log file is created with a heading and a dated section."""
        wiki = tmp_path / "demo.wiki"
        wiki.mkdir()

        _append_log(wiki, "- [2026-01-01T00:00:00Z] created: [A](/a.md) by markrag")

        content = (wiki / "log.md").read_text(encoding="utf-8")
        assert content.startswith("# Directory Update Log")
        assert "## " in content
        assert "created: [A](/a.md)" in content

    def test_log_without_a_trailing_newline_is_extended(self, tmp_path):
        """An entry is added even when the log lacks a final newline."""
        wiki = tmp_path / "demo.wiki"
        wiki.mkdir()
        (wiki / "log.md").write_text(
            "# Directory Update Log\n\n- [2026-01-01T00:00:00Z] first", encoding="utf-8"
        )

        _append_log(wiki, "- [2026-01-02T00:00:00Z] second")

        content = (wiki / "log.md").read_text(encoding="utf-8")
        assert content.endswith("\n")
        assert "first" in content
        assert "second" in content

    def test_entries_are_grouped_under_a_dated_section(self, tmp_path):
        """An entry joins the section for its date."""
        wiki = tmp_path / "demo.wiki"
        wiki.mkdir()
        (wiki / "log.md").write_text(
            "# Directory Update Log\n\n## 2026-01-01\n\n- first entry\n", encoding="utf-8"
        )

        _append_log(wiki, "- second entry", date="2026-01-01")

        content = (wiki / "log.md").read_text(encoding="utf-8")
        assert content.count("## 2026-01-01") == 1
        assert "- second entry" in content
        assert "- first entry" in content

    def test_a_new_date_gets_its_own_section_at_the_top(self, tmp_path):
        """The newest section stays first."""
        wiki = tmp_path / "demo.wiki"
        wiki.mkdir()
        (wiki / "log.md").write_text(
            "# Directory Update Log\n\n## 2026-01-01\n\n- first entry\n", encoding="utf-8"
        )

        _append_log(wiki, "- newer entry", date="2026-02-02")

        content = (wiki / "log.md").read_text(encoding="utf-8")
        assert content.index("## 2026-02-02") < content.index("## 2026-01-01")
        assert "- first entry" in content


class TestEmptyBundle:
    """Test rendering an empty bundle."""

    def test_empty_index_says_so(self, tmp_path):
        """An empty bundle renders a placeholder rather than an empty page."""
        wiki = tmp_path / "empty.wiki"
        wiki.mkdir()
        bundle = Bundle(root_path=wiki)
        bundle.load()

        bundle_map = build_bundle_map(bundle)
        index = render_index(bundle_map)

        assert "no concepts yet" in index


class TestNavigationOwnership:
    """Test which files markrag curates in a wiki it did not create."""

    def test_a_new_wiki_has_no_home_page(self, tmp_path, source_repo):
        """markrag writes navigation, never the user-facing homepage."""
        wiki = wiki_path_for_repo(source_repo)
        bootstrap_wiki(source_repo, wiki, version="0.1.0")

        assert (wiki / "index.md").exists()
        assert (wiki / "log.md").exists()
        assert not (wiki / "home.md").exists()

    def test_a_hand_made_wiki_is_not_seeded(self, tmp_path, source_repo):
        """A wiki that already has content gains no seed documents."""
        wiki = wiki_path_for_repo(source_repo)
        wiki.mkdir()
        (wiki / "index.md").write_text("# Hand written index\n", encoding="utf-8")
        (wiki / "Home.md").write_text("# Hand written home\n", encoding="utf-8")
        (wiki / "log.md").write_text("# Hand written log\n", encoding="utf-8")
        (wiki / "existing.md").write_text(
            "---\ntype: Test\ntitle: Existing\n---\n\nBody.\n", encoding="utf-8"
        )

        result = bootstrap_wiki(source_repo, wiki, version="0.1.0")

        assert result.created == []
        assert result.notes
        assert not (wiki / "readme.md").exists()

    def test_a_hand_made_wiki_keeps_its_home_page(self, tmp_path, source_repo):
        """The user-facing homepage survives a bootstrap untouched."""
        wiki = wiki_path_for_repo(source_repo)
        wiki.mkdir()
        (wiki / "Home.md").write_text("# Hand written home\n", encoding="utf-8")
        (wiki / "existing.md").write_text(
            "---\ntype: Test\ntitle: Existing\n---\n\nBody.\n", encoding="utf-8"
        )

        bootstrap_wiki(source_repo, wiki, version="0.1.0")

        assert (wiki / "Home.md").read_text(encoding="utf-8") == "# Hand written home\n"

    def test_a_hand_made_wiki_gets_a_generated_index(self, tmp_path, source_repo):
        """index.md is OKF-curated, so markrag regenerates it."""
        wiki = wiki_path_for_repo(source_repo)
        wiki.mkdir()
        (wiki / "index.md").write_text("# Hand written index\n", encoding="utf-8")
        (wiki / "existing.md").write_text(
            "---\ntype: Test\ntitle: Existing\n---\n\nBody.\n", encoding="utf-8"
        )

        bootstrap_wiki(source_repo, wiki, version="0.1.0")

        assert "okf_version" in (wiki / "index.md").read_text(encoding="utf-8")

    def test_a_hand_made_log_keeps_its_history(self, tmp_path, source_repo):
        """The log is appended to, never replaced."""
        wiki = wiki_path_for_repo(source_repo)
        wiki.mkdir()
        (wiki / "log.md").write_text(
            "# Directory Update Log\n\n## 2026-01-01\n\n- a human entry\n", encoding="utf-8"
        )
        (wiki / "existing.md").write_text(
            "---\ntype: Test\ntitle: Existing\n---\n\nBody.\n", encoding="utf-8"
        )

        bootstrap_wiki(source_repo, wiki, version="0.1.0")

        content = (wiki / "log.md").read_text(encoding="utf-8")
        assert "- a human entry" in content
        assert "bootstrap" in content

    def test_the_bundle_map_is_always_written(self, tmp_path):
        """A wiki markrag did not create still gets its indexes."""
        wiki = tmp_path / "handmade.wiki"
        wiki.mkdir()
        (wiki / "index.md").write_text("# Hand written\n", encoding="utf-8")
        (wiki / "doc.md").write_text(
            "---\ntype: Test\ntitle: Doc\n---\n\nBody.\n", encoding="utf-8"
        )

        bundle = Bundle(root_path=wiki)
        bundle.load()

        assert write_bundle_map(bundle) is not None
        assert "okf_version" in (wiki / "index.md").read_text(encoding="utf-8")
        assert not (wiki / "home.md").exists()

    def test_a_populated_wiki_is_never_committed(self, tmp_path, source_repo):
        """A bootstrap must not commit work in progress it did not write."""
        wiki = wiki_path_for_repo(source_repo)
        wiki.mkdir()
        (wiki / "existing.md").write_text(
            "---\ntype: Test\ntitle: Existing\n---\n\nBody.\n", encoding="utf-8"
        )

        result = bootstrap_wiki(source_repo, wiki, version="0.1.0")

        assert result.committed is False

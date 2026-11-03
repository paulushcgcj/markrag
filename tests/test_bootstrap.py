"""Tests for the wiki bootstrap."""

import subprocess
from pathlib import Path

import frontmatter
import pytest

from markrag.bootstrap import bootstrap_wiki, build_seed_docs
from markrag.okf import Bundle, wiki_path_for_repo


@pytest.fixture(autouse=True)
def no_global_git_config(tmp_path, monkeypatch):
    """Hide the developer's global git config from every git call.

    The bootstrap sets its own commit identity; the suite must not depend on
    the machine having one.
    """
    empty = tmp_path / "empty-gitconfig"
    empty.write_text("", encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(empty))
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", str(empty))


@pytest.fixture
def source_repo(tmp_path):
    """Build a repository with the layout markrag is expected to handle."""
    repo = tmp_path / "sample-repo"
    (repo / "src" / "sample_repo").mkdir(parents=True)

    (repo / "README.md").write_text(
        "# Sample: Project\n\nA project whose title contains a colon.\n\n"
        "## Install\n\nRun the installer.\n\n## Usage\n\nUse the tool.\n",
        encoding="utf-8",
    )
    (repo / "pyproject.toml").write_text(
        '[project]\nname = "sample-repo"\ndependencies = ["httpx>=1.0"]\n', encoding="utf-8"
    )
    (repo / "src" / "sample_repo" / "cli.py").write_text(
        'def main():\n    """Entry point."""\n    return 0\n', encoding="utf-8"
    )
    return repo


class TestSeedDocuments:
    """Test the draft seed concepts."""

    def test_seed_documents_are_derived_from_real_files(self, source_repo):
        """Only files that exist are cited as sources."""
        for doc in build_seed_docs(source_repo):
            for source in doc.sources:
                assert (source_repo / source.resource).exists(), source.resource

    def test_seed_documents_cover_the_repository(self, source_repo):
        """The seed set spans the overview, structure, and the sources."""
        concept_ids = {doc.concept_id for doc in build_seed_docs(source_repo)}

        assert "readme" in concept_ids
        assert "structure" in concept_ids
        assert "dependencies" in concept_ids
        assert "modules/cli" in concept_ids

    def test_seed_frontmatter_is_valid_yaml(self, source_repo):
        """A title containing a colon must not corrupt the frontmatter."""
        wiki = wiki_path_for_repo(source_repo)
        bootstrap_wiki(source_repo, wiki, version="0.1.0")

        readme = frontmatter.load(wiki / "readme.md")
        metadata: dict[str, object] = dict(readme.metadata)
        assert metadata["title"] == "Sample: Project"
        assert metadata["status"] == "draft"
        generated = metadata["generated"]
        assert isinstance(generated, dict)
        assert generated["by"] == "markrag/0.1.0"

    def test_every_seed_is_draft_stamped(self, source_repo):
        """Generated documents start as drafts with a real timestamp."""
        wiki = wiki_path_for_repo(source_repo)
        bootstrap_wiki(source_repo, wiki, version="0.1.0")

        bundle = Bundle(root_path=wiki)
        bundle.load()
        assert bundle.concepts

        for concept in bundle.concepts.values():
            assert concept.frontmatter.status == "draft"
            assert concept.frontmatter.generated is not None
            assert concept.frontmatter.generated.by.startswith("markrag/")
            assert concept.frontmatter.generated.at.endswith("Z")
            assert concept.frontmatter.verified == []

    def test_seed_bodies_contain_no_placeholders(self, source_repo):
        """Seeded documents carry real content, never TODO markers."""
        wiki = wiki_path_for_repo(source_repo)
        bootstrap_wiki(source_repo, wiki, version="0.1.0")

        for path in wiki.rglob("*.md"):
            text = path.read_text(encoding="utf-8")
            assert "TODO:" not in text
            assert "TODO " not in text


class TestBootstrap:
    """Test the bootstrap procedure."""

    def test_bootstrap_creates_nested_concepts(self, source_repo):
        """Seed concepts in subdirectories are written without errors."""
        wiki = wiki_path_for_repo(source_repo)
        result = bootstrap_wiki(source_repo, wiki, version="0.1.0")

        assert (wiki / "modules" / "cli.md").exists()
        assert "modules/cli" in result.created

    def test_bootstrap_creates_the_server_managed_files(self, source_repo):
        """index.md and log.md are created; home.md is left to the user."""
        wiki = wiki_path_for_repo(source_repo)
        bootstrap_wiki(source_repo, wiki, version="0.1.0")

        assert (wiki / "index.md").exists()
        assert (wiki / "log.md").exists()
        assert not (wiki / "home.md").exists()

    def test_bootstrap_creates_an_index_in_every_directory(self, source_repo):
        """Nested seed concepts get their own index.md."""
        wiki = wiki_path_for_repo(source_repo)
        bootstrap_wiki(source_repo, wiki, version="0.1.0")

        assert (wiki / "modules" / "index.md").exists()
        assert (wiki / "components" / "index.md").exists()

    def test_bootstrap_initialises_a_git_repository(self, source_repo):
        """The wiki is its own git repository, committed as markrag."""
        wiki = wiki_path_for_repo(source_repo)
        result = bootstrap_wiki(source_repo, wiki, version="0.1.0")

        assert (wiki / ".git").exists()
        assert result.committed

        log = subprocess.run(
            ["git", "log", "-1", "--format=%an <%ae>"],
            cwd=wiki,
            capture_output=True,
            text=True,
            check=True,
        )
        assert "markrag" in log.stdout

    def test_bootstrap_is_not_destructive(self, source_repo):
        """Re-running keeps human content and human log entries."""
        wiki = wiki_path_for_repo(source_repo)
        bootstrap_wiki(source_repo, wiki, version="0.1.0")

        # A human edits the wiki.
        (wiki / "readme.md").write_text(
            "---\ntype: Project\ntitle: Hand written\n---\n\nHuman content.\n",
            encoding="utf-8",
        )
        with (wiki / "log.md").open("a", encoding="utf-8") as handle:
            handle.write("- [manual] human-authored entry\n")

        result = bootstrap_wiki(source_repo, wiki, version="0.1.0")

        assert result.created == []
        assert "Human content." in (wiki / "readme.md").read_text(encoding="utf-8")
        assert "human-authored entry" in (wiki / "log.md").read_text(encoding="utf-8")

    def test_bootstrap_uses_the_sibling_convention(self, source_repo):
        """The wiki is created next to the repository."""
        wiki = wiki_path_for_repo(source_repo)
        assert wiki == source_repo.parent / f"{source_repo.name}.wiki"

    def test_bootstrap_keeps_every_index_in_step(self, source_repo):
        """Every index.md describes the concepts in its scope."""
        from markrag.bundle_map import INDEX_FILENAME, map_link_targets

        wiki = wiki_path_for_repo(source_repo)
        bootstrap_wiki(source_repo, wiki, version="0.1.0")

        index_targets = map_link_targets((wiki / INDEX_FILENAME).read_text(encoding="utf-8"))
        assert index_targets

        modules_targets = map_link_targets(
            (wiki / "modules" / INDEX_FILENAME).read_text(encoding="utf-8")
        )
        assert modules_targets
        assert modules_targets < index_targets


class TestWikiPath:
    """Test the sibling-folder convention."""

    def test_plain_name(self, tmp_path):
        """A repository without a dot gets a .wiki sibling."""
        repo = tmp_path / "project"
        repo.mkdir()
        assert wiki_path_for_repo(repo) == tmp_path / "project.wiki"

    def test_name_containing_a_dot(self, tmp_path):
        """A dotted repository name is not truncated."""
        repo = tmp_path / "my.project"
        repo.mkdir()
        assert wiki_path_for_repo(repo) == tmp_path / "my.project.wiki"

    def test_result_is_absolute(self, tmp_path):
        """The derived path is always absolute."""
        repo = tmp_path / "project"
        repo.mkdir()
        assert Path(wiki_path_for_repo(repo)).is_absolute()

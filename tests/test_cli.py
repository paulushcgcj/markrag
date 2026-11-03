"""Tests for the command-line interface."""

import json

import pytest
from click.testing import CliRunner

from markrag.cli import cli


@pytest.fixture
def repo(tmp_path):
    """Build a small repository to point the CLI at."""
    path = tmp_path / "repo"
    path.mkdir()
    (path / "README.md").write_text("# Demo Project\n\nA demo repository.\n", encoding="utf-8")
    (path / "src").mkdir()
    (path / "src" / "app.py").write_text("def main():\n    return 0\n", encoding="utf-8")
    return path


@pytest.fixture
def runner():
    """Return a click CLI runner."""
    return CliRunner()


class TestInitCommand:
    """Test the init command."""

    def test_init_creates_the_wiki(self, runner, repo):
        """Init bootstraps the sibling wiki directory."""
        result = runner.invoke(cli, ["init", "--repo", str(repo)])

        assert result.exit_code == 0, result.output
        assert "Wiki ready at" in result.output
        assert (repo.parent / "repo.wiki").exists()
        assert (repo.parent / "repo.wiki" / "index.md").exists()

    def test_init_is_idempotent(self, runner, repo):
        """Running init twice keeps the wiki intact."""
        runner.invoke(cli, ["init", "--repo", str(repo)])
        wiki = repo.parent / "repo.wiki"
        before = sorted(path.name for path in wiki.rglob("*.md"))

        second = runner.invoke(cli, ["init", "--repo", str(repo)])

        assert second.exit_code == 0, second.output
        assert sorted(path.name for path in wiki.rglob("*.md")) == before
        assert "no seed documents were added" in second.output


class TestIndexCommand:
    """Test the index command."""

    def test_index_text_output(self, runner, repo, monkeypatch):
        """The text format lists the concepts."""
        runner.invoke(cli, ["init", "--repo", str(repo)])
        monkeypatch.setenv("MARKRAG_REPO", str(repo))

        result = runner.invoke(cli, ["index", "--repo", str(repo)])

        assert result.exit_code == 0, result.output
        assert "Total concepts" in result.output
        assert "readme" in result.output

    def test_index_json_output(self, runner, repo, monkeypatch):
        """The JSON format is machine readable."""
        runner.invoke(cli, ["init", "--repo", str(repo)])
        monkeypatch.setenv("MARKRAG_REPO", str(repo))

        result = runner.invoke(cli, ["index", "--repo", str(repo), "--format", "json"])

        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["count"] >= 1
        assert "concepts" in payload


class TestSearchCommand:
    """Test the search command."""

    def test_search_text_output(self, runner, repo, monkeypatch):
        """The text format shows the retrieval diagnostics."""
        runner.invoke(cli, ["init", "--repo", str(repo)])
        monkeypatch.setenv("MARKRAG_REPO", str(repo))

        result = runner.invoke(cli, ["search", "readme", "--repo", str(repo)])

        assert result.exit_code == 0, result.output
        assert "Query: readme" in result.output
        assert "Retrieval:" in result.output

    def test_search_json_output(self, runner, repo, monkeypatch):
        """The JSON format carries the full result payload."""
        runner.invoke(cli, ["init", "--repo", str(repo)])
        monkeypatch.setenv("MARKRAG_REPO", str(repo))

        result = runner.invoke(cli, ["search", "readme", "--repo", str(repo), "--format", "json"])

        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["query"] == "readme"
        assert "results" in payload

    def test_search_with_filters(self, runner, repo, monkeypatch):
        """Type and status filters are accepted."""
        runner.invoke(cli, ["init", "--repo", str(repo)])
        monkeypatch.setenv("MARKRAG_REPO", str(repo))

        result = runner.invoke(
            cli,
            ["search", "readme", "--repo", str(repo), "--type", "Project", "--status", "draft"],
        )

        assert result.exit_code == 0, result.output

    def test_search_top_k(self, runner, repo, monkeypatch):
        """The top-k option is honoured."""
        runner.invoke(cli, ["init", "--repo", str(repo)])
        monkeypatch.setenv("MARKRAG_REPO", str(repo))

        result = runner.invoke(
            cli, ["search", "readme", "--repo", str(repo), "--top-k", "1", "--format", "json"]
        )

        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert len(payload["results"]) <= 1


class TestReadCommand:
    """Test the read command."""

    def test_read_text_output(self, runner, repo, monkeypatch):
        """Reading a concept prints its body."""
        runner.invoke(cli, ["init", "--repo", str(repo)])
        monkeypatch.setenv("MARKRAG_REPO", str(repo))

        result = runner.invoke(cli, ["read", "readme", "--repo", str(repo)])

        assert result.exit_code == 0, result.output
        assert "Content:" in result.output

    def test_read_json_output(self, runner, repo, monkeypatch):
        """The JSON format carries the full document."""
        runner.invoke(cli, ["init", "--repo", str(repo)])
        monkeypatch.setenv("MARKRAG_REPO", str(repo))

        result = runner.invoke(cli, ["read", "readme", "--repo", str(repo), "--format", "json"])

        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["concept_id"] == "readme"
        assert "content" in payload

    def test_read_missing_concept_reports_an_error(self, runner, repo, monkeypatch):
        """An unknown concept is an error, not a crash."""
        runner.invoke(cli, ["init", "--repo", str(repo)])
        monkeypatch.setenv("MARKRAG_REPO", str(repo))

        result = runner.invoke(cli, ["read", "no/such/concept", "--repo", str(repo)])

        assert result.exit_code == 0, result.output
        assert "Error:" in result.output


class TestStatusCommand:
    """Test the status command."""

    def test_status_text_output(self, runner, repo, monkeypatch):
        """The text format shows the bundle statistics."""
        runner.invoke(cli, ["init", "--repo", str(repo)])
        monkeypatch.setenv("MARKRAG_REPO", str(repo))

        result = runner.invoke(cli, ["status", "--repo", str(repo)])

        assert result.exit_code == 0, result.output
        assert "Bundle Statistics:" in result.output
        assert "By Type:" in result.output

    def test_status_json_output(self, runner, repo, monkeypatch):
        """The JSON format is machine readable."""
        runner.invoke(cli, ["init", "--repo", str(repo)])
        monkeypatch.setenv("MARKRAG_REPO", str(repo))

        result = runner.invoke(cli, ["status", "--repo", str(repo), "--format", "json"])

        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["total_docs"] >= 1
        assert "by_type" in payload


class TestGlobalOptions:
    """Test the group-level options."""

    def test_version(self, runner):
        """The version flag reports the package version."""
        result = runner.invoke(cli, ["--version"])

        assert result.exit_code == 0
        assert "markrag" in result.output

    def test_global_format_option(self, runner, repo, monkeypatch):
        """The group-level --format applies to subcommands."""
        runner.invoke(cli, ["init", "--repo", str(repo)])
        monkeypatch.setenv("MARKRAG_REPO", str(repo))

        result = runner.invoke(cli, ["--format", "json", "status", "--repo", str(repo)])

        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert "total_docs" in payload

    def test_repo_option_overrides_the_environment(self, runner, repo, monkeypatch):
        """An explicit --repo wins over MARKRAG_REPO."""
        other = repo.parent / "other"
        other.mkdir()
        monkeypatch.setenv("MARKRAG_REPO", str(other))

        result = runner.invoke(cli, ["init", "--repo", str(repo)])

        assert result.exit_code == 0, result.output
        assert (repo.parent / "repo.wiki").exists()

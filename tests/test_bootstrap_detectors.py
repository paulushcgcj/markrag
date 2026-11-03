"""Tests for the repository detectors used to seed a wiki.

These functions read real build files. Each one has to cope with a file that
is absent, empty, or malformed, because a wiki is generated from whatever
happens to be in the repository.
"""

import json

from markrag.bootstrap import (
    _detect_test_layout,
    _makefile_targets,
    _parse_compose,
    _parse_package_json,
    _parse_pom,
    _parse_pyproject,
    _parse_requirements,
    _python_module_summary,
    _readme_sections,
    _readme_title,
    _seed_dependencies,
    _seed_python_modules,
    _seed_testing,
    _seed_workflow,
)


class TestPyproject:
    """Test pyproject.toml parsing."""

    def test_runtime_dependencies(self, tmp_path):
        """Project dependencies become a runtime group."""
        path = tmp_path / "pyproject.toml"
        path.write_text(
            '[project]\nname = "x"\ndependencies = ["httpx>=1.0", "pydantic"]\n', encoding="utf-8"
        )

        groups = _parse_pyproject(path)
        assert groups["runtime (project.dependencies)"] == ["httpx>=1.0", "pydantic"]

    def test_optional_and_dev_groups(self, tmp_path):
        """Optional extras and dependency groups are separated."""
        path = tmp_path / "pyproject.toml"
        path.write_text(
            "[project]\n"
            'name = "x"\n'
            "[project.optional-dependencies]\n"
            'openai = ["openai"]\n'
            "[dependency-groups]\n"
            'dev = ["pytest"]\n',
            encoding="utf-8",
        )

        groups = _parse_pyproject(path)
        assert groups["optional: openai"] == ["openai"]
        assert groups["group: dev"] == ["pytest"]

    def test_malformed_file_returns_nothing(self, tmp_path):
        """A broken TOML file yields no groups instead of raising."""
        path = tmp_path / "pyproject.toml"
        path.write_text("this is not toml [[[", encoding="utf-8")

        assert _parse_pyproject(path) == {}

    def test_missing_file_returns_nothing(self, tmp_path):
        """A missing file yields no groups."""
        assert _parse_pyproject(tmp_path / "absent.toml") == {}


class TestPackageJson:
    """Test package.json parsing."""

    def test_runtime_and_dev_dependencies(self, tmp_path):
        """Runtime and development dependencies are separated."""
        path = tmp_path / "package.json"
        path.write_text(
            json.dumps(
                {
                    "dependencies": {"react": "^19.0.0"},
                    "devDependencies": {"vitest": "^3.0.0"},
                }
            ),
            encoding="utf-8",
        )

        groups = _parse_package_json(path)
        assert groups["runtime (dependencies)"] == ["react ^19.0.0"]
        assert groups["development (devDependencies)"] == ["vitest ^3.0.0"]

    def test_malformed_file_returns_nothing(self, tmp_path):
        """Broken JSON yields no groups."""
        path = tmp_path / "package.json"
        path.write_text("{not json", encoding="utf-8")

        assert _parse_package_json(path) == {}


class TestPom:
    """Test Maven pom.xml parsing."""

    def test_dependencies_are_read(self, tmp_path):
        """Maven coordinates become dependency entries."""
        path = tmp_path / "pom.xml"
        path.write_text(
            """<project xmlns="http://maven.apache.org/POM/4.0.0">
  <dependencies>
    <dependency>
      <groupId>org.springframework.boot</groupId>
      <artifactId>spring-boot-starter-web</artifactId>
    </dependency>
    <dependency>
      <artifactId>standalone</artifactId>
    </dependency>
  </dependencies>
</project>
""",
            encoding="utf-8",
        )

        groups = _parse_pom(path)
        assert groups["maven (dependencies)"] == [
            "org.springframework.boot:spring-boot-starter-web",
            "standalone",
        ]

    def test_malformed_file_returns_nothing(self, tmp_path):
        """Broken XML yields no groups."""
        path = tmp_path / "pom.xml"
        path.write_text("<project><unclosed>", encoding="utf-8")

        assert _parse_pom(path) == {}


class TestCompose:
    """Test docker compose parsing."""

    def test_services_are_read(self, tmp_path):
        """Service names and images are listed."""
        path = tmp_path / "docker-compose.yml"
        path.write_text(
            "services:\n  postgres:\n    image: postgres:16\n  api:\n    build: .\n",
            encoding="utf-8",
        )

        groups = _parse_compose(path)
        assert groups["compose services"] == [
            "api (image: build)",
            "postgres (image: postgres:16)",
        ]

    def test_non_mapping_file_returns_nothing(self, tmp_path):
        """A compose file that is not a mapping yields nothing."""
        path = tmp_path / "compose.yml"
        path.write_text("- just\n- a\n- list\n", encoding="utf-8")

        assert _parse_compose(path) == {}

    def test_malformed_file_returns_nothing(self, tmp_path):
        """Broken YAML yields nothing."""
        path = tmp_path / "compose.yml"
        path.write_text("services: [unclosed\n", encoding="utf-8")

        assert _parse_compose(path) == {}


class TestRequirements:
    """Test requirements.txt parsing."""

    def test_requirements_are_read(self, tmp_path):
        """Comments and options are skipped."""
        path = tmp_path / "requirements.txt"
        path.write_text("# a comment\n\n-r other.txt\nhttpx>=1.0\nfastapi\n", encoding="utf-8")

        groups = _parse_requirements(path)
        assert groups["requirements"] == ["httpx>=1.0", "fastapi"]

    def test_empty_file_returns_nothing(self, tmp_path):
        """A file with only comments yields nothing."""
        path = tmp_path / "requirements.txt"
        path.write_text("# nothing here\n", encoding="utf-8")

        assert _parse_requirements(path) == {}


class TestMakefile:
    """Test Makefile target extraction."""

    def test_targets_are_read(self, tmp_path):
        """Target names are collected in file order."""
        path = tmp_path / "Makefile"
        path.write_text(
            ".PHONY: all\n\nbuild:\n\techo build\n\ntest:\n\techo test\n\nVAR := value\n",
            encoding="utf-8",
        )

        assert _makefile_targets(path) == ["build", "test"]


class TestSeedDependencies:
    """Test the dependency seed document."""

    def test_no_manifest_means_no_document(self, tmp_path):
        """A repository without manifests gets no dependency document."""
        assert _seed_dependencies(tmp_path) is None

    def test_manifest_without_dependencies_is_still_cited(self, tmp_path):
        """A manifest with nothing to list is still a real source."""
        (tmp_path / "package.json").write_text(json.dumps({"name": "x"}), encoding="utf-8")

        doc = _seed_dependencies(tmp_path)
        assert doc is not None
        assert doc.sources[0].resource == "package.json"
        assert "No dependency declarations" in doc.body

    def test_sources_cite_real_files(self, tmp_path):
        """Every cited source exists in the repository."""
        (tmp_path / "package.json").write_text(
            json.dumps({"dependencies": {"react": "^19.0.0"}}), encoding="utf-8"
        )

        doc = _seed_dependencies(tmp_path)
        assert doc is not None
        for source in doc.sources:
            assert (tmp_path / source.resource).exists()


class TestSeedWorkflow:
    """Test the development workflow seed document."""

    def test_no_build_files_means_no_document(self, tmp_path):
        """A repository with no build entry point gets no workflow document."""
        assert _seed_workflow(tmp_path) is None

    def test_makefile_targets_are_listed(self, tmp_path):
        """Make targets appear in the workflow document."""
        (tmp_path / "Makefile").write_text("build:\n\techo\n", encoding="utf-8")

        doc = _seed_workflow(tmp_path)
        assert doc is not None
        assert "- `make build`" in doc.body

    def test_npm_scripts_are_listed(self, tmp_path):
        """npm scripts appear in the workflow document."""
        (tmp_path / "package.json").write_text(
            json.dumps({"scripts": {"build": "vite build", "test": "vitest"}}), encoding="utf-8"
        )

        doc = _seed_workflow(tmp_path)
        assert doc is not None
        assert "`npm run build`" in doc.body
        assert "`npm run test`" in doc.body

    def test_ci_workflows_are_listed(self, tmp_path):
        """GitHub Actions workflows appear in the workflow document."""
        workflows = tmp_path / ".github" / "workflows"
        workflows.mkdir(parents=True)
        (workflows / "ci.yml").write_text("name: CI\n", encoding="utf-8")

        doc = _seed_workflow(tmp_path)
        assert doc is not None
        assert ".github/workflows/ci.yml" in doc.body


class TestTestLayout:
    """Test the test-layout detector."""

    def test_test_directories_are_found(self, tmp_path):
        """Test source directories are counted."""
        tests = tmp_path / "tests"
        tests.mkdir()
        (tests / "test_a.py").write_text("", encoding="utf-8")
        (tests / "test_b.py").write_text("", encoding="utf-8")

        assert _detect_test_layout(tmp_path) == [("tests", 2)]

    def test_non_test_files_are_ignored(self, tmp_path):
        """Files that are not test sources do not count."""
        tests = tmp_path / "tests"
        tests.mkdir()
        (tests / "test_a.py").write_text("", encoding="utf-8")
        (tests / "notes.txt").write_text("", encoding="utf-8")

        assert _detect_test_layout(tmp_path) == [("tests", 1)]

    def test_no_tests_means_no_document(self, tmp_path):
        """A repository without tests gets no testing document."""
        assert _seed_testing(tmp_path) is None

    def test_frameworks_are_detected(self, tmp_path):
        """Test frameworks are recognised from their config files."""
        tests = tmp_path / "tests"
        tests.mkdir()
        (tests / "test_a.py").write_text("", encoding="utf-8")
        (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
        (tmp_path / "vitest.config.ts").write_text("export default {}\n", encoding="utf-8")

        doc = _seed_testing(tmp_path)
        assert doc is not None
        assert "pytest" in doc.body
        assert "Vitest" in doc.body
        assert "tests/` — 1 test source files" in doc.body


class TestPythonModuleSummary:
    """Test the Python module summary extractor."""

    def test_docstring_and_public_api(self, tmp_path):
        """The docstring and public functions are extracted."""
        path = tmp_path / "module.py"
        path.write_text(
            '"""Module summary.\n\nMore detail here.\n"""\n\n\n'
            "def public_function():\n    pass\n\n\n"
            "def _private():\n    pass\n\n\n"
            "class PublicClass:\n    pass\n",
            encoding="utf-8",
        )

        summary, api = _python_module_summary(path)
        assert summary == "Module summary."
        assert "`public_function()`" in api
        assert "_private" not in "".join(api)
        assert "`PublicClass`" in api

    def test_syntax_error_yields_nothing(self, tmp_path):
        """An unparseable file yields no summary."""
        path = tmp_path / "broken.py"
        path.write_text("def (:\n", encoding="utf-8")

        assert _python_module_summary(path) == ("", [])

    def test_python_modules_are_seeded(self, tmp_path):
        """Python package modules become module documents."""
        package = tmp_path / "src" / "my_pkg"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text('"""My package."""\n', encoding="utf-8")
        (package / "core.py").write_text(
            '"""Core logic."""\n\n\ndef run():\n    return 1\n', encoding="utf-8"
        )

        docs = _seed_python_modules(tmp_path)
        assert docs
        concept_ids = {doc.concept_id for doc in docs}
        assert "modules/core" in concept_ids
        # Private modules are not documented.
        assert "modules/__init__" not in concept_ids

    def test_python_modules_need_a_src_directory(self, tmp_path):
        """A repository without src/ has no module documents."""
        assert _seed_python_modules(tmp_path) == []


class TestReadme:
    """Test the README helpers."""

    def test_title_from_the_first_heading(self):
        """The first markdown heading becomes the title."""
        assert _readme_title("# My Project\n\nText.\n", "fallback") == "My Project"

    def test_title_falls_back(self):
        """A README without a heading uses the fallback."""
        assert _readme_title("Just text.\n", "fallback") == "fallback"

    def test_sections_are_split(self):
        """Second-level headings become sections."""
        sections = _readme_sections("# Title\n\nIntro.\n\n## One\n\nFirst.\n\n## Two\n\nSecond.\n")

        assert [name for name, _ in sections] == ["One", "Two"]
        assert "First." in dict(sections)["One"]

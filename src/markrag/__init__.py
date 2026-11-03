"""markrag - OKF-backed RAG MCP server for per-repo wikis."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("markrag")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "unknown"

__all__ = ["__version__"]

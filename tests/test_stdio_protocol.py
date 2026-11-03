"""End-to-end tests for the MCP stdio transport.

The MCP stdio transport reserves stdout for JSON-RPC frames. Anything else
written to stdout - a banner, a log line, a stray print - corrupts the
protocol and breaks the client. These tests drive the server as a real
subprocess and inspect every byte it puts on stdout.
"""

import json
import os
import select
import subprocess
import sys
from pathlib import Path

import pytest

PROTOCOL_VERSION = "2024-11-05"


def _frame(payload: dict) -> str:
    """Encode a JSON-RPC message as a stdio frame."""
    return json.dumps(payload) + "\n"


def _read_frame(stream, timeout: float = 60.0) -> dict:
    """Read one JSON-RPC frame from a stream."""
    line = stream.readline()
    assert line, "server closed stdout without answering"
    return json.loads(line)


def _read_optional_line(stream, wait: float = 1.0) -> str | None:
    """Read a line if one is already available, otherwise return None.

    Used to prove that nothing extra is waiting on stdout without blocking
    forever when the server is behaving correctly.
    """
    ready, _, _ = select.select([stream], [], [], wait)
    if not ready:
        return None
    return stream.readline() or None


@pytest.fixture
def stdio_server(tmp_path):
    """Start the MCP server as a subprocess with an empty wiki."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "README.md").write_text("# Test Repo\n\nA test repository.\n", encoding="utf-8")

    env = dict(os.environ)
    env.update(
        {
            "MARKRAG_REPO": str(repo),
            "MARKRAG_INDEX_DIR": str(tmp_path / "index"),
            "MARKRAG_EMBEDDINGS": "off",
            "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
        }
    )

    process = subprocess.Popen(  # noqa: S603
        [sys.executable, "-m", "markrag.cli", "serve"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
        bufsize=1,
    )

    try:
        yield process
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover - defensive cleanup
            process.kill()


def _initialize(process) -> dict:
    """Perform the MCP initialize handshake."""
    assert process.stdin is not None
    assert process.stdout is not None

    process.stdin.write(
        _frame(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "markrag-test", "version": "1.0.0"},
                },
            }
        )
    )
    process.stdin.flush()
    return _read_frame(process.stdout)


class TestStdioProtocol:
    """Test the stdio transport."""

    def test_initialize_is_answered(self, stdio_server):
        """The server completes the MCP handshake."""
        response = _initialize(stdio_server)

        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 1
        assert "result" in response
        assert response["result"]["serverInfo"]["name"] == "markrag"

    def test_stdout_carries_only_jsonrpc(self, stdio_server):
        """Every line on stdout parses as a JSON-RPC message."""
        _initialize(stdio_server)

        assert stdio_server.stdin is not None
        assert stdio_server.stdout is not None

        stdio_server.stdin.write(
            _frame({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})
        )
        stdio_server.stdin.write(
            _frame({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        )
        stdio_server.stdin.flush()

        response = _read_frame(stdio_server.stdout)
        assert response["id"] == 2

        # Nothing else may be waiting on stdout: a banner or a log line would
        # show up here as a non-JSON-RPC payload.
        extra = _read_optional_line(stdio_server.stdout)
        assert extra is None, f"unexpected non-JSON-RPC output on stdout: {extra!r}"

    def test_banner_goes_to_stderr(self, stdio_server):
        """The startup banner is written to stderr, not stdout."""
        _initialize(stdio_server)

        assert stdio_server.stderr is not None
        stderr_line = stdio_server.stderr.readline()
        assert "markrag" in stderr_line.lower()

    def test_tools_are_discoverable(self, stdio_server):
        """All five tools are advertised."""
        _initialize(stdio_server)

        assert stdio_server.stdin is not None
        assert stdio_server.stdout is not None

        stdio_server.stdin.write(
            _frame({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})
        )
        stdio_server.stdin.write(
            _frame({"jsonrpc": "2.0", "id": 3, "method": "tools/list", "params": {}})
        )
        stdio_server.stdin.flush()

        response = _read_frame(stdio_server.stdout)
        tool_names = {tool["name"] for tool in response["result"]["tools"]}

        assert tool_names == {"okf_index", "okf_search", "okf_read", "okf_write", "okf_status"}

    def test_tool_call_returns_a_result(self, stdio_server):
        """A tool call round-trips over stdio."""
        _initialize(stdio_server)

        assert stdio_server.stdin is not None
        assert stdio_server.stdout is not None

        stdio_server.stdin.write(
            _frame({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})
        )
        stdio_server.stdin.write(
            _frame(
                {
                    "jsonrpc": "2.0",
                    "id": 4,
                    "method": "tools/call",
                    "params": {"name": "okf_index", "arguments": {}},
                }
            )
        )
        stdio_server.stdin.flush()

        response = _read_frame(stdio_server.stdout)
        assert response["id"] == 4
        assert "result" in response
        assert not response.get("isError")

"""Tests for the vorlo-mcp server (protocol + tools + formatters)."""
from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from vorlo_trace import mcp

FAILED_SESSION = {
    "session_id": "sess_1",
    "agent_name": "order-agent",
    "status": "failed",
    "total_duration_ms": 812,
    "steps": [
        {"step_number": 1, "tool_name": "get_order", "status": "success", "latency_ms": 100},
        {
            "step_number": 2,
            "tool_name": "charge_card",
            "status": "failed",
            "latency_ms": 400,
            "error_title": "Stripe authentication failed",
            "error_root_cause": "The API key expired.",
            "error_fix_hint": "Rotate the Stripe key.",
            "error_confidence": "verified",
        },
    ],
}


def _mock_api(responses: dict[str, Any]):
    """Patch requests.get to serve canned responses keyed by path prefix."""

    def fake_get(url: str, **kwargs: Any) -> MagicMock:
        path = url.split("up.railway.app")[-1]
        for key, value in responses.items():
            if path.startswith(key):
                res = MagicMock(ok=True, status_code=200)
                res.json.return_value = value
                return res
        return MagicMock(ok=False, status_code=404)

    return patch.object(mcp.requests, "get", side_effect=fake_get)


@pytest.fixture(autouse=True)
def _api_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("VORLO_API_KEY", "vrlo_test")


class TestProtocol:
    def _collect(self, message: dict[str, Any]) -> list[dict[str, Any]]:
        frames: list[dict[str, Any]] = []
        with patch.object(mcp, "_send", side_effect=frames.append):
            mcp.handle_message(message)
        return frames

    def test_initialize(self) -> None:
        frames = self._collect({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2025-03-26"},
        })
        assert frames[0]["result"]["protocolVersion"] == "2025-03-26"
        assert frames[0]["result"]["serverInfo"]["name"] == "vorlo"
        assert "tools" in frames[0]["result"]["capabilities"]

    def test_tools_list(self) -> None:
        frames = self._collect({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        names = [t["name"] for t in frames[0]["result"]["tools"]]
        assert names == [
            "why_did_my_last_run_fail",
            "get_session_diagnosis",
            "list_recent_sessions",
            "get_failure_clusters",
        ]

    def test_unknown_method_and_notifications(self) -> None:
        frames = self._collect({"jsonrpc": "2.0", "id": 3, "method": "bogus/method"})
        assert frames[0]["error"]["code"] == -32601
        # notifications get no response
        assert self._collect({"jsonrpc": "2.0", "method": "notifications/initialized"}) == []

    def test_tool_failure_is_an_isError_result(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("VORLO_API_KEY")
        frames = self._collect({
            "jsonrpc": "2.0", "id": 4, "method": "tools/call",
            "params": {"name": "list_recent_sessions", "arguments": {}},
        })
        assert frames[0]["result"]["isError"] is True
        assert "VORLO_API_KEY" in frames[0]["result"]["content"][0]["text"]


class TestTools:
    def test_why_did_my_last_run_fail(self) -> None:
        with _mock_api({
            "/v1/sessions?page=1&status=failed": {"sessions": [{"session_id": "sess_1"}]},
            "/v1/sessions/sess_1": FAILED_SESSION,
        }):
            text = mcp.call_tool("why_did_my_last_run_fail", {})
        assert "order-agent" in text
        assert "FAILED at step 2" in text
        assert "Root cause: The API key expired." in text
        assert "Fix: Rotate the Stripe key." in text
        assert "vorlo.dev/sessions/sess_1" in text

    def test_clean_runs(self) -> None:
        with _mock_api({"/v1/sessions?page=1&status=failed": {"sessions": []}}):
            text = mcp.call_tool("why_did_my_last_run_fail", {})
        assert "No failed runs" in text

    def test_list_and_clusters(self) -> None:
        with _mock_api({
            "/v1/sessions?page=1&status=failed": {
                "total": 1,
                "sessions": [{
                    "session_id": "sess_9", "agent_name": "mail-agent",
                    "status": "failed", "total_steps": 4,
                    "last_error_title": "Gmail rate limit",
                }],
            },
            "/v1/sessions/failure-clusters?days=14": {
                "total_failed_sessions": 6,
                "clusters": [{
                    "title": "OAuth/authentication failures",
                    "affected_sessions": 5,
                    "root_cause": "Tokens expiring",
                    "fix_hint": "Reconnect the account",
                }],
            },
        }):
            listed = mcp.call_tool("list_recent_sessions", {"status": "failed"})
            clusters = mcp.call_tool("get_failure_clusters", {"days": 14})
        assert "[failed] sess_9 · mail-agent · 4 steps — Gmail rate limit" in listed
        assert "OAuth/authentication failures — 5 sessions" in clusters
        assert "Fix: Reconnect the account" in clusters


class TestFormatters:
    def test_diagnosis_marks_steps(self) -> None:
        text = mcp.format_session_diagnosis(FAILED_SESSION)
        assert "✓ 1. get_order (100ms)" in text
        assert "✗ 2. charge_card (400ms)" in text
        assert "Confidence: verified" in text

    def test_empty_states(self) -> None:
        assert "No sessions" in mcp.format_session_list({"sessions": []})
        assert "No failure clusters" in mcp.format_clusters({"clusters": []})

"""
Vorlo MCP server — debug your AI agent from inside your AI coding assistant.

Exposes Vorlo's diagnosis API as MCP tools so Claude Code / Cursor / any MCP
client can ask "why did my last run fail?" and get the root cause + fix
without leaving the editor.

A minimal, correct JSON-RPC 2.0 server over stdio (newline-delimited JSON,
per the MCP stdio transport). Mirrors the vorlo-mcp bin in the JS package.

Usage:
    pip install vorlo-trace
    claude mcp add vorlo --env VORLO_API_KEY=vrlo_... -- vorlo-mcp
    # or, without installing first:
    claude mcp add vorlo --env VORLO_API_KEY=vrlo_... -- uvx --from vorlo-trace vorlo-mcp
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Optional
from urllib.parse import quote

import requests

SERVER_NAME = "vorlo"
SERVER_VERSION = "0.3.0"
_DEFAULT_SERVER_URL = "https://vorlo-server-production.up.railway.app"


# ── Vorlo API client ─────────────────────────────────────────────────────

def _api_base() -> str:
    return (os.environ.get("VORLO_SERVER_URL") or _DEFAULT_SERVER_URL).rstrip("/")


def _vorlo_get(path: str) -> dict[str, Any]:
    api_key = os.environ.get("VORLO_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "VORLO_API_KEY is not set. Create a key at https://www.vorlo.dev/settings "
            "and add it to the MCP server env."
        )
    res = requests.get(
        f"{_api_base()}{path}",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=15,
    )
    if not res.ok:
        raise RuntimeError(f"Vorlo API {path} responded {res.status_code}")
    return res.json()


# ── Formatters ────────────────────────────────────────────────────────────

def format_session_diagnosis(session: dict[str, Any]) -> str:
    steps = session.get("steps") or []
    failed = next((s for s in steps if s.get("status") == "failed"), None)
    lines = [
        f"Session {session.get('session_id', '')} — agent \"{session.get('agent_name', 'unknown')}\"",
        f"Status: {session.get('status') or ('failed' if failed else 'success')} · "
        f"{len(steps)} steps · {session.get('total_duration_ms', 0)}ms total",
        "",
    ]

    if failed:
        lines.append(
            f"FAILED at step {failed.get('step_number', '?')} — tool: {failed.get('tool_name', 'unknown')}"
        )
        lines.append(f"Title: {failed.get('error_title') or 'Step failed'}")
        if failed.get("error_plain_english"):
            lines.append(f"What happened: {failed['error_plain_english']}")
        if failed.get("error_root_cause"):
            lines.append(f"Root cause: {failed['error_root_cause']}")
        if failed.get("error_fix_hint"):
            lines.append(f"Fix: {failed['error_fix_hint']}")
        if failed.get("error_confidence"):
            lines.append(f"Confidence: {failed['error_confidence']}")
        lines.append("")

    lines.append("Steps:")
    for s in steps:
        mark = "✗" if s.get("status") == "failed" else "✓"
        lines.append(
            f"  {mark} {s.get('step_number', '?')}. {s.get('tool_name', 'unknown')} "
            f"({s.get('latency_ms', 0)}ms)"
        )
    lines.append("")
    lines.append(f"Replay: https://www.vorlo.dev/sessions/{session.get('session_id', '')}")
    return "\n".join(lines)


def format_session_list(data: dict[str, Any]) -> str:
    sessions = data.get("sessions") or []
    if not sessions:
        return "No sessions found."
    lines = [f"{len(sessions)} of {data.get('total', len(sessions))} sessions:"]
    for s in sessions:
        status = s.get("status") or ("failed" if int(s.get("fail_count") or 0) > 0 else "success")
        fail_note = (
            f" — {s.get('last_error_title')}"
            if status == "failed" and s.get("last_error_title")
            else ""
        )
        lines.append(
            f"[{status}] {s.get('session_id')} · {s.get('agent_name', 'unknown')} · "
            f"{s.get('total_steps', 0)} steps{fail_note}"
        )
    return "\n".join(lines)


def format_clusters(data: dict[str, Any]) -> str:
    clusters = data.get("clusters") or []
    if not clusters:
        return "No failure clusters in this window. Clean runs!"
    lines = [
        f"{len(clusters)} failure clusters ({data.get('total_failed_sessions', 0)} failed sessions):"
    ]
    for i, c in enumerate(clusters):
        lines.append(
            f"{i + 1}. {c.get('title') or c.get('cluster_key') or 'cluster'} — "
            f"{c.get('affected_sessions', 0)} sessions"
        )
        if c.get("root_cause"):
            lines.append(f"   Root cause: {c['root_cause']}")
        if c.get("fix_hint"):
            lines.append(f"   Fix: {c['fix_hint']}")
    return "\n".join(lines)


# ── Tools ─────────────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "why_did_my_last_run_fail",
        "description": (
            "Get the diagnosis for the agent's most recent failed run: root cause, "
            "the exact fix, and the step where it broke. Start here when an agent "
            "is misbehaving."
        ),
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "get_session_diagnosis",
        "description": (
            "Full diagnosis and step replay summary for one Vorlo session id "
            "(root cause, fix, confidence, every step with status and latency)."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["session_id"],
            "properties": {
                "session_id": {"type": "string", "description": "The Vorlo session id"}
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "list_recent_sessions",
        "description": (
            "List recent agent runs traced by Vorlo, optionally filtered by status "
            "(all | failed | success | running)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["all", "failed", "success", "running"]},
                "page": {"type": "integer", "minimum": 1},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "get_failure_clusters",
        "description": (
            "Failures grouped by root cause across recent sessions — fix the pattern, "
            "not the symptom. Optionally set the lookback window in days (1-30)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"days": {"type": "integer", "minimum": 1, "maximum": 30}},
            "additionalProperties": False,
        },
    },
]


def call_tool(name: str, args: dict[str, Any]) -> str:
    if name == "why_did_my_last_run_fail":
        listing = _vorlo_get("/v1/sessions?page=1&status=failed")
        sessions = listing.get("sessions") or []
        if not sessions:
            return "No failed runs found — the most recent sessions all succeeded."
        detail = _vorlo_get(f"/v1/sessions/{quote(str(sessions[0].get('session_id')))}")
        return format_session_diagnosis(detail)

    if name == "get_session_diagnosis":
        session_id = str(args.get("session_id") or "")
        if not session_id:
            raise ValueError("session_id is required")
        detail = _vorlo_get(f"/v1/sessions/{quote(session_id)}")
        return format_session_diagnosis(detail)

    if name == "list_recent_sessions":
        status = args.get("status") if isinstance(args.get("status"), str) else "all"
        page = int(args.get("page") or 1)
        listing = _vorlo_get(f"/v1/sessions?page={page}&status={quote(str(status))}")
        return format_session_list(listing)

    if name == "get_failure_clusters":
        days = int(args.get("days") or 7)
        data = _vorlo_get(f"/v1/sessions/failure-clusters?days={days}")
        return format_clusters(data)

    raise ValueError(f"Unknown tool: {name}")


# ── JSON-RPC 2.0 over stdio (newline-delimited) ──────────────────────────

def _send(message: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def _reply(msg_id: Any, result: dict[str, Any]) -> None:
    _send({"jsonrpc": "2.0", "id": msg_id, "result": result})


def _reply_error(msg_id: Any, code: int, message: str) -> None:
    _send({"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}})


def handle_message(msg: dict[str, Any]) -> None:
    msg_id = msg.get("id")
    is_notification = "id" not in msg
    method = msg.get("method", "")
    params = msg.get("params") or {}

    if method == "initialize":
        _reply(msg_id, {
            "protocolVersion": params.get("protocolVersion") or "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
        })
    elif method == "ping":
        _reply(msg_id, {})
    elif method == "tools/list":
        _reply(msg_id, {"tools": TOOLS})
    elif method == "tools/call":
        name = str(params.get("name") or "")
        args = params.get("arguments") or {}
        try:
            text = call_tool(name, args)
            _reply(msg_id, {"content": [{"type": "text", "text": text}]})
        except Exception as err:  # tool failures are results, not protocol errors
            _reply(msg_id, {
                "content": [{"type": "text", "text": str(err)}],
                "isError": True,
            })
    elif not is_notification:
        _reply_error(msg_id, -32601, f"Method not found: {method}")


def main() -> None:
    """Console entry point: vorlo-mcp"""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            _reply_error(None, -32700, "Parse error")
            continue
        handle_message(msg)


if __name__ == "__main__":
    main()

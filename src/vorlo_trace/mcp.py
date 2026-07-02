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
SERVER_VERSION = "0.5.0"
_DEFAULT_SERVER_URL = "https://vorlo-server-production.up.railway.app"


# ── Vorlo API client ─────────────────────────────────────────────────────

def _api_base() -> str:
    return (os.environ.get("VORLO_SERVER_URL") or _DEFAULT_SERVER_URL).rstrip("/")


def _api_key() -> str:
    api_key = os.environ.get("VORLO_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "VORLO_API_KEY is not set. Create a key at https://www.vorlo.dev/settings "
            "and add it to the MCP server env."
        )
    return api_key


def _vorlo_get(path: str) -> dict[str, Any]:
    res = requests.get(
        f"{_api_base()}{path}",
        headers={"Authorization": f"Bearer {_api_key()}"},
        timeout=15,
    )
    if not res.ok:
        raise RuntimeError(f"Vorlo API {path} responded {res.status_code}")
    return res.json()


def _vorlo_post(path: str, body: dict[str, Any]) -> dict[str, Any]:
    res = requests.post(
        f"{_api_base()}{path}",
        headers={"Authorization": f"Bearer {_api_key()}"},
        json=body,
        timeout=15,
    )
    if not res.ok:
        raise RuntimeError(f"Vorlo API {path} responded {res.status_code}")
    return res.json()


def _latest_failed_session() -> Optional[dict[str, Any]]:
    """Full detail of the most recent failed session, or None."""
    listing = _vorlo_get("/v1/sessions?page=1&status=failed")
    sessions = listing.get("sessions") or []
    if not sessions:
        return None
    return _vorlo_get(f"/v1/sessions/{quote(str(sessions[0].get('session_id')))}")


def _failed_step(session: dict[str, Any]) -> Optional[dict[str, Any]]:
    return next(
        (s for s in (session.get("steps") or []) if s.get("status") == "failed"),
        None,
    )


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


_CONFIDENCE_NOTES = {
    "verified": (
        "VERIFIED — this exact fix worked before (confirmed by a developer or "
        "by the error stopping). Apply it with confidence."
    ),
    "likely": (
        "LIKELY — matched a curated error pattern, not an AI guess. "
        "Apply it, then confirm the outcome."
    ),
    "guess": (
        "GUESS — an unconfirmed hypothesis from a fresh diagnosis. "
        "Verify it against the code before applying."
    ),
}


def format_fix_briefing(session: dict[str, Any]) -> str:
    """Structured patch briefing for a coding agent to act on."""
    session_id = session.get("session_id", "")
    failed = _failed_step(session)
    if failed is None:
        return (
            f"Session {session_id} has no failed step — nothing to fix. "
            "Use why_did_my_last_run_fail to find the latest failure."
        )

    tool_name = failed.get("tool_name", "unknown")
    confidence = (failed.get("error_confidence") or "guess").lower()
    steps = session.get("steps") or []

    lines = [
        f"# Vorlo Fix Briefing — session {session_id}",
        "",
        "## Failure",
        f"Agent: {session.get('agent_name', 'unknown')}",
        f"Step {failed.get('step_number', '?')} of {len(steps)} — tool `{tool_name}` "
        f"({failed.get('tool_type') or 'unknown'}) failed after {failed.get('latency_ms', 0)}ms",
        f"Error: {failed.get('error') or failed.get('error_title') or 'unknown'}",
        "",
        f"## Diagnosis ({_CONFIDENCE_NOTES.get(confidence, confidence)})",
    ]
    if failed.get("error_root_cause"):
        lines.append(f"Root cause: {failed['error_root_cause']}")
    if failed.get("error_plain_english"):
        lines.append(f"What happened: {failed['error_plain_english']}")
    lines.append(f"Fix: {failed.get('error_fix_hint') or 'No fix hint available — inspect the context below.'}")

    lines.append("")
    lines.append("## Context")
    if failed.get("input"):
        lines.append(f"Tool input: {str(failed['input'])[:800]}")
    if failed.get("reasoning"):
        lines.append(f"Agent reasoning before the call: {str(failed['reasoning'])[:800]}")
    prior = [s for s in steps if s.get("step_number", 0) < failed.get("step_number", 0)]
    if prior:
        lines.append("Steps before the failure:")
        for s in prior:
            mark = "✗" if s.get("status") == "failed" else "✓"
            lines.append(f"  {mark} {s.get('step_number', '?')}. {s.get('tool_name', 'unknown')}")

    lines += [
        "",
        "## Apply the fix",
        f"1. Locate where tool `{tool_name}` is defined, configured, or called in this "
        f"codebase (search for \"{tool_name}\").",
        "2. Apply the fix described in the Diagnosis section.",
        "3. Re-run the agent to confirm the failure is gone.",
        f"4. IMPORTANT: report the outcome by calling the Vorlo MCP tool `confirm_fix` "
        f"with session_id=\"{session_id}\" and worked=true (or false). A confirmed fix "
        "is promoted to VERIFIED and served to every developer who hits this failure next.",
        "",
        f"Replay: https://www.vorlo.dev/sessions/{session_id}",
    ]
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
    {
        "name": "propose_fix",
        "description": (
            "Get a ready-to-apply fix briefing for a failed agent run: the diagnosis, "
            "its confidence (verified fixes have worked before), the failing tool's "
            "input and reasoning context, and step-by-step instructions to apply the "
            "fix in this codebase. Defaults to the most recent failed run. After "
            "applying, report back with confirm_fix."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "Vorlo session id (defaults to the latest failed run)",
                }
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "confirm_fix",
        "description": (
            "Report whether a fix from propose_fix actually worked. worked=true "
            "promotes the diagnosis to VERIFIED for every developer who hits this "
            "failure next; worked=false lowers its confidence. Optionally pass "
            "better_fix with what actually fixed it."
        ),
        "inputSchema": {
            "type": "object",
            "required": ["worked"],
            "properties": {
                "worked": {"type": "boolean", "description": "Did the fix resolve the failure?"},
                "session_id": {
                    "type": "string",
                    "description": "Vorlo session id (defaults to the latest failed run)",
                },
                "better_fix": {
                    "type": "string",
                    "description": "If a different fix worked, describe it — it becomes the served fix",
                },
            },
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

    if name == "propose_fix":
        session_id = str(args.get("session_id") or "")
        if session_id:
            session = _vorlo_get(f"/v1/sessions/{quote(session_id)}")
        else:
            session = _latest_failed_session()
            if session is None:
                return "No failed runs found — the most recent sessions all succeeded."
        return format_fix_briefing(session)

    if name == "confirm_fix":
        worked = bool(args.get("worked"))
        session_id = str(args.get("session_id") or "")
        better_fix = str(args.get("better_fix") or "").strip()

        if session_id:
            session = _vorlo_get(f"/v1/sessions/{quote(session_id)}")
        else:
            session = _latest_failed_session()
            if session is None:
                return "No failed runs found — nothing to confirm."
        failed = _failed_step(session)
        if failed is None:
            return f"Session {session.get('session_id', '')} has no failed step — nothing to confirm."

        # The step's stored error is exactly what the server fingerprinted, so
        # echoing it verbatim attributes this verdict to the right diagnosis.
        result = _vorlo_post("/v1/feedback", {
            "tool_name": failed.get("tool_name", ""),
            "error": failed.get("error", ""),
            "helpful": worked,
            "fix": better_fix,
        })

        confidence = result.get("confidence", "")
        if worked:
            lines = ["Fix outcome recorded: WORKED ✓"]
            if confidence == "verified":
                lines.append(
                    "This diagnosis is now VERIFIED — its fix will be served to every "
                    "developer who hits this failure next. The library just got smarter."
                )
            if better_fix:
                lines.append("Your improved fix replaced the original hint.")
            return "\n".join(lines)
        return (
            "Fix outcome recorded: did not work. The diagnosis's confidence was "
            "lowered so it is not trusted blindly."
        )

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

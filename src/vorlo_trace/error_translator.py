"""
Vorlo Error Translator — transforms raw HTTP/exception errors into
plain-English root cause diagnoses with actionable fix hints.

This is the core of Vorlo's value proposition. Every error that reaches
the dashboard goes through this translator first. We never show raw
stack traces or HTTP status codes to developers.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class ErrorDiagnosis:
    """Machine-readable and human-readable error diagnosis."""

    code: str
    title: str
    plain_english: str
    root_cause: str
    fix_hint: str
    severity: str  # "critical" | "warning" | "info"
    docs_url: str = ""


# ---------------------------------------------------------------------------
# Pattern registry — maps (tool_prefix, http_status, message_regex) → builder
# ---------------------------------------------------------------------------

_PATTERNS: list[dict[str, Any]] = [
    # ── Auth / Token errors ──────────────────────────────────────────────
    {
        "tool_prefix": "stripe",
        "statuses": {401, 403},
        "message_re": re.compile(r"(invalid.api.key|no.such.customer|expired|authentication)", re.I),
        "build": lambda ctx: ErrorDiagnosis(
            code="stripe_auth_error",
            title="Stripe authentication failed",
            plain_english=(
                f"Stripe rejected the request to '{ctx['tool_name']}' with "
                f"HTTP {ctx['http_status']}. This usually means the API key is "
                "invalid, expired, or does not have the required scope."
            ),
            root_cause=_auth_root_cause(ctx),
            fix_hint=(
                "Check your Stripe API key in Settings → Integrations. "
                "Ensure it has the required scope for this operation. "
                "If using test-mode keys against live resources (or vice versa), switch to the correct key."
            ),
            severity="critical",
            docs_url="https://vorlo.dev/errors/stripe_auth_error",
        ),
    },
    {
        "tool_prefix": "salesforce",
        "statuses": {401, 403},
        "message_re": re.compile(r"(unauthorized|session.expired|invalid.session|token)", re.I),
        "build": lambda ctx: ErrorDiagnosis(
            code="salesforce_token_expired",
            title="Salesforce OAuth token expired or invalid",
            plain_english=(
                f"Salesforce rejected '{ctx['tool_name']}' with HTTP {ctx['http_status']}. "
                "Salesforce OAuth tokens expire after 24 hours by default."
            ),
            root_cause=(
                "The OAuth token was likely issued more than 24 hours ago and has expired. "
                "Salesforce does not auto-refresh tokens — the client must handle renewal."
            ),
            fix_hint=(
                "Reconnect your Salesforce account in Settings → Integrations to get a fresh token. "
                "In Phase 2, Vorlo will auto-refresh tokens 5 minutes before expiry."
            ),
            severity="critical",
            docs_url="https://vorlo.dev/errors/salesforce_token_expired",
        ),
    },
    {
        "tool_prefix": "gmail",
        "statuses": {401, 403},
        "message_re": re.compile(r"(invalid.credentials|token.expired|insufficient.permission|forbidden)", re.I),
        "build": lambda ctx: ErrorDiagnosis(
            code="gmail_auth_error",
            title="Gmail authentication or permission error",
            plain_english=(
                f"Gmail rejected '{ctx['tool_name']}' with HTTP {ctx['http_status']}. "
                "The OAuth token may be expired or the required Gmail API scope was not granted."
            ),
            root_cause=(
                "Google OAuth tokens expire after 1 hour. If the token was not refreshed, "
                "any Gmail API call will fail with a 401. Alternatively, the OAuth consent "
                "screen may not have included the required scope for this operation."
            ),
            fix_hint=(
                "Re-authorize your Google account and ensure the required Gmail scopes are granted. "
                "Check that the OAuth consent screen includes the scope needed for this tool."
            ),
            severity="critical",
            docs_url="https://vorlo.dev/errors/gmail_auth_error",
        ),
    },
    # ── Generic auth for any tool ────────────────────────────────────────
    {
        "tool_prefix": None,  # matches any tool
        "statuses": {401},
        "message_re": re.compile(r".*", re.I),
        "build": lambda ctx: ErrorDiagnosis(
            code="auth_unauthorized",
            title="Authentication failed",
            plain_english=(
                f"The tool '{ctx['tool_name']}' returned HTTP 401 Unauthorized. "
                "The API key or OAuth token used for this request is invalid or expired."
            ),
            root_cause=(
                "The credentials provided to this tool were rejected by the upstream service. "
                "This typically happens when tokens expire, keys are rotated, or the wrong "
                "environment (test vs production) is used."
            ),
            fix_hint=(
                "Verify the API key or OAuth token for this tool is current and valid. "
                "Check if the token has expired and needs to be refreshed or re-issued."
            ),
            severity="critical",
            docs_url="https://vorlo.dev/errors/auth_unauthorized",
        ),
    },
    {
        "tool_prefix": None,
        "statuses": {403},
        "message_re": re.compile(r".*", re.I),
        "build": lambda ctx: ErrorDiagnosis(
            code="auth_forbidden",
            title="Permission denied",
            plain_english=(
                f"The tool '{ctx['tool_name']}' returned HTTP 403 Forbidden. "
                "The credentials are valid but lack the required permissions for this operation."
            ),
            root_cause=(
                "The API key or OAuth token has been accepted by the service but does not have "
                "the necessary scope or role to perform this specific action. This is a permissions "
                "issue, not an authentication issue."
            ),
            fix_hint=(
                "Grant the required scope or permission for this tool. Check the tool's "
                "API documentation for the exact scope needed for this operation."
            ),
            severity="critical",
            docs_url="https://vorlo.dev/errors/auth_forbidden",
        ),
    },
    # ── Rate limiting ────────────────────────────────────────────────────
    {
        "tool_prefix": "gmail",
        "statuses": {429},
        "message_re": re.compile(r".*", re.I),
        "build": lambda ctx: ErrorDiagnosis(
            code="gmail_rate_limit_exceeded",
            title="Gmail API rate limit exceeded",
            plain_english=(
                f"Gmail rate-limited '{ctx['tool_name']}'. "
                "Gmail allows approximately 25 requests per minute per user."
            ),
            root_cause=_rate_limit_root_cause(ctx, "Gmail", "25 requests per minute per user"),
            fix_hint=(
                "Add a 2-3 second delay between consecutive Gmail API calls. "
                "If the agent is in a retry loop, add exponential backoff. "
                "Check if a previous step triggered excessive retries."
            ),
            severity="warning",
            docs_url="https://vorlo.dev/errors/gmail_rate_limit",
        ),
    },
    {
        "tool_prefix": None,
        "statuses": {429},
        "message_re": re.compile(r".*", re.I),
        "build": lambda ctx: ErrorDiagnosis(
            code="rate_limit_exceeded",
            title="Rate limit exceeded",
            plain_english=(
                f"The tool '{ctx['tool_name']}' returned HTTP 429 Too Many Requests. "
                "The agent is sending requests faster than the upstream service allows."
            ),
            root_cause=_rate_limit_root_cause(ctx, ctx["tool_name"], "the service's limit"),
            fix_hint=(
                "Add a delay between consecutive calls to this tool. If the agent "
                "is in a retry loop, implement exponential backoff. Check if a "
                "previous step triggered excessive retries."
            ),
            severity="warning",
            docs_url="https://vorlo.dev/errors/rate_limit_exceeded",
        ),
    },
    # ── Not found ────────────────────────────────────────────────────────
    {
        "tool_prefix": None,
        "statuses": {404},
        "message_re": re.compile(r".*", re.I),
        "build": lambda ctx: ErrorDiagnosis(
            code="resource_not_found",
            title="Resource not found",
            plain_english=(
                f"The tool '{ctx['tool_name']}' returned HTTP 404. "
                "The resource the agent tried to access does not exist."
            ),
            root_cause=_not_found_root_cause(ctx),
            fix_hint=(
                "Check that the resource ID or identifier passed to this tool is correct. "
                "Look at the previous step's output — the ID may have been formatted incorrectly "
                "or the resource may have been deleted."
            ),
            severity="critical",
            docs_url="https://vorlo.dev/errors/resource_not_found",
        ),
    },
    # ── Server errors ────────────────────────────────────────────────────
    {
        "tool_prefix": None,
        "statuses": {500, 502, 503},
        "message_re": re.compile(r".*", re.I),
        "build": lambda ctx: ErrorDiagnosis(
            code="upstream_server_error",
            title="Upstream service error",
            plain_english=(
                f"The tool '{ctx['tool_name']}' returned HTTP {ctx['http_status']}. "
                "The upstream service is experiencing an internal error or outage."
            ),
            root_cause=(
                "This is a server-side error from the upstream service — not caused by the agent "
                "or its configuration. The service may be temporarily degraded or experiencing an outage."
            ),
            fix_hint=(
                "Retry the operation after a short delay. If the error persists, check "
                "the upstream service's status page for known outages."
            ),
            severity="warning",
            docs_url="https://vorlo.dev/errors/upstream_server_error",
        ),
    },
]

# ── Exception-based patterns (no HTTP status) ───────────────────────────

_EXCEPTION_PATTERNS: list[dict[str, Any]] = [
    {
        "error_types": {"TimeoutError", "ConnectTimeout", "ReadTimeout", "Timeout", "timeout"},
        "build": lambda ctx: ErrorDiagnosis(
            code="connection_timeout",
            title="Connection timed out",
            plain_english=(
                f"The tool '{ctx['tool_name']}' timed out — the upstream service "
                "did not respond within the expected time window."
            ),
            root_cause=(
                "The upstream service took too long to respond. This may indicate "
                "the service is degraded, overloaded, or the network path has high latency."
            ),
            fix_hint=(
                "Check if the upstream service is healthy on its status page. "
                "If the tool is performing a large query, consider breaking it into smaller requests. "
                "Increase the timeout if the operation is expected to be slow."
            ),
            severity="warning",
            docs_url="https://vorlo.dev/errors/connection_timeout",
        ),
    },
    {
        "error_types": {"ConnectionError", "ConnectionRefusedError", "ConnectionResetError"},
        "build": lambda ctx: ErrorDiagnosis(
            code="connection_refused",
            title="Connection refused",
            plain_english=(
                f"Could not connect to the service behind '{ctx['tool_name']}'. "
                "The connection was refused or reset."
            ),
            root_cause=(
                "The upstream service is either down, unreachable, or actively refusing connections. "
                "This could also indicate a firewall or DNS resolution issue."
            ),
            fix_hint=(
                "Verify that the service URL is correct. Check the service's status page. "
                "If this is a self-hosted service, ensure it is running and accessible."
            ),
            severity="critical",
            docs_url="https://vorlo.dev/errors/connection_refused",
        ),
    },
    {
        "error_types": {"KeyError"},
        "build": lambda ctx: ErrorDiagnosis(
            code="data_transformation_error",
            title="Missing expected data field",
            plain_english=(
                f"The tool '{ctx['tool_name']}' encountered a KeyError — a required data "
                "field was missing from the input or a previous step's output."
            ),
            root_cause=_key_error_root_cause(ctx),
            fix_hint=(
                "Check the output of the previous step — the field name or data structure "
                "may have changed. Ensure the data passed between steps is in the expected format."
            ),
            severity="critical",
            docs_url="https://vorlo.dev/errors/data_transformation_error",
        ),
    },
    {
        "error_types": {"TypeError", "AttributeError"},
        "build": lambda ctx: ErrorDiagnosis(
            code="data_type_error",
            title="Unexpected data type or missing attribute",
            plain_english=(
                f"The tool '{ctx['tool_name']}' received data in an unexpected format. "
                "A value was None when an object was expected, or the wrong type was passed."
            ),
            root_cause=_type_error_root_cause(ctx),
            fix_hint=(
                "Check that the previous step returned data in the expected format. "
                "A common cause is a previous step returning None (no data) when "
                "a valid object was expected."
            ),
            severity="critical",
            docs_url="https://vorlo.dev/errors/data_type_error",
        ),
    },
]


# ---------------------------------------------------------------------------
# Helper functions for building context-aware root causes
# ---------------------------------------------------------------------------

def _auth_root_cause(ctx: dict[str, Any]) -> str:
    """Build an auth-specific root cause that includes cross-step context."""
    parts = [
        f"The credentials used for '{ctx['tool_name']}' were rejected by the upstream service."
    ]
    if ctx.get("previous_steps"):
        prev_tools = [s.get("tool_name", "unknown") for s in ctx["previous_steps"]]
        parts.append(
            f"Previous steps in this session called: {', '.join(prev_tools)}. "
            "If any of those steps modified or refreshed auth state, the token "
            "may not have propagated to this step."
        )
    return " ".join(parts)


def _rate_limit_root_cause(ctx: dict[str, Any], service: str, limit: str) -> str:
    """Build a rate-limit root cause with context about previous calls."""
    parts = [f"The agent exceeded {service}'s rate limit of {limit}."]
    if ctx.get("previous_steps"):
        same_tool_count = sum(
            1 for s in ctx["previous_steps"]
            if s.get("tool_name", "").startswith(ctx.get("tool_prefix", ""))
        )
        if same_tool_count > 0:
            parts.append(
                f"In the last {len(ctx['previous_steps'])} steps, {same_tool_count} "
                f"also called {service} tools — this burst likely triggered the limit."
            )
    return " ".join(parts)


def _not_found_root_cause(ctx: dict[str, Any]) -> str:
    """Build a 404 root cause that looks at data flow from previous steps."""
    parts = [
        f"The resource requested by '{ctx['tool_name']}' does not exist on the upstream service."
    ]
    if ctx.get("previous_steps"):
        parts.append(
            "This may be caused by an incorrect ID or identifier passed from a previous step. "
            "Check the output of earlier steps — the ID format may differ from what this tool expects."
        )
    return " ".join(parts)


def _key_error_root_cause(ctx: dict[str, Any]) -> str:
    """Build a KeyError root cause with cross-step data flow analysis."""
    parts = [
        f"A required field was missing when '{ctx['tool_name']}' tried to access it."
    ]
    raw = ctx.get("raw_message", "")
    # Try to extract the missing key name from the error message
    key_match = re.search(r"['\"](\w+)['\"]", raw)
    if key_match:
        missing_key = key_match.group(1)
        parts.append(f"The missing field is '{missing_key}'.")
        if ctx.get("previous_steps"):
            parts.append(
                f"Check the output of previous steps — the field '{missing_key}' may have been "
                "renamed, nested differently, or omitted from the response."
            )
    return " ".join(parts)


def _type_error_root_cause(ctx: dict[str, Any]) -> str:
    """Build a TypeError root cause with context."""
    parts = [
        f"'{ctx['tool_name']}' received data in an unexpected type or format."
    ]
    raw = ctx.get("raw_message", "")
    if "NoneType" in raw:
        parts.append(
            "A value was None when a valid object was expected. "
            "This often happens when a previous step returned no data (e.g., a search "
            "returned no results) and the agent passed None to the next step."
        )
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def translate_error(
    tool_name: str,
    error_type: str,
    http_status: Optional[int],
    raw_message: str,
    previous_steps: Optional[list[dict[str, Any]]] = None,
) -> ErrorDiagnosis:
    """
    Translate a raw error into a structured, human-readable diagnosis.

    Args:
        tool_name: Name of the tool that failed (e.g., "stripe_charge_card").
        error_type: Exception class name (e.g., "HTTPError", "KeyError").
        http_status: HTTP status code if available, None for non-HTTP errors.
        raw_message: The raw error message string.
        previous_steps: List of previous step summaries for cross-step context.

    Returns:
        ErrorDiagnosis with code, title, explanation, root cause, and fix hint.
    """
    if previous_steps is None:
        previous_steps = []

    tool_lower = tool_name.lower()
    # Derive a tool prefix for pattern matching (e.g., "stripe" from "stripe_charge_card")
    tool_prefix = tool_lower.split("_")[0] if "_" in tool_lower else tool_lower

    ctx: dict[str, Any] = {
        "tool_name": tool_name,
        "tool_prefix": tool_prefix,
        "error_type": error_type,
        "http_status": http_status,
        "raw_message": raw_message,
        "previous_steps": previous_steps,
    }

    # Try HTTP-status-based patterns first (more specific patterns first)
    if http_status is not None:
        for pattern in _PATTERNS:
            prefix = pattern["tool_prefix"]
            if prefix is not None and not tool_lower.startswith(prefix):
                continue
            if http_status not in pattern["statuses"]:
                continue
            if not pattern["message_re"].search(raw_message):
                continue
            return pattern["build"](ctx)

    # Try exception-type-based patterns
    for pattern in _EXCEPTION_PATTERNS:
        if error_type in pattern["error_types"]:
            return pattern["build"](ctx)

    # Fallback — generic but still helpful diagnosis
    return _generic_diagnosis(ctx)


def _generic_diagnosis(ctx: dict[str, Any]) -> ErrorDiagnosis:
    """Produce a generic diagnosis when no specific pattern matches."""
    status_part = ""
    if ctx["http_status"]:
        status_part = f" (HTTP {ctx['http_status']})"

    return ErrorDiagnosis(
        code="unknown_error",
        title=f"Tool '{ctx['tool_name']}' failed{status_part}",
        plain_english=(
            f"The tool '{ctx['tool_name']}' encountered an error: "
            f"{truncate_keep_tail(ctx['raw_message'], 200)}. "
            "Vorlo could not match this to a known error pattern."
        ),
        root_cause=(
            f"Error type: {ctx['error_type']}. "
            f"Raw message: {truncate_keep_tail(ctx['raw_message'], 300)}. "
            "This error does not match any known pattern in the Vorlo error catalog. "
            "If you see this frequently, report it so we can add a specific diagnosis."
        ),
        fix_hint=(
            "Review the raw error message above. Check the tool's documentation for "
            "this specific error. If this is a recurring issue, contact support and "
            "we will add a specific diagnosis for this error pattern."
        ),
        severity="warning",
        docs_url="https://vorlo.dev/errors/unknown_error",
    )


def _truncate(text: str, max_length: int) -> str:
    """Truncate text to max_length, adding ellipsis if truncated."""
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def truncate_keep_tail(text: str, max_length: int) -> str:
    """
    Truncate keeping both head and tail. Python tracebacks put the actual
    exception on the LAST line, so head-only truncation would show the
    'Traceback (most recent call last)' boilerplate and cut the error itself.
    """
    if len(text) <= max_length:
        return text
    head = max_length // 3
    tail = max_length - head - 5
    return f"{text[:head]} ... {text[-tail:]}"

"""
Vorlo Handler — LangChain callback handler that captures every step
of an agent's execution for debugging and observability.

This is the core of the vorlo-trace SDK. It extends LangChain's
BaseCallbackHandler to intercept tool calls, LLM reasoning, and
chain lifecycle events, then ships them to the Vorlo server for
step replay.

Design principles:
- NEVER crash or slow the agent — all sends are fire-and-forget
- Capture cross-step context for root cause analysis
- Classify every tool as sensor (read) or actuator (write)
- Include OTel-compatible trace/span IDs
"""
from __future__ import annotations

import re
import time
import uuid
from typing import Any, Optional, Union

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult

from vorlo_trace.error_translator import ErrorDiagnosis, translate_error
from vorlo_trace.sender import AsyncSender
from vorlo_trace.session import VorloSession

# ── Tool type classification ────────────────────────────────────────────
_SENSOR_PREFIXES = ("get_", "read_", "fetch_", "search_", "list_", "find_", "query_", "lookup_", "check_")
_ACTUATOR_PREFIXES = ("send_", "create_", "update_", "delete_", "charge_", "post_", "write_", "set_", "remove_", "put_")

_MAX_INPUT_CHARS = 2000
_MAX_OUTPUT_CHARS = 2000
_MAX_REASONING_CHARS = 3000


def _classify_tool(tool_name: str) -> str:
    """Classify a tool as 'sensor' (read) or 'actuator' (write) based on name prefix."""
    lower = tool_name.lower()
    if any(lower.startswith(p) for p in _SENSOR_PREFIXES):
        return "sensor"
    if any(lower.startswith(p) for p in _ACTUATOR_PREFIXES):
        return "actuator"
    # Default to actuator for safety — unknown tools are treated as potentially state-changing
    return "actuator"


def _truncate(text: str, max_length: int) -> str:
    """Truncate text to max_length, preserving meaningful content."""
    text = str(text)
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def _safe_str(value: Any) -> str:
    """Safely convert any value to string without raising."""
    try:
        return str(value)
    except Exception:
        return "<unserializable>"


class VorloHandler(BaseCallbackHandler):
    """
    LangChain callback handler that captures agent execution steps
    and ships them to the Vorlo server for step replay debugging.

    Usage:
        handler = VorloHandler(server_url="https://...", api_key="vrlo_...")
        agent.invoke({"input": "..."}, config={"callbacks": [handler]})
    """

    # LangChain handler config — capture all event types
    raise_error = False  # never raise — we swallow all errors

    def __init__(
        self,
        server_url: str,
        api_key: str,
        agent_name: str = "default",
        verify_ssl: bool = True,
    ) -> None:
        super().__init__()
        self._sender = AsyncSender(
            server_url=server_url,
            api_key=api_key,
            verify_ssl=verify_ssl,
        )
        self._session = VorloSession(agent_name=agent_name)
        self._api_key = api_key

        # Per-tool-call state (keyed by run_id to handle concurrent calls)
        self._active_steps: dict[str, dict[str, Any]] = {}

        # OTel span registry — maps every run_id (chain, llm, tool) to a span_id
        # so a child run can resolve its parent_span_id from parent_run_id.
        # This is what links nested-agent and sub-chain steps into one trace tree.
        self._span_by_run: dict[str, str] = {}

    # ── Span registry helpers ────────────────────────────────────────────

    # Hard cap so a stream of un-paired start events can never leak memory.
    _MAX_TRACKED_SPANS = 5000

    def _register_span(self, run_id: uuid.UUID) -> str:
        """Get or create the span ID for a run. Idempotent per run_id."""
        key = str(run_id)
        span_id = self._span_by_run.get(key)
        if span_id is None:
            if len(self._span_by_run) >= self._MAX_TRACKED_SPANS:
                self._span_by_run.clear()
            span_id = self._session.generate_span_id()
            self._span_by_run[key] = span_id
        return span_id

    def _resolve_parent_span(self, parent_run_id: Optional[uuid.UUID]) -> str:
        """Return the span ID of the parent run, or '' if it is the root."""
        if parent_run_id is None:
            return ""
        return self._span_by_run.get(str(parent_run_id), "")

    def _release_span(self, run_id: uuid.UUID) -> None:
        """Drop a run's span once it has ended. Safe if already gone."""
        self._span_by_run.pop(str(run_id), None)

    # ── Properties ───────────────────────────────────────────────────────

    @property
    def session_id(self) -> str:
        return self._session.session_id

    @property
    def agent_name(self) -> str:
        return self._session.agent_name

    # ── Tool callbacks ───────────────────────────────────────────────────

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        inputs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Called when a tool starts executing."""
        try:
            tool_name = serialized.get("name", "") or serialized.get("id", ["unknown"])[-1]
            step_number = self._session.next_step()
            tool_type = _classify_tool(tool_name)
            span_id = self._register_span(run_id)
            parent_span_id = self._resolve_parent_span(parent_run_id)
            reasoning = self._session.consume_reasoning()

            self._active_steps[str(run_id)] = {
                "step_number": step_number,
                "tool_name": tool_name,
                "tool_type": tool_type,
                "input": _truncate(input_str, _MAX_INPUT_CHARS),
                "start_time": time.time(),
                "span_id": span_id,
                "parent_span_id": parent_span_id,
                "reasoning": _truncate(reasoning, _MAX_REASONING_CHARS) if reasoning else "",
                # Tokens spent by the LLM call(s) that decided this tool call
                "cost_tokens": self._session.consume_tokens(),
            }
        except Exception:
            pass  # never affect the agent

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when a tool completes successfully."""
        try:
            self._release_span(run_id)
            step_data = self._active_steps.pop(str(run_id), None)
            if step_data is None:
                return

            output_str = _truncate(_safe_str(output), _MAX_OUTPUT_CHARS)
            latency_ms = int((time.time() - step_data["start_time"]) * 1000)

            # Build event BEFORE adding to previous steps — context should
            # only include steps that came before this one
            event = self._build_event(
                step_data=step_data,
                status="success",
                output=output_str,
                latency_ms=latency_ms,
                error="",
                error_diagnosis=None,
            )
            self._sender.send(event)

            # Record step for cross-step context of future steps
            self._session.add_completed_step(
                step_number=step_data["step_number"],
                tool_name=step_data["tool_name"],
                tool_type=step_data["tool_type"],
                status="success",
                output_preview=output_str[:200],
            )
        except Exception:
            pass

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when a tool raises an error."""
        try:
            self._release_span(run_id)
            step_data = self._active_steps.pop(str(run_id), None)
            if step_data is None:
                return

            latency_ms = int((time.time() - step_data["start_time"]) * 1000)
            error_str = _safe_str(error)

            # Extract HTTP status if present in the error message
            http_status = _extract_http_status(error_str)
            error_type = type(error).__name__

            # Translate the error into a human-readable diagnosis
            diagnosis = translate_error(
                tool_name=step_data["tool_name"],
                error_type=error_type,
                http_status=http_status,
                raw_message=error_str,
                previous_steps=self._session.get_previous_steps(),
            )

            # Build event BEFORE adding to previous steps — context should
            # only include steps that came before this one
            event = self._build_event(
                step_data=step_data,
                status="failed",
                output="",
                latency_ms=latency_ms,
                error=error_str,
                error_diagnosis=diagnosis,
            )
            self._sender.send(event)

            # Record step for cross-step context of future steps
            self._session.add_completed_step(
                step_number=step_data["step_number"],
                tool_name=step_data["tool_name"],
                tool_type=step_data["tool_type"],
                status="failed",
                output_preview=error_str[:200],
                error_code=diagnosis.code,
            )
        except Exception:
            pass

    # ── LLM callbacks ────────────────────────────────────────────────────

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when the LLM starts generating. Captures the prompt as reasoning input."""
        try:
            self._register_span(run_id)
            # Store the prompt — this becomes the "why" in step replay
            combined = "\n".join(prompts) if prompts else ""
            self._session.set_reasoning(_truncate(combined, _MAX_REASONING_CHARS))
        except Exception:
            pass

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when the LLM finishes generating. Captures the decision output."""
        try:
            self._release_span(run_id)
            self._session.add_tokens(_extract_total_tokens(response))
            if response.generations:
                # Get the text of the first generation — this contains the agent's decision
                first_gen = response.generations[0]
                if first_gen:
                    text = first_gen[0].text if first_gen[0].text else ""
                    # Append LLM output to reasoning so we capture both input and decision
                    current = self._session.consume_reasoning() or ""
                    combined = f"{current}\n---LLM OUTPUT---\n{text}" if current else text
                    self._session.set_reasoning(_truncate(combined, _MAX_REASONING_CHARS))
        except Exception:
            pass

    # ── Agent callbacks ──────────────────────────────────────────────────

    def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when the agent decides which tool to call."""
        try:
            reasoning_parts = []
            current = self._session.consume_reasoning()
            if current:
                reasoning_parts.append(current)
            reasoning_parts.append(
                f"Agent decided to call tool '{action.tool}' with input: "
                f"{_truncate(_safe_str(action.tool_input), 500)}"
            )
            if action.log:
                reasoning_parts.append(f"Agent reasoning: {_truncate(action.log, 1000)}")
            self._session.set_reasoning(
                _truncate("\n".join(reasoning_parts), _MAX_REASONING_CHARS)
            )
        except Exception:
            pass

    def on_agent_finish(
        self,
        finish: AgentFinish,
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when the agent completes its final answer."""
        try:
            event = {
                "event_type": "session_complete",
                "session_id": self._session.session_id,
                "trace_id": self._session.trace_id,
                "agent_name": self._session.agent_name,
                "api_key": self._api_key,
                "total_steps": self._session.step_count,
                "duration_ms": self._session.duration_ms,
                "status": "success",
                "output": _truncate(_safe_str(finish.return_values), _MAX_OUTPUT_CHARS),
            }
            self._sender.send(event)
        except Exception:
            pass

    # ── Chain callbacks ──────────────────────────────────────────────────

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when a chain starts. Marks session start for top-level chains."""
        try:
            self._register_span(run_id)
            if parent_run_id is None:
                # Top-level chain — this is the session start
                event = {
                    "event_type": "session_start",
                    "session_id": self._session.session_id,
                    "trace_id": self._session.trace_id,
                    "agent_name": self._session.agent_name,
                    "api_key": self._api_key,
                    "input": _truncate(_safe_str(inputs), _MAX_INPUT_CHARS),
                }
                self._sender.send(event)
        except Exception:
            pass

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when a chain ends."""
        try:
            self._release_span(run_id)
            if parent_run_id is None:
                event = {
                    "event_type": "session_complete",
                    "session_id": self._session.session_id,
                    "trace_id": self._session.trace_id,
                    "agent_name": self._session.agent_name,
                    "api_key": self._api_key,
                    "total_steps": self._session.step_count,
                    "duration_ms": self._session.duration_ms,
                    "status": "success",
                    "output": _truncate(_safe_str(outputs), _MAX_OUTPUT_CHARS),
                }
                self._sender.send(event)
        except Exception:
            pass

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when a chain errors. Sends session_error for top-level chains."""
        try:
            self._release_span(run_id)
            if parent_run_id is None:
                event = {
                    "event_type": "session_error",
                    "session_id": self._session.session_id,
                    "trace_id": self._session.trace_id,
                    "agent_name": self._session.agent_name,
                    "api_key": self._api_key,
                    "total_steps": self._session.step_count,
                    "duration_ms": self._session.duration_ms,
                    "status": "failed",
                    "error": _truncate(_safe_str(error), _MAX_OUTPUT_CHARS),
                }
                self._sender.send(event)
        except Exception:
            pass

    # ── Chat model callbacks ─────────────────────────────────────────────

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Called when a chat model starts. Captures messages as reasoning."""
        try:
            self._register_span(run_id)
            all_messages = []
            for msg_list in messages:
                for msg in msg_list:
                    all_messages.append(f"[{msg.type}] {msg.content}")
            combined = "\n".join(all_messages)
            self._session.set_reasoning(_truncate(combined, _MAX_REASONING_CHARS))
        except Exception:
            pass

    # ── Internal helpers ─────────────────────────────────────────────────

    def _build_event(
        self,
        step_data: dict[str, Any],
        status: str,
        output: str,
        latency_ms: int,
        error: str,
        error_diagnosis: Optional[ErrorDiagnosis],
    ) -> dict[str, Any]:
        """Build a trace event dict ready for sending to the server."""
        event: dict[str, Any] = {
            "event_type": "step",
            "session_id": self._session.session_id,
            "trace_id": self._session.trace_id,
            "span_id": step_data["span_id"],
            "parent_span_id": step_data.get("parent_span_id", ""),
            "agent_name": self._session.agent_name,
            "api_key": self._api_key,
            "step_number": step_data["step_number"],
            "tool_name": step_data["tool_name"],
            "tool_type": step_data["tool_type"],
            "input": step_data["input"],
            "output": output,
            "status": status,
            "latency_ms": latency_ms,
            "cost_tokens": step_data.get("cost_tokens", 0),
            "reasoning": step_data.get("reasoning", ""),
            "previous_step_context": self._session.get_previous_steps(),
        }

        if error_diagnosis:
            event["error"] = error_diagnosis.plain_english
            event["error_diagnosis"] = {
                "code": error_diagnosis.code,
                "title": error_diagnosis.title,
                "plain_english": error_diagnosis.plain_english,
                "root_cause": error_diagnosis.root_cause,
                "fix_hint": error_diagnosis.fix_hint,
                "severity": error_diagnosis.severity,
                "docs_url": error_diagnosis.docs_url,
            }
        else:
            event["error"] = error

        return event


# ── Utility ──────────────────────────────────────────────────────────────

def _extract_total_tokens(response: LLMResult) -> int:
    """
    Extract total token usage from an LLMResult, across provider shapes:
    OpenAI-style llm_output["token_usage"], Anthropic-style llm_output["usage"],
    and per-message usage_metadata (newer LangChain chat models).
    Returns 0 when usage is unavailable — never raises.
    """
    try:
        llm_output = getattr(response, "llm_output", None) or {}
        usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
        if usage:
            total = usage.get("total_tokens")
            if total is None:
                total = (
                    (usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
                    + (usage.get("completion_tokens") or usage.get("output_tokens") or 0)
                )
            if total:
                return int(total)

        for gen_list in getattr(response, "generations", None) or []:
            for gen in gen_list:
                meta = getattr(getattr(gen, "message", None), "usage_metadata", None)
                if meta:
                    total = meta.get("total_tokens") or (
                        (meta.get("input_tokens") or 0) + (meta.get("output_tokens") or 0)
                    )
                    if total:
                        return int(total)
    except Exception:
        pass
    return 0


# A bare \b([1-5]\d{2})\b would treat ANY 3-digit number as an HTTP status
# ("KeyError at line 403 of utils.py" → diagnosed as 403 Forbidden), and a
# wrong diagnosis is worse than none. Only extract a code when it appears in
# an HTTP-shaped context.
_HTTP_STATUS_PATTERNS = [
    # "HTTP 403", "HTTP/1.1 403", "HTTPS 503", "http status: 403"
    re.compile(r"\bhttps?(?:/\d\.\d)?\s*(?:status)?\s*[:=]?\s*([1-5]\d{2})\b", re.I),
    # "status code 403", "status: 403", "status_code=403", "StatusCode: 429"
    re.compile(r"\bstatus(?:[ _-]?code)?\s*[:=]?\s*([1-5]\d{2})\b", re.I),
    # "error code: 429", "code=503"
    re.compile(r"\b(?:error\s+)?code\s*[:=]\s*([1-5]\d{2})\b", re.I),
    # "returned 503", "responded with 502", "got a 404", "received 429"
    re.compile(r"\b(?:returned|respond(?:ed)?\s+with|got(?:\s+a)?|received)\s+([1-5]\d{2})\b", re.I),
    # "SomeError: 403", "ToolError(429)" — a code immediately after an error label
    re.compile(r"\b\w*(?:error|exception)\w*\s*[:(]\s*([1-5]\d{2})\b", re.I),
    # "rate limited: 429", "rate limit (429)"
    re.compile(r"\brate[ -]?limit\w*\s*[:=(]?\s*([1-5]\d{2})\b", re.I),
    # "403 Forbidden", "429 Too Many Requests" — a code paired with ITS OWN
    # canonical reason phrase ("503 not found in table" must not match).
    re.compile(
        r"\b(?:"
        r"(400)\s+bad request|(401)\s+unauthorized|(402)\s+payment required|"
        r"(403)\s+forbidden|(404)\s+not found|(405)\s+method not allowed|"
        r"(406)\s+not acceptable|(408)\s+request timeout|(409)\s+conflict|"
        r"(410)\s+gone|(412)\s+precondition failed|(413)\s+payload too large|"
        r"(422)\s+unprocessable|(429)\s+too many requests|"
        r"(500)\s+internal server error|(501)\s+not implemented|"
        r"(502)\s+bad gateway|(503)\s+service unavailable|(504)\s+gateway timeout"
        r")\b",
        re.I,
    ),
]


def _extract_http_status(error_message: str) -> Optional[int]:
    """Extract an HTTP status code from an error message, requiring HTTP context."""
    for pattern in _HTTP_STATUS_PATTERNS:
        match = pattern.search(error_message)
        if match:
            # The paired code+reason pattern has many groups; take the one that hit.
            code_str = next((g for g in match.groups() if g), None)
            if code_str is None:
                continue
            code = int(code_str)
            if 100 <= code <= 599:
                return code
    return None
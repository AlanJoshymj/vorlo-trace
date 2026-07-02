"""
Vorlo × OpenAI Agents SDK — trace-processor adapter.

Two lines, same as LangChain:

    import vorlo_trace
    vorlo_trace.init(api_key="vrlo_...", agent_name="my-agent")
    vorlo_trace.instrument_openai_agents()

Every Runner.run() becomes a Vorlo session: function-tool calls become
numbered steps (with reasoning and token cost pulled from the surrounding
model spans), failures go through the same error-translation registry as
the LangChain handler, and handoffs/guardrails are captured so multi-agent
runs replay faithfully.

Design principles (identical to the LangChain handler):
- NEVER crash or slow the agent — every hook body is wrapped, all sends
  are fire-and-forget through AsyncSender
- Duck-typed access to SDK objects, so minor SDK shape changes degrade
  gracefully instead of raising
"""
from __future__ import annotations

import threading
from datetime import datetime
from typing import Any, Callable, Optional

from vorlo_trace.error_translator import translate_error, truncate_keep_tail
from vorlo_trace.handler import (
    _classify_tool,
    _extract_http_status,
    _safe_str,
    _truncate,
    _MAX_INPUT_CHARS,
    _MAX_OUTPUT_CHARS,
    _MAX_REASONING_CHARS,
)
from vorlo_trace.sender import AsyncSender
from vorlo_trace.session import VorloSession

# Cap concurrent live traces so un-ended traces can never leak memory.
_MAX_LIVE_TRACES = 1000


def _parse_iso(ts: Any) -> Optional[datetime]:
    """Parse the SDK's started_at/ended_at (ISO string or datetime). Never raises."""
    try:
        if isinstance(ts, datetime):
            return ts
        if isinstance(ts, str) and ts:
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        pass
    return None


def _span_latency_ms(span: Any) -> int:
    start = _parse_iso(getattr(span, "started_at", None))
    end = _parse_iso(getattr(span, "ended_at", None))
    if start and end:
        try:
            return max(int((end - start).total_seconds() * 1000), 0)
        except Exception:
            pass
    return 0


def _span_error_message(span: Any) -> str:
    """Extract the error message from span.error (dict or object), '' if none."""
    err = getattr(span, "error", None)
    if not err:
        return ""
    try:
        if isinstance(err, dict):
            message = err.get("message") or ""
            data = err.get("data")
        else:
            message = getattr(err, "message", "") or ""
            data = getattr(err, "data", None)
        if data:
            return f"{message}: {_safe_str(data)}" if message else _safe_str(data)
        return str(message)
    except Exception:
        return "<error unavailable>"


def _extract_usage_tokens(usage: Any) -> int:
    """Total tokens from a usage dict/object across provider naming schemes."""
    if usage is None:
        return 0
    try:
        get = usage.get if isinstance(usage, dict) else lambda k, d=None: getattr(usage, k, d)
        total = get("total_tokens")
        if total:
            return int(total)
        input_t = get("input_tokens") or get("prompt_tokens") or 0
        output_t = get("output_tokens") or get("completion_tokens") or 0
        return int(input_t) + int(output_t)
    except Exception:
        return 0


class VorloTraceProcessor:
    """
    OpenAI Agents SDK tracing processor that ships runs to Vorlo.

    Implements the TracingProcessor interface by shape (on_trace_start,
    on_trace_end, on_span_start, on_span_end, shutdown, force_flush) —
    duck-typed on purpose so importing this module never requires the
    `agents` package.
    """

    def __init__(
        self,
        server_url: str,
        api_key: str,
        agent_name: str = "default",
        verify_ssl: bool = True,
        redact: Optional[Callable[[str], str]] = None,
        sender: Optional[AsyncSender] = None,
    ) -> None:
        self._sender = sender or AsyncSender(
            server_url=server_url, api_key=api_key, verify_ssl=verify_ssl
        )
        self._api_key = api_key
        self._agent_name = agent_name
        self._redact = redact
        # One Vorlo session per OpenAI trace (Runner.run) — concurrent runs safe.
        self._sessions: dict[str, VorloSession] = {}
        # Trace-level error stash: agent-span failures mark the whole run failed.
        self._trace_errors: dict[str, str] = {}
        self._lock = threading.Lock()

    def _scrub(self, text: str) -> str:
        """Same privacy-safe failure mode as the LangChain handler."""
        if not self._redact or not text:
            return text
        try:
            return str(self._redact(text))
        except Exception:
            return "<redacted: redact callback raised>"

    def _get_session(self, trace_id: str) -> Optional[VorloSession]:
        with self._lock:
            return self._sessions.get(trace_id)

    # ── TracingProcessor interface ────────────────────────────────────────

    def on_trace_start(self, trace: Any) -> None:
        try:
            trace_id = str(getattr(trace, "trace_id", "") or "")
            if not trace_id:
                return
            # Workflow name from the SDK when present; fall back to init()'s.
            name = getattr(trace, "name", None) or self._agent_name
            session = VorloSession(agent_name=str(name))
            with self._lock:
                if len(self._sessions) >= _MAX_LIVE_TRACES:
                    self._sessions.clear()
                    self._trace_errors.clear()
                self._sessions[trace_id] = session
            self._sender.send({
                "event_type": "session_start",
                "session_id": session.session_id,
                "trace_id": session.trace_id,
                "agent_name": session.agent_name,
                "api_key": self._api_key,
                "input": "",
            })
        except Exception:
            pass  # never affect the agent

    def on_trace_end(self, trace: Any) -> None:
        try:
            trace_id = str(getattr(trace, "trace_id", "") or "")
            with self._lock:
                session = self._sessions.pop(trace_id, None)
                trace_error = self._trace_errors.pop(trace_id, "")
            if session is None:
                return
            event: dict[str, Any] = {
                "session_id": session.session_id,
                "trace_id": session.trace_id,
                "agent_name": session.agent_name,
                "api_key": self._api_key,
                "total_steps": session.step_count,
                "duration_ms": session.duration_ms,
            }
            if trace_error:
                event["event_type"] = "session_error"
                event["status"] = "failed"
                event["error"] = truncate_keep_tail(
                    self._scrub(trace_error), _MAX_OUTPUT_CHARS
                )
            else:
                event["event_type"] = "session_complete"
                event["status"] = "success"
                event["output"] = ""
            self._sender.send(event)
        except Exception:
            pass

    def on_span_start(self, span: Any) -> None:
        # Span data is only fully populated at end; all work happens there.
        pass

    def on_span_end(self, span: Any) -> None:
        try:
            trace_id = str(getattr(span, "trace_id", "") or "")
            session = self._get_session(trace_id)
            if session is None:
                return
            data = getattr(span, "span_data", None)
            span_type = str(getattr(data, "type", "") or "")

            if span_type == "function":
                self._on_function_span(session, span, data)
            elif span_type in ("generation", "response"):
                self._on_model_span(session, span, data, span_type)
            elif span_type == "handoff":
                self._on_handoff_span(session, span, data)
            elif span_type == "guardrail":
                self._on_guardrail_span(session, span, data)
            elif span_type == "agent":
                error = _span_error_message(span)
                if error:
                    with self._lock:
                        self._trace_errors[trace_id] = error
        except Exception:
            pass

    def shutdown(self) -> None:
        pass  # AsyncSender is a daemon thread — nothing to tear down

    def force_flush(self) -> None:
        pass  # fire-and-forget by design; no sync flush

    # ── Span handlers ─────────────────────────────────────────────────────

    def _on_model_span(
        self, session: VorloSession, span: Any, data: Any, span_type: str
    ) -> None:
        """Model call ended — stash reasoning + token cost for the next tool step."""
        scope = str(getattr(span, "parent_id", "") or "")

        if span_type == "generation":
            usage = getattr(data, "usage", None)
            output = getattr(data, "output", None)
        else:  # "response" — Responses API; data.response is an OpenAI Response
            response = getattr(data, "response", None)
            usage = getattr(response, "usage", None)
            output = getattr(response, "output", None)

        session.add_tokens(_extract_usage_tokens(usage), scope=scope)
        if output is not None:
            session.set_reasoning(
                _truncate(self._scrub(_safe_str(output)), _MAX_REASONING_CHARS),
                scope=scope,
            )

    def _on_function_span(self, session: VorloSession, span: Any, data: Any) -> None:
        """Function tool call ended — this is a Vorlo step."""
        tool_name = str(getattr(data, "name", "") or "unknown_tool")
        input_str = _truncate(
            self._scrub(_safe_str(getattr(data, "input", "") or "")), _MAX_INPUT_CHARS
        )
        error_str = self._scrub(_span_error_message(span))
        status = "failed" if error_str else "success"
        step_number = session.next_step()
        # The model span that decided this call shares the same parent (the
        # agent span), so its reasoning/tokens were stashed under that scope.
        scope = str(getattr(span, "parent_id", "") or "")
        reasoning = session.consume_reasoning(scope=scope) or ""

        event: dict[str, Any] = {
            "event_type": "step",
            "session_id": session.session_id,
            "trace_id": session.trace_id,
            "span_id": str(getattr(span, "span_id", "") or session.generate_span_id()),
            "parent_span_id": str(getattr(span, "parent_id", "") or ""),
            "agent_name": session.agent_name,
            "api_key": self._api_key,
            "step_number": step_number,
            "tool_name": tool_name,
            "tool_type": _classify_tool(tool_name),
            "input": input_str,
            "output": "",
            "status": status,
            "latency_ms": _span_latency_ms(span),
            "cost_tokens": session.consume_tokens(scope=scope),
            "reasoning": _truncate(reasoning, _MAX_REASONING_CHARS),
            "previous_step_context": session.get_previous_steps(),
        }

        error_code = None
        if error_str:
            diagnosis = translate_error(
                tool_name=tool_name,
                error_type="FunctionToolError",
                http_status=_extract_http_status(error_str),
                raw_message=error_str,
                previous_steps=session.get_previous_steps(),
            )
            error_code = diagnosis.code
            event["error"] = diagnosis.plain_english
            event["error_diagnosis"] = {
                "code": diagnosis.code,
                "title": diagnosis.title,
                "plain_english": diagnosis.plain_english,
                "root_cause": diagnosis.root_cause,
                "fix_hint": diagnosis.fix_hint,
                "severity": diagnosis.severity,
                "docs_url": diagnosis.docs_url,
            }
            output_preview = error_str[:200]
        else:
            output_str = _truncate(
                self._scrub(_safe_str(getattr(data, "output", "") or "")),
                _MAX_OUTPUT_CHARS,
            )
            event["output"] = output_str
            event["error"] = ""
            output_preview = output_str[:200]

        self._sender.send(event)
        session.add_completed_step(
            step_number=step_number,
            tool_name=tool_name,
            tool_type=event["tool_type"],
            status=status,
            output_preview=output_preview,
            error_code=error_code,
        )

    def _on_handoff_span(self, session: VorloSession, span: Any, data: Any) -> None:
        """Agent-to-agent handoff — recorded as a step so multi-agent runs replay."""
        from_agent = str(getattr(data, "from_agent", "") or "?")
        to_agent = str(getattr(data, "to_agent", "") or "?")
        step_number = session.next_step()
        error_str = self._scrub(_span_error_message(span))
        status = "failed" if error_str else "success"
        tool_name = f"handoff_to_{to_agent}"

        self._sender.send({
            "event_type": "step",
            "session_id": session.session_id,
            "trace_id": session.trace_id,
            "span_id": str(getattr(span, "span_id", "") or session.generate_span_id()),
            "parent_span_id": str(getattr(span, "parent_id", "") or ""),
            "agent_name": session.agent_name,
            "api_key": self._api_key,
            "step_number": step_number,
            "tool_name": tool_name,
            "tool_type": "actuator",
            "input": f"handoff from '{from_agent}' to '{to_agent}'",
            "output": "" if error_str else f"control passed to '{to_agent}'",
            "status": status,
            "latency_ms": _span_latency_ms(span),
            "cost_tokens": 0,
            "reasoning": "",
            "error": error_str,
            "previous_step_context": session.get_previous_steps(),
        })
        session.add_completed_step(
            step_number=step_number,
            tool_name=tool_name,
            tool_type="actuator",
            status=status,
            output_preview=f"→ {to_agent}",
        )

    def _on_guardrail_span(self, session: VorloSession, span: Any, data: Any) -> None:
        """Guardrail evaluations are only interesting when they trip."""
        if not getattr(data, "triggered", False):
            return
        name = str(getattr(data, "name", "") or "guardrail")
        step_number = session.next_step()
        error_str = f"Guardrail '{name}' triggered and blocked the run."

        self._sender.send({
            "event_type": "step",
            "session_id": session.session_id,
            "trace_id": session.trace_id,
            "span_id": str(getattr(span, "span_id", "") or session.generate_span_id()),
            "parent_span_id": str(getattr(span, "parent_id", "") or ""),
            "agent_name": session.agent_name,
            "api_key": self._api_key,
            "step_number": step_number,
            "tool_name": f"guardrail_{name}",
            "tool_type": "sensor",
            "input": "",
            "output": "",
            "status": "failed",
            "latency_ms": _span_latency_ms(span),
            "cost_tokens": 0,
            "reasoning": "",
            "error": error_str,
            "previous_step_context": session.get_previous_steps(),
        })
        session.add_completed_step(
            step_number=step_number,
            tool_name=f"guardrail_{name}",
            tool_type="sensor",
            status="failed",
            output_preview=error_str[:200],
        )

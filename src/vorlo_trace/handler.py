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
            span_id = self._session.generate_span_id()
            reasoning = self._session.consume_reasoning()

            self._active_steps[str(run_id)] = {
                "step_number": step_number,
                "tool_name": tool_name,
                "tool_type": tool_type,
                "input": _truncate(input_str, _MAX_INPUT_CHARS),
                "start_time": time.time(),
                "span_id": span_id,
                "parent_span_id": "",
                "reasoning": _truncate(reasoning, _MAX_REASONING_CHARS) if reasoning else "",
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

_HTTP_STATUS_RE = re.compile(r"\b([1-5]\d{2})\b")


def _extract_http_status(error_message: str) -> Optional[int]:
    """Try to extract an HTTP status code from an error message string."""
    match = _HTTP_STATUS_RE.search(error_message)
    if match:
        code = int(match.group(1))
        if 100 <= code <= 599:
            return code
    return None
"""Tests for the OpenAI Agents SDK adapter (VorloTraceProcessor).

The processor is duck-typed against the SDK's tracing objects, so these
tests drive it with plain fakes — no `openai-agents` install required.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from vorlo_trace.openai_agents import (
    VorloTraceProcessor,
    _extract_usage_tokens,
    _span_latency_ms,
)


class RecordingSender:
    """Stands in for AsyncSender — records events instead of sending."""

    def __init__(self) -> None:
        self.events: list[dict] = []

    def send(self, event: dict) -> None:
        self.events.append(event)


def make_processor() -> tuple[VorloTraceProcessor, RecordingSender]:
    sender = RecordingSender()
    proc = VorloTraceProcessor(
        server_url="https://example.test",
        api_key="vrlo_test",
        agent_name="test-agent",
        sender=sender,  # type: ignore[arg-type]
    )
    return proc, sender


def fake_trace(trace_id: str = "trace_1", name: str = "wf") -> SimpleNamespace:
    return SimpleNamespace(trace_id=trace_id, name=name)


def fake_span(
    trace_id: str = "trace_1",
    span_id: str = "span_1",
    parent_id: str = "span_agent",
    data: object = None,
    error: object = None,
    started_at: str = "2026-07-02T10:00:00+00:00",
    ended_at: str = "2026-07-02T10:00:01.500+00:00",
) -> SimpleNamespace:
    return SimpleNamespace(
        trace_id=trace_id,
        span_id=span_id,
        parent_id=parent_id,
        span_data=data,
        error=error,
        started_at=started_at,
        ended_at=ended_at,
    )


def test_trace_lifecycle_success():
    proc, sender = make_processor()
    proc.on_trace_start(fake_trace())
    proc.on_trace_end(fake_trace())

    assert [e["event_type"] for e in sender.events] == [
        "session_start",
        "session_complete",
    ]
    assert sender.events[0]["agent_name"] == "wf"
    assert sender.events[1]["status"] == "success"


def test_function_span_becomes_step_with_reasoning_and_tokens():
    proc, sender = make_processor()
    proc.on_trace_start(fake_trace())

    # Model decides (generation span ends first, same parent as the tool)
    gen_data = SimpleNamespace(
        type="generation",
        usage={"input_tokens": 100, "output_tokens": 20},
        output=[{"role": "assistant", "content": "call get_weather"}],
    )
    proc.on_span_end(fake_span(span_id="span_gen", data=gen_data))

    # Tool call succeeds
    fn_data = SimpleNamespace(
        type="function", name="get_weather", input='{"city": "Paris"}', output="22C"
    )
    proc.on_span_end(fake_span(span_id="span_fn", data=fn_data))

    steps = [e for e in sender.events if e["event_type"] == "step"]
    assert len(steps) == 1
    step = steps[0]
    assert step["tool_name"] == "get_weather"
    assert step["tool_type"] == "sensor"  # get_ prefix
    assert step["status"] == "success"
    assert step["step_number"] == 1
    assert step["output"] == "22C"
    assert step["cost_tokens"] == 120
    assert "call get_weather" in step["reasoning"]
    assert step["latency_ms"] == 1500
    assert step["span_id"] == "span_fn"
    assert step["parent_span_id"] == "span_agent"


def test_failed_function_span_carries_diagnosis():
    proc, sender = make_processor()
    proc.on_trace_start(fake_trace())

    fn_data = SimpleNamespace(type="function", name="charge_card", input="{}", output=None)
    err = {"message": "Tool failed: HTTP 403 Forbidden", "data": None}
    proc.on_span_end(fake_span(data=fn_data, error=err))

    steps = [e for e in sender.events if e["event_type"] == "step"]
    assert len(steps) == 1
    step = steps[0]
    assert step["status"] == "failed"
    assert step["tool_type"] == "actuator"  # charge_ prefix
    assert "error_diagnosis" in step
    assert step["error_diagnosis"]["code"]  # registry classified something
    # Failed step lands in cross-step context for the NEXT step
    fn2 = SimpleNamespace(type="function", name="get_status", input="", output="ok")
    proc.on_span_end(fake_span(span_id="span_2", data=fn2))
    step2 = [e for e in sender.events if e["event_type"] == "step"][1]
    assert step2["previous_step_context"][0]["tool_name"] == "charge_card"
    assert step2["previous_step_context"][0]["status"] == "failed"


def test_agent_span_error_marks_session_failed():
    proc, sender = make_processor()
    proc.on_trace_start(fake_trace())

    agent_data = SimpleNamespace(type="agent", name="triage")
    proc.on_span_end(
        fake_span(data=agent_data, error={"message": "Max turns exceeded", "data": None})
    )
    proc.on_trace_end(fake_trace())

    end = sender.events[-1]
    assert end["event_type"] == "session_error"
    assert end["status"] == "failed"
    assert "Max turns exceeded" in end["error"]


def test_handoff_span_recorded_as_step():
    proc, sender = make_processor()
    proc.on_trace_start(fake_trace())

    handoff = SimpleNamespace(type="handoff", from_agent="triage", to_agent="billing")
    proc.on_span_end(fake_span(data=handoff))

    step = [e for e in sender.events if e["event_type"] == "step"][0]
    assert step["tool_name"] == "handoff_to_billing"
    assert step["status"] == "success"
    assert "triage" in step["input"]


def test_triggered_guardrail_recorded_untriggered_skipped():
    proc, sender = make_processor()
    proc.on_trace_start(fake_trace())

    proc.on_span_end(
        fake_span(data=SimpleNamespace(type="guardrail", name="pii", triggered=False))
    )
    proc.on_span_end(
        fake_span(data=SimpleNamespace(type="guardrail", name="pii", triggered=True))
    )

    steps = [e for e in sender.events if e["event_type"] == "step"]
    assert len(steps) == 1
    assert steps[0]["tool_name"] == "guardrail_pii"
    assert steps[0]["status"] == "failed"


def test_spans_for_unknown_trace_are_ignored():
    proc, sender = make_processor()
    # No on_trace_start — span arrives for a trace we never saw
    fn = SimpleNamespace(type="function", name="get_x", input="", output="y")
    proc.on_span_end(fake_span(trace_id="ghost", data=fn))
    assert sender.events == []


def test_redact_applies_before_send():
    sender = RecordingSender()
    proc = VorloTraceProcessor(
        server_url="https://example.test",
        api_key="vrlo_test",
        redact=lambda s: s.replace("secret", "[HIDDEN]"),
        sender=sender,  # type: ignore[arg-type]
    )
    proc.on_trace_start(fake_trace())
    fn = SimpleNamespace(type="function", name="get_data", input="key=secret", output="secret ok")
    proc.on_span_end(fake_span(data=fn))
    step = [e for e in sender.events if e["event_type"] == "step"][0]
    assert "secret" not in step["input"]
    assert "secret" not in step["output"]
    assert "[HIDDEN]" in step["input"]


def test_usage_token_shapes():
    assert _extract_usage_tokens({"total_tokens": 42}) == 42
    assert _extract_usage_tokens({"input_tokens": 10, "output_tokens": 5}) == 15
    assert _extract_usage_tokens({"prompt_tokens": 7, "completion_tokens": 3}) == 10
    assert _extract_usage_tokens(SimpleNamespace(input_tokens=4, output_tokens=6, total_tokens=None)) == 10
    assert _extract_usage_tokens(None) == 0
    assert _extract_usage_tokens("garbage") == 0


def test_latency_parsing_handles_bad_timestamps():
    assert _span_latency_ms(SimpleNamespace(started_at=None, ended_at=None)) == 0
    assert _span_latency_ms(SimpleNamespace(started_at="not-a-date", ended_at="also-bad")) == 0
    span = SimpleNamespace(
        started_at="2026-07-02T10:00:00Z", ended_at="2026-07-02T10:00:02Z"
    )
    assert _span_latency_ms(span) == 2000


def test_processor_never_raises_on_malformed_spans():
    proc, _ = make_processor()
    proc.on_trace_start(fake_trace())
    # Garbage in every slot — the processor must swallow everything
    proc.on_span_end(None)
    proc.on_span_end(SimpleNamespace())  # no span_data at all
    proc.on_span_end(fake_span(data=SimpleNamespace(type="function")))  # no name
    proc.on_trace_end(SimpleNamespace())  # no trace_id
    proc.on_trace_end(fake_trace())  # normal end still works

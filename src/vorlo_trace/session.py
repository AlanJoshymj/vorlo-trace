"""
Vorlo Session — manages state for a single agent execution session.

Tracks session ID, step counter, timing, and a rolling window of
previous step summaries for cross-step root cause analysis.
"""
from __future__ import annotations

import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional


# Maximum number of previous step summaries to retain for cross-step context
_MAX_PREVIOUS_STEPS = 5


@dataclass
class StepSummary:
    """Compact summary of a completed step, used for cross-step context."""

    step_number: int
    tool_name: str
    tool_type: str  # "sensor" | "actuator"
    status: str  # "success" | "failed"
    output_preview: str  # first 200 chars of output
    error_code: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "step_number": self.step_number,
            "tool_name": self.tool_name,
            "tool_type": self.tool_type,
            "status": self.status,
            "output_preview": self.output_preview,
        }
        if self.error_code:
            d["error_code"] = self.error_code
        return d


class VorloSession:
    """
    Manages state for one agent execution session.

    A session starts when the agent begins running and ends when it
    completes (success or failure). Each tool call within the session
    is a numbered step.
    """

    def __init__(self, agent_name: str = "default") -> None:
        self.session_id: str = uuid.uuid4().hex
        self.trace_id: str = uuid.uuid4().hex  # OTel-compatible trace ID
        self.agent_name: str = agent_name
        self.start_time: float = time.time()
        self.step_count: int = 0
        self._previous_steps: deque[StepSummary] = deque(maxlen=_MAX_PREVIOUS_STEPS)
        self._current_reasoning: Optional[str] = None
        self._pending_tokens: int = 0
        self.total_tokens: int = 0

    def next_step(self) -> int:
        """Increment and return the next step number."""
        self.step_count += 1
        return self.step_count

    def generate_span_id(self) -> str:
        """Generate a unique span ID for OTel compatibility."""
        return uuid.uuid4().hex[:16]

    def add_completed_step(
        self,
        step_number: int,
        tool_name: str,
        tool_type: str,
        status: str,
        output_preview: str,
        error_code: Optional[str] = None,
    ) -> None:
        """Record a completed step for cross-step context tracking."""
        summary = StepSummary(
            step_number=step_number,
            tool_name=tool_name,
            tool_type=tool_type,
            status=status,
            output_preview=output_preview[:200],
            error_code=error_code,
        )
        self._previous_steps.append(summary)

    def get_previous_steps(self) -> list[dict[str, Any]]:
        """Return previous step summaries as a list of dicts."""
        return [s.to_dict() for s in self._previous_steps]

    def set_reasoning(self, reasoning: str) -> None:
        """Store the LLM's reasoning/chain-of-thought before a tool call."""
        self._current_reasoning = reasoning

    def consume_reasoning(self) -> Optional[str]:
        """Return and clear the stored reasoning. Called once per tool call."""
        reasoning = self._current_reasoning
        self._current_reasoning = None
        return reasoning

    def add_tokens(self, count: int) -> None:
        """Accumulate token usage from LLM calls since the last tool start."""
        if count > 0:
            self._pending_tokens += count
            self.total_tokens += count

    def consume_tokens(self) -> int:
        """Return and reset the pending token count. Called once per tool call."""
        tokens = self._pending_tokens
        self._pending_tokens = 0
        return tokens

    def get_context(self) -> dict[str, Any]:
        """Return full session context dict for inclusion in trace events."""
        return {
            "session_id": self.session_id,
            "trace_id": self.trace_id,
            "agent_name": self.agent_name,
            "start_time": self.start_time,
            "step_count": self.step_count,
            "previous_steps": self.get_previous_steps(),
        }

    @property
    def duration_ms(self) -> int:
        """Elapsed time since session start in milliseconds."""
        return int((time.time() - self.start_time) * 1000)

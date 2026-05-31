"""
Vorlo Async Sender — fire-and-forget HTTP sender for trace events.

Uses a daemon thread with a queue so the main agent thread is never blocked.
All failures are silently swallowed — the SDK must never affect the agent.
"""
from __future__ import annotations

import json
import logging
import os
import queue
import sys
import threading
from datetime import datetime, timezone
from typing import Any, Optional

import requests

logger = logging.getLogger("vorlo_trace")

# Only log errors if VORLO_DEBUG is set — never pollute the developer's stderr
_DEBUG = os.environ.get("VORLO_DEBUG", "").lower() in ("1", "true", "yes")
if _DEBUG:
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG, format="[vorlo] %(message)s")


_SEND_TIMEOUT_SECONDS = 2
_MAX_QUEUE_SIZE = 10_000  # drop events rather than OOM if server is unreachable


class AsyncSender:
    """
    Fire-and-forget HTTP sender backed by a daemon thread.

    Events are put on an in-memory queue. A single background daemon thread
    reads from the queue and sends via requests.post(). If sending fails,
    the event is silently dropped (logged only in debug mode).

    The daemon thread dies automatically when the main process exits —
    it never blocks shutdown.
    """

    def __init__(self, server_url: str, api_key: str, verify_ssl: bool = True) -> None:
        self._server_url = server_url.rstrip("/")
        self._api_key = api_key
        self._verify_ssl = verify_ssl
        self._queue: queue.Queue[Optional[dict[str, Any]]] = queue.Queue(
            maxsize=_MAX_QUEUE_SIZE
        )
        self._session = requests.Session()
        self._session.headers.update({
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
            "User-Agent": "vorlo-trace-sdk/0.1.0",
        })
        self._thread = threading.Thread(
            target=self._worker,
            name="vorlo-sender",
            daemon=True,  # dies with the main process, never blocks shutdown
        )
        self._thread.start()

    def send(self, event: dict[str, Any]) -> None:
        """
        Enqueue an event for async sending. Never blocks, never raises.

        If the queue is full (server has been unreachable for too long),
        the event is silently dropped.
        """
        if event.get("event_type") != "step":
            return

        try:
            self._queue.put_nowait(event)
        except queue.Full:
            if _DEBUG:
                logger.debug("Queue full — dropping event for session %s", event.get("session_id", "?"))

    def flush(self, timeout: float = 5.0) -> None:
        """
        Wait for the queue to drain, up to timeout seconds.
        Used primarily in tests. Never blocks indefinitely.
        """
        try:
            self._queue.join()
        except Exception:
            pass

    def shutdown(self) -> None:
        """Signal the worker thread to stop. Used in tests and cleanup."""
        try:
            self._queue.put_nowait(None)  # sentinel value
        except queue.Full:
            pass

    def _worker(self) -> None:
        """Background worker: reads events from queue and sends them."""
        endpoint = f"{self._server_url}/v1/trace"
        while True:
            try:
                event = self._queue.get()
                if event is None:
                    # Sentinel value — shutdown requested
                    self._queue.task_done()
                    break
                self._send_event(endpoint, event)
                self._queue.task_done()
            except Exception:
                # Never let the worker thread die from an unexpected error
                if _DEBUG:
                    logger.debug("Worker thread error", exc_info=True)
                try:
                    self._queue.task_done()
                except ValueError:
                    pass

    def _send_event(self, endpoint: str, event: dict[str, Any]) -> None:
        """Send a single event to the Vorlo server. Silent on failure."""
        payload = _to_trace_payload(event)
        if payload is None:
            return

        try:
            self._session.post(
                endpoint,
                data=json.dumps(payload, default=str),
                timeout=_SEND_TIMEOUT_SECONDS,
                verify=self._verify_ssl,
            )
            if _DEBUG:
                logger.debug(
                    "Sent step %s for session %s",
                    event.get("step_number", "?"),
                    event.get("session_id", "?"),
                )
        except requests.exceptions.Timeout:
            if _DEBUG:
                logger.debug("Timeout sending to %s", endpoint)
        except requests.exceptions.ConnectionError:
            if _DEBUG:
                logger.debug("Connection error sending to %s", endpoint)
        except Exception:
            if _DEBUG:
                logger.debug("Unexpected send error", exc_info=True)


def _to_trace_payload(event: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Convert SDK event shape into the server's /v1/trace payload contract."""
    if event.get("event_type") != "step":
        return None

    tool_type = str(event.get("tool_type") or "sensor").upper()
    if tool_type not in {"SENSOR", "ACTUATOR"}:
        tool_type = "ACTUATOR"

    step = {
        "step_number": event.get("step_number", 1),
        "tool_name": event.get("tool_name", "unknown"),
        "tool_type": tool_type,
        "input": event.get("input", ""),
        "output": event.get("output", ""),
        "error": event.get("error", ""),
        "error_diagnosis": event.get("error_diagnosis"),
        "reasoning": event.get("reasoning", ""),
        "status": event.get("status", "success"),
        "latency_ms": event.get("latency_ms", 0),
        "cost_tokens": event.get("cost_tokens", 0),
        "previous_step_context": event.get("previous_step_context", []),
        "trace_id": event.get("trace_id", ""),
        "span_id": event.get("span_id", ""),
        "parent_span_id": event.get("parent_span_id", ""),
        "created_at": event.get("created_at") or datetime.now(timezone.utc).isoformat(),
    }

    return {
        "session_id": event.get("session_id", ""),
        "agent_name": event.get("agent_name", ""),
        "api_key": event.get("api_key", ""),
        "step": step,
    }

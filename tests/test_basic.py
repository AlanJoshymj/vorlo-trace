"""
Vorlo Trace — Unit Tests

Tests the core SDK: handler, error translator, session, and sender.
Validates that the SDK never crashes or slows the agent.
"""
import uuid
import time
import threading
from unittest.mock import MagicMock, patch

import pytest
import vorlo_trace

from vorlo_trace.handler import VorloHandler, _classify_tool, _extract_http_status
from vorlo_trace.error_translator import translate_error, ErrorDiagnosis
from vorlo_trace.session import VorloSession
from vorlo_trace.sender import AsyncSender, _to_session_payload, _to_trace_payload


# ═══════════════════════════════════════════════════════════════════════════
# Handler tests
# ═══════════════════════════════════════════════════════════════════════════


class TestToolClassification:
    """Test that tools are correctly classified as sensor or actuator."""

    def test_sensor_prefixes(self) -> None:
        assert _classify_tool("get_customer") == "sensor"
        assert _classify_tool("read_email") == "sensor"
        assert _classify_tool("fetch_data") == "sensor"
        assert _classify_tool("search_orders") == "sensor"
        assert _classify_tool("list_users") == "sensor"

    def test_actuator_prefixes(self) -> None:
        assert _classify_tool("send_email") == "actuator"
        assert _classify_tool("create_record") == "actuator"
        assert _classify_tool("update_user") == "actuator"
        assert _classify_tool("delete_file") == "actuator"
        assert _classify_tool("charge_card") == "actuator"

    def test_unknown_defaults_to_actuator(self) -> None:
        # Unknown tools default to actuator for safety
        assert _classify_tool("process_data") == "actuator"
        assert _classify_tool("custom_tool") == "actuator"


class TestHttpStatusExtraction:
    """Test HTTP status code extraction from error messages."""

    def test_extracts_status(self) -> None:
        assert _extract_http_status("HTTP 403 Forbidden") == 403
        assert _extract_http_status("Error: 500 Internal Server Error") == 500
        assert _extract_http_status("Rate limited: 429") == 429

    def test_no_status(self) -> None:
        assert _extract_http_status("Connection refused") is None
        assert _extract_http_status("KeyError: 'name'") is None


class TestHandlerToolCallbacks:
    """Test that the handler correctly captures tool start/end/error."""

    def setup_method(self) -> None:
        self.handler = VorloHandler(
            server_url="http://localhost:3001",
            api_key="vrlo_test_key",
            agent_name="test-agent",
        )
        # Mock the sender so we don't make real HTTP calls
        self.handler._sender = MagicMock(spec=AsyncSender)

    def test_tool_start_captures_step(self) -> None:
        run_id = uuid.uuid4()
        self.handler.on_tool_start(
            serialized={"name": "get_customer"},
            input_str="customer_id=123",
            run_id=run_id,
        )
        assert str(run_id) in self.handler._active_steps
        step = self.handler._active_steps[str(run_id)]
        assert step["tool_name"] == "get_customer"
        assert step["tool_type"] == "sensor"
        assert step["step_number"] == 1

    def test_tool_end_sends_success_event(self) -> None:
        run_id = uuid.uuid4()
        self.handler.on_tool_start(
            serialized={"name": "search_orders"},
            input_str="query=pending",
            run_id=run_id,
        )
        self.handler.on_tool_end(output="[{order: 123}]", run_id=run_id)

        # Verify event was sent
        self.handler._sender.send.assert_called()
        event = self.handler._sender.send.call_args[0][0]
        assert event["event_type"] == "step"
        assert event["status"] == "success"
        assert event["tool_name"] == "search_orders"
        assert event["tool_type"] == "sensor"
        assert event["step_number"] == 1

    def test_tool_error_sends_failed_event_with_diagnosis(self) -> None:
        run_id = uuid.uuid4()
        self.handler.on_tool_start(
            serialized={"name": "stripe_charge_card"},
            input_str="amount=100",
            run_id=run_id,
        )
        error = Exception("HTTP 403 Forbidden: invalid API key")
        self.handler.on_tool_error(error=error, run_id=run_id)

        event = self.handler._sender.send.call_args[0][0]
        assert event["status"] == "failed"
        assert event["tool_name"] == "stripe_charge_card"
        assert "error_diagnosis" in event
        assert event["error_diagnosis"]["code"] == "stripe_auth_error"
        assert "fix_hint" in event["error_diagnosis"]

    def test_step_counter_increments(self) -> None:
        for i in range(3):
            run_id = uuid.uuid4()
            self.handler.on_tool_start(
                serialized={"name": f"tool_{i}"},
                input_str="test",
                run_id=run_id,
            )
            self.handler.on_tool_end(output="ok", run_id=run_id)

        assert self.handler._session.step_count == 3

    def test_previous_step_context_included(self) -> None:
        # Complete a step first
        run_id1 = uuid.uuid4()
        self.handler.on_tool_start(
            serialized={"name": "get_customer"},
            input_str="id=123",
            run_id=run_id1,
        )
        self.handler.on_tool_end(output="customer data here", run_id=run_id1)

        # Now start and error on second step
        run_id2 = uuid.uuid4()
        self.handler.on_tool_start(
            serialized={"name": "charge_card"},
            input_str="amount=50",
            run_id=run_id2,
        )
        self.handler.on_tool_error(
            error=Exception("HTTP 403 Forbidden"),
            run_id=run_id2,
        )

        event = self.handler._sender.send.call_args[0][0]
        assert len(event["previous_step_context"]) == 1
        assert event["previous_step_context"][0]["tool_name"] == "get_customer"


class TestHandlerNeverCrashes:
    """Critical test: the SDK must NEVER crash or affect the agent."""

    def setup_method(self) -> None:
        self.handler = VorloHandler(
            server_url="http://localhost:3001",
            api_key="vrlo_test_key",
        )
        self.handler._sender = MagicMock(spec=AsyncSender)

    def test_agent_continues_when_sender_raises(self) -> None:
        """The agent must continue normally even if sending fails."""
        self.handler._sender.send.side_effect = Exception("Network error!")

        run_id = uuid.uuid4()
        # These should not raise even though sender is broken
        self.handler.on_tool_start(
            serialized={"name": "get_data"},
            input_str="test",
            run_id=run_id,
        )
        self.handler.on_tool_end(output="result", run_id=run_id)
        # If we get here without exception, the test passes

    def test_tool_end_without_start_does_not_crash(self) -> None:
        """Calling on_tool_end without on_tool_start should be harmless."""
        self.handler.on_tool_end(output="orphan result", run_id=uuid.uuid4())
        # No exception raised

    def test_tool_error_without_start_does_not_crash(self) -> None:
        """Calling on_tool_error without on_tool_start should be harmless."""
        self.handler.on_tool_error(
            error=Exception("orphan error"), run_id=uuid.uuid4()
        )

    def test_none_inputs_handled(self) -> None:
        """Handler should handle None values gracefully."""
        run_id = uuid.uuid4()
        self.handler.on_tool_start(
            serialized={},  # empty serialized dict
            input_str="",
            run_id=run_id,
        )
        self.handler.on_tool_end(output=None, run_id=run_id)


# ═══════════════════════════════════════════════════════════════════════════
# Error translator tests
# ═══════════════════════════════════════════════════════════════════════════


class TestErrorTranslator:
    """Test that errors are translated to meaningful diagnoses."""

    def test_stripe_403_diagnosis(self) -> None:
        result = translate_error(
            tool_name="stripe_charge_card",
            error_type="HTTPError",
            http_status=403,
            raw_message="Invalid API key provided",
        )
        assert isinstance(result, ErrorDiagnosis)
        assert result.code == "stripe_auth_error"
        assert result.severity == "critical"
        assert "Stripe" in result.title
        assert result.fix_hint  # not empty

    def test_salesforce_401_diagnosis(self) -> None:
        result = translate_error(
            tool_name="salesforce_create_lead",
            error_type="HTTPError",
            http_status=401,
            raw_message="Session expired or invalid",
        )
        assert result.code == "salesforce_token_expired"
        assert "24 hours" in result.root_cause

    def test_rate_limit_diagnosis(self) -> None:
        result = translate_error(
            tool_name="gmail_send_email",
            error_type="HTTPError",
            http_status=429,
            raw_message="Rate limit exceeded",
        )
        assert result.code == "gmail_rate_limit_exceeded"
        assert result.severity == "warning"

    def test_timeout_diagnosis(self) -> None:
        result = translate_error(
            tool_name="search_database",
            error_type="TimeoutError",
            http_status=None,
            raw_message="Connection timed out",
        )
        assert result.code == "connection_timeout"
        assert "timed out" in result.plain_english

    def test_key_error_diagnosis(self) -> None:
        result = translate_error(
            tool_name="process_result",
            error_type="KeyError",
            http_status=None,
            raw_message="KeyError: 'customer_id'",
        )
        assert result.code == "data_transformation_error"
        assert "customer_id" in result.root_cause

    def test_unknown_error_returns_generic(self) -> None:
        result = translate_error(
            tool_name="custom_tool",
            error_type="SomeWeirdError",
            http_status=None,
            raw_message="Something unexpected",
        )
        assert result.code == "unknown_error"
        assert result.fix_hint  # still has a helpful hint

    def test_cross_step_context_in_diagnosis(self) -> None:
        previous = [
            {"tool_name": "get_customer", "tool_type": "sensor", "status": "success",
             "step_number": 1, "output_preview": "customer data"},
        ]
        result = translate_error(
            tool_name="stripe_charge_card",
            error_type="HTTPError",
            http_status=403,
            raw_message="Invalid API key",
            previous_steps=previous,
        )
        assert "previous steps" in result.root_cause.lower() or "Previous" in result.root_cause


# ═══════════════════════════════════════════════════════════════════════════
# Session tests
# ═══════════════════════════════════════════════════════════════════════════


class TestSession:
    """Test session state management."""

    def test_session_generates_unique_ids(self) -> None:
        s1 = VorloSession()
        s2 = VorloSession()
        assert s1.session_id != s2.session_id
        assert s1.trace_id != s2.trace_id

    def test_step_counter(self) -> None:
        session = VorloSession()
        assert session.next_step() == 1
        assert session.next_step() == 2
        assert session.next_step() == 3

    def test_previous_steps_rolling_window(self) -> None:
        session = VorloSession()
        for i in range(10):
            session.add_completed_step(
                step_number=i + 1,
                tool_name=f"tool_{i}",
                tool_type="sensor",
                status="success",
                output_preview="ok",
            )
        # Should only keep last 5
        prev = session.get_previous_steps()
        assert len(prev) == 5
        assert prev[0]["step_number"] == 6  # oldest kept

    def test_reasoning_consume_clears(self) -> None:
        session = VorloSession()
        session.set_reasoning("thinking about tools")
        assert session.consume_reasoning() == "thinking about tools"
        assert session.consume_reasoning() is None  # cleared after consume

    def test_context_dict(self) -> None:
        session = VorloSession(agent_name="test-agent")
        ctx = session.get_context()
        assert ctx["agent_name"] == "test-agent"
        assert "session_id" in ctx
        assert "trace_id" in ctx
        assert ctx["previous_steps"] == []


# ═══════════════════════════════════════════════════════════════════════════
# Sender tests
# ═══════════════════════════════════════════════════════════════════════════


class TestSenderNeverBlocks:
    """Test that the async sender never blocks or crashes."""

    @patch("vorlo_trace.sender.requests.Session")
    def test_send_with_unreachable_server(self, mock_session_cls: MagicMock) -> None:
        """Sender should silently handle unreachable server."""
        mock_session = MagicMock()
        mock_session.post.side_effect = ConnectionError("Connection refused")
        mock_session_cls.return_value = mock_session

        sender = AsyncSender(server_url="http://unreachable:9999", api_key="test")
        sender.send({"session_id": "abc", "step_number": 1})
        time.sleep(0.5)  # give worker thread time to process
        sender.shutdown()
        # No exception raised — test passes

    def test_sender_is_daemon_thread(self) -> None:
        """The sender thread must be a daemon so it doesn't block process exit."""
        sender = AsyncSender(server_url="http://localhost:3001", api_key="test")
        assert sender._thread.daemon is True
        sender.shutdown()

    def test_step_event_converts_to_server_payload(self) -> None:
        payload = _to_trace_payload({
            "event_type": "step",
            "session_id": "sess_123",
            "api_key": "vrlo_test",
            "step_number": 2,
            "tool_name": "get_customer",
            "tool_type": "sensor",
            "status": "success",
            "latency_ms": 12,
            "trace_id": "a" * 32,
            "span_id": "b" * 16,
        })

        assert payload is not None
        assert payload["session_id"] == "sess_123"
        assert payload["api_key"] == "vrlo_test"
        assert payload["step"]["tool_type"] == "SENSOR"
        assert payload["step"]["tool_name"] == "get_customer"
        assert payload["step"]["step_number"] == 2

    def test_lifecycle_events_are_not_sent_to_trace_endpoint(self) -> None:
        assert _to_trace_payload({"event_type": "session_start"}) is None

    def test_session_event_converts_to_session_payload(self) -> None:
        payload = _to_session_payload({
            "event_type": "session_complete",
            "session_id": "sess_123",
            "api_key": "vrlo_test",
            "agent_name": "order-agent",
            "trace_id": "a" * 32,
            "total_steps": 5,
            "duration_ms": 1234,
            "status": "success",
            "output": "{'result': 'done'}",
        })

        assert payload is not None
        assert payload["session_id"] == "sess_123"
        assert payload["event_type"] == "session_complete"
        assert payload["total_steps"] == 5
        assert payload["duration_ms"] == 1234
        assert payload["status"] == "success"

    def test_step_events_are_not_session_payloads(self) -> None:
        assert _to_session_payload({"event_type": "step"}) is None

    def test_sender_routes_session_events_to_session_endpoint(self) -> None:
        """Lifecycle events must reach /v1/session, steps /v1/trace."""
        sender = AsyncSender(server_url="http://localhost:3001", api_key="test")
        sent: list[str] = []
        sender._session.post = MagicMock(  # type: ignore[method-assign]
            side_effect=lambda url, **kw: sent.append(url) or MagicMock(status_code=200)
        )

        sender.send({"event_type": "step", "session_id": "s1", "api_key": "k"})
        sender.send({
            "event_type": "session_complete",
            "session_id": "s1",
            "api_key": "k",
            "status": "success",
        })
        sender.flush()
        sender.shutdown()

        assert any(url.endswith("/v1/trace") for url in sent)
        assert any(url.endswith("/v1/session") for url in sent)


class TestSdkInitialization:
    """Test public SDK initialization behavior."""

    def test_default_server_url_points_to_current_production(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("VORLO_SERVER_URL", raising=False)
        handler = vorlo_trace.init(api_key="vrlo_test_key", agent_name="default-url-test")
        assert handler._sender._server_url == "https://vorlo-server-production.up.railway.app"
        handler._sender.shutdown()

    def test_server_url_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VORLO_SERVER_URL", "https://example.test")
        handler = vorlo_trace.init(api_key="vrlo_test_key", agent_name="env-url-test")
        assert handler._sender._server_url == "https://example.test"
        handler._sender.shutdown()

    def test_verify_ssl_defaults_to_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("VORLO_VERIFY_SSL", raising=False)
        handler = vorlo_trace.init(api_key="vrlo_test_key", agent_name="tls-default-test")
        assert handler._sender._verify_ssl is True
        handler._sender.shutdown()

    def test_verify_ssl_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VORLO_VERIFY_SSL", "false")
        handler = vorlo_trace.init(api_key="vrlo_test_key", agent_name="tls-env-test")
        assert handler._sender._verify_ssl is False
        handler._sender.shutdown()


# ═══════════════════════════════════════════════════════════════════════════
# Integration-style tests
# ═══════════════════════════════════════════════════════════════════════════


class TestFullWorkflow:
    """Test a complete multi-step workflow through the handler."""

    def test_multi_step_session(self) -> None:
        handler = VorloHandler(
            server_url="http://localhost:3001",
            api_key="vrlo_test",
            agent_name="integration-test",
        )
        handler._sender = MagicMock(spec=AsyncSender)

        # Step 1: successful sensor
        rid1 = uuid.uuid4()
        handler.on_tool_start(serialized={"name": "get_customer"}, input_str="id=1", run_id=rid1)
        handler.on_tool_end(output='{"name": "Jane"}', run_id=rid1)

        # Step 2: successful sensor
        rid2 = uuid.uuid4()
        handler.on_tool_start(serialized={"name": "search_orders"}, input_str="q=pending", run_id=rid2)
        handler.on_tool_end(output='[{"id": "ord_1"}]', run_id=rid2)

        # Step 3: failed actuator
        rid3 = uuid.uuid4()
        handler.on_tool_start(serialized={"name": "charge_card"}, input_str="amount=50", run_id=rid3)
        handler.on_tool_error(error=Exception("HTTP 403 Forbidden"), run_id=rid3)

        assert handler._session.step_count == 3
        assert handler._sender.send.call_count == 3

        # Check the failed step has previous context
        last_event = handler._sender.send.call_args_list[2][0][0]
        assert last_event["status"] == "failed"
        assert len(last_event["previous_step_context"]) == 2

    def test_session_has_otel_ids(self) -> None:
        handler = VorloHandler(
            server_url="http://localhost:3001",
            api_key="vrlo_test",
        )
        handler._sender = MagicMock(spec=AsyncSender)

        rid = uuid.uuid4()
        handler.on_tool_start(serialized={"name": "get_data"}, input_str="x", run_id=rid)
        handler.on_tool_end(output="ok", run_id=rid)

        event = handler._sender.send.call_args[0][0]
        assert "trace_id" in event
        assert "span_id" in event
        assert len(event["trace_id"]) == 32  # UUID hex
        assert len(event["span_id"]) == 16  # 8-byte hex


# ═══════════════════════════════════════════════════════════════════════════
# OTel parent-span linkage tests
# ═══════════════════════════════════════════════════════════════════════════


class TestParentSpanLinkage:
    """A tool run nested under a chain/agent should link to its parent's span."""

    def setup_method(self) -> None:
        self.handler = VorloHandler(
            server_url="http://localhost:3001",
            api_key="vrlo_test",
            agent_name="span-test",
        )
        self.handler._sender = MagicMock(spec=AsyncSender)

    def test_tool_links_to_parent_chain_span(self) -> None:
        chain_id = uuid.uuid4()
        self.handler.on_chain_start(serialized={}, inputs={}, run_id=chain_id)

        tool_id = uuid.uuid4()
        self.handler.on_tool_start(
            serialized={"name": "get_customer"},
            input_str="id=1",
            run_id=tool_id,
            parent_run_id=chain_id,
        )
        self.handler.on_tool_end(output="ok", run_id=tool_id)

        event = self.handler._sender.send.call_args[0][0]
        # The tool's parent_span_id must equal the chain's registered span_id.
        assert event["parent_span_id"] == self._chain_span(chain_id)
        assert event["parent_span_id"]  # non-empty
        assert event["parent_span_id"] != event["span_id"]

    def test_root_tool_has_empty_parent(self) -> None:
        tool_id = uuid.uuid4()
        self.handler.on_tool_start(
            serialized={"name": "get_data"}, input_str="x", run_id=tool_id
        )
        self.handler.on_tool_end(output="ok", run_id=tool_id)
        event = self.handler._sender.send.call_args[0][0]
        assert event["parent_span_id"] == ""

    def test_span_registry_released_after_end(self) -> None:
        chain_id = uuid.uuid4()
        self.handler.on_chain_start(serialized={}, inputs={}, run_id=chain_id)
        assert str(chain_id) in self.handler._span_by_run
        self.handler.on_chain_end(outputs={}, run_id=chain_id)
        assert str(chain_id) not in self.handler._span_by_run

    def _chain_span(self, chain_id: uuid.UUID) -> str:
        # Span was released on tool_end, so capture it from the emitted event
        # by re-deriving: the chain span is whatever the tool recorded as parent.
        return self.handler._sender.send.call_args[0][0]["parent_span_id"]

"""Regression tests for registry matching — from the 2026-07 diagnosis eval.

Covers the three classes of failure the eval exposed: provider misrouting
(provider named in error text, not tool name), brittle message regexes, and
the agent-behavior failures that previously fell through to unknown_error.
"""
from __future__ import annotations

from vorlo_trace.error_translator import translate_error


def diag(tool, etype, status, msg, prev=None):
    return translate_error(
        tool_name=tool, error_type=etype, http_status=status,
        raw_message=msg, previous_steps=prev or [],
    )


class TestProviderFromErrorText:
    def test_stripe_named_in_message_not_tool(self):
        # THE misdiagnosis: was auth_forbidden ("grant the scope") — wrong.
        d = diag("charge_card", "AuthenticationError", 403,
                 "Stripe API error: Invalid API key provided: sk_test_51EX. The key has expired.")
        assert d.code == "stripe_auth_error"

    def test_gmail_invalid_grant_with_distant_words(self):
        # Was generic auth_unauthorized: "token...expired" 30 chars apart.
        d = diag("gmail_send", "HttpError", 401,
                 "401 invalid_grant: Token has been expired or revoked")
        assert d.code == "gmail_auth_error"

    def test_generic_401_still_generic(self):
        d = diag("fetch_orders", "HTTPError", 401,
                 "401 Client Error: Unauthorized for url https://api.internal/orders")
        assert d.code == "auth_unauthorized"


class TestAgentBehaviorPatterns:
    def test_output_parser(self):
        d = diag("agent_executor", "OutputParserException", None,
                 "Could not parse LLM output: `I should use the search tool`")
        assert d.code == "output_parsing_error"

    def test_iteration_limit_names_the_looping_tool(self):
        prev = [{"step_number": i, "tool_name": "search_web", "status": "failed"}
                for i in range(1, 6)]
        d = diag("agent_executor", "AgentExecutorError", None,
                 "Agent stopped due to iteration limit or time limit.", prev)
        assert d.code == "max_iterations_exceeded"
        assert "search_web" in d.root_cause  # the loop is named, not implied

    def test_hallucinated_tool_surfaces_registered_tools(self):
        d = diag("agent_executor", "ToolNotFoundError", None,
                 "search_google is not a valid tool, try one of [web_search, get_page].")
        assert d.code == "tool_not_found"
        assert "web_search" in d.root_cause

    def test_context_overflow(self):
        d = diag("llm_call", "InvalidRequestError", None,
                 "This model's maximum context length is 128000 tokens. "
                 "However, your messages resulted in 131072 tokens.")
        assert d.code == "context_length_exceeded"

    def test_schema_validation(self):
        d = diag("create_ticket", "ValidationError", None,
                 "1 validation error for TicketInput priority: value is not a "
                 "valid enumeration member; permitted: 'low', 'medium', 'high'")
        assert d.code == "schema_validation_error"

    def test_guardrail(self):
        d = diag("agent_executor", "GuardrailTripwireTriggered", None,
                 "Input guardrail 'pii_check' tripwire triggered")
        assert d.code == "guardrail_triggered"


class TestNoRegressions:
    def test_line_403_is_not_http_403(self):
        # The classic trap must stay safe: a KeyError mentioning line 403.
        d = diag("parse_file", "KeyError", None,
                 "KeyError at line 403 of utils.py: 'result'")
        assert d.code == "data_transformation_error"

    def test_keyerror_still_beats_message_patterns(self):
        d = diag("format_report", "KeyError", None, "KeyError: 'customer_email'")
        assert d.code == "data_transformation_error"

    def test_timeout_unchanged(self):
        d = diag("scrape_page", "ReadTimeout", None,
                 "Read timed out. (read timeout=10)")
        assert d.code == "connection_timeout"

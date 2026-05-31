"""
Real end-to-end agent run against the live Vorlo server, using only
langchain_core primitives (no external LLM key needed).

Each scenario builds a genuine ReAct-style chain: a parent RunnableLambda
("agent") invokes child tools through LangChain's callback manager, so the
tool runs receive a real parent_run_id == the chain's run_id. This exercises
the Vorlo handler's span registry / parent_span_id linkage end-to-end and
ships the traces to the production server via the real sender thread.
"""

import os
import sys
import time
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from langchain_core.tools import tool
from langchain_core.runnables import RunnableLambda

import vorlo_trace

SERVER = os.getenv("VORLO_SERVER_URL", "https://vorlo-server-production.up.railway.app")


# ─── Tools (mix of sensors / actuators, some fail) ──────────────────────────

@tool
def get_customer(customer_id: str) -> str:
    """Look up customer details by ID."""
    time.sleep(0.05)
    return json.dumps({"id": customer_id, "name": "Jane Doe", "plan": "enterprise"})


@tool
def search_orders(query: str) -> str:
    """Search for orders matching a query."""
    time.sleep(0.05)
    return json.dumps([{"order_id": "ord_8821", "status": "shipped"}])


@tool
def charge_card(customer_id: str, amount: float) -> str:
    """Charge a customer's card."""
    time.sleep(0.05)
    raise Exception(
        "HTTP 403: Your API key does not have the required scope 'charges:write'. "
        "You are using a read-only key rk_test_xxxx."
    )


@tool
def send_email(to: str, subject: str) -> str:
    """Send an email."""
    time.sleep(0.05)
    raise TimeoutError("Connection timed out after 30s — smtp.gmail.com did not respond")


@tool
def sync_salesforce(account_id: str) -> str:
    """Sync data with Salesforce."""
    time.sleep(0.05)
    raise Exception(
        "HTTP 401: INVALID_SESSION_ID — Session expired or invalid. "
        "Your Salesforce OAuth token has expired after 24 hours."
    )


@tool
def update_crm(customer_id: str, field: str, value: str) -> str:
    """Update a CRM field."""
    time.sleep(0.05)
    return json.dumps({"updated": True, "customer_id": customer_id})


# ─── Scenarios: (agent_name, [(tool, kwargs), ...]) ─────────────────────────

SCENARIOS = [
    ("customer-lookup-agent", [(get_customer, {"customer_id": "cus_ABC123"})]),
    ("order-search-agent", [(search_orders, {"query": "pending enterprise"})]),
    ("payment-agent", [
        (get_customer, {"customer_id": "cus_ABC123"}),
        (charge_card, {"customer_id": "cus_ABC123", "amount": 299.99}),
    ]),
    ("email-agent", [(send_email, {"to": "john@co.com", "subject": "Update"})]),
    ("salesforce-agent", [(sync_salesforce, {"account_id": "ACC-5501"})]),
    ("crm-multi-step-agent", [
        (get_customer, {"customer_id": "cus_XYZ789"}),
        (update_crm, {"customer_id": "cus_XYZ789", "field": "plan", "value": "enterprise"}),
    ]),
]


def make_agent(steps):
    """A parent chain that invokes child tools, propagating callbacks/run_id."""
    def _run(_input, config=None):
        results = []
        for tool_obj, kwargs in steps:
            try:
                results.append(tool_obj.invoke(kwargs, config=config))
            except Exception as exc:  # noqa: BLE001 — tool failures are expected
                results.append(f"ERROR: {exc}")
        return results
    return RunnableLambda(_run)


def main():
    api_key = os.getenv("VORLO_API_KEY")
    if not api_key:
        raise SystemExit("Set VORLO_API_KEY before running.")

    print(f"Server: {SERVER}")
    session_ids = []

    for name, steps in SCENARIOS:
        vorlo_trace.init(api_key=api_key, server_url=SERVER, agent_name=name)
        handler = vorlo_trace.get_handler()
        session_ids.append(handler._session.session_id)

        agent = make_agent(steps)
        agent.invoke({"task": name}, config={"callbacks": [handler]})

        tool_names = [t.name for t, _ in steps]
        print(f"  ✓ ran {name:24s} tools={tool_names}")

        time.sleep(0.6)  # let the sender flush
        vorlo_trace._handler = None

    print("\nFlushing...")
    time.sleep(2)
    print("Session IDs:")
    for sid in session_ids:
        print(f"  {sid}")


if __name__ == "__main__":
    main()

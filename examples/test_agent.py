"""
Vorlo Trace — Example Agent

Demonstrates a LangChain agent with 4 tools that produce different outcomes:
  1. get_customer — succeeds (sensor)
  2. search_orders — succeeds (sensor)
  3. charge_card — fails with HTTP 403 (actuator)
  4. send_email — times out (actuator)

Run with:
    VORLO_API_KEY=vrlo_test python examples/test_agent.py
"""
import os
import sys
import time

# Add src to path so we can import vorlo_trace without installing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import vorlo_trace

from langchain_core.tools import tool


# ── Define test tools ────────────────────────────────────────────────────

@tool
def get_customer(customer_id: str) -> str:
    """Fetch customer details by ID. (sensor — read-only)"""
    # Simulates a successful API call
    time.sleep(0.1)
    return f'{{"id": "{customer_id}", "name": "Jane Doe", "email": "jane@example.com", "plan": "pro"}}'


@tool
def search_orders(query: str) -> str:
    """Search orders matching a query. (sensor — read-only)"""
    time.sleep(0.15)
    return '[{"order_id": "ord_123", "amount": 4999, "status": "pending"}]'


@tool
def charge_card(customer_id: str, amount: int) -> str:
    """Charge a customer's card. (actuator — state-changing)"""
    time.sleep(0.05)
    # Simulates a Stripe 403 error — wrong API key scope
    raise Exception("HTTP 403 Forbidden: Your API key does not have the required scope 'charges:write'. "
                    "You are using a read-only key.")


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to a customer. (actuator — state-changing)"""
    time.sleep(0.05)
    # Simulates a timeout
    raise TimeoutError("Connection timed out after 30s — smtp.gmail.com did not respond")


# ── Run the demo ─────────────────────────────────────────────────────────

def main() -> None:
    # Initialize Vorlo — 2 lines of code
    api_key = os.environ.get("VORLO_API_KEY", "vrlo_test_key_for_demo")
    server_url = os.environ.get("VORLO_SERVER_URL", "http://localhost:3001")

    vorlo_trace.init(
        api_key=api_key,
        server_url=server_url,
        agent_name="demo-order-agent",
    )
    handler = vorlo_trace.get_handler()

    print(f"Session ID: {handler.session_id}")
    print(f"Agent name: {handler.agent_name}")
    print("-" * 60)

    # Simulate an agent workflow by calling tools directly with handler callbacks
    # (In production, the LangChain agent calls these automatically via callbacks)

    import uuid

    tools = [get_customer, search_orders, charge_card, send_email]

    for i, tool_fn in enumerate(tools):
        run_id = uuid.uuid4()
        tool_name = tool_fn.name
        print(f"\nStep {i + 1}: calling {tool_name}...")

        # Simulate on_tool_start
        handler.on_tool_start(
            serialized={"name": tool_name},
            input_str=f"test input for {tool_name}",
            run_id=run_id,
        )

        try:
            if tool_name == "get_customer":
                result = get_customer.invoke({"customer_id": "cus_ABC123"})
            elif tool_name == "search_orders":
                result = search_orders.invoke({"query": "pending orders"})
            elif tool_name == "charge_card":
                result = charge_card.invoke({"customer_id": "cus_ABC123", "amount": 4999})
            elif tool_name == "send_email":
                result = send_email.invoke({"to": "jane@example.com", "subject": "Receipt", "body": "Thanks!"})
            else:
                result = "ok"

            handler.on_tool_end(output=result, run_id=run_id)
            print(f"  ✓ Success: {result[:80]}...")

        except Exception as e:
            handler.on_tool_error(error=e, run_id=run_id)
            print(f"  ✗ Failed: {e}")

    print("\n" + "=" * 60)
    handler._sender.flush(timeout=5.0)
    print(f"Session complete. {handler._session.step_count} steps recorded.")
    print(f"Duration: {handler._session.duration_ms}ms")
    print(f"\nView this session at: https://vorlo.dev/sessions/{handler.session_id}")


if __name__ == "__main__":
    main()

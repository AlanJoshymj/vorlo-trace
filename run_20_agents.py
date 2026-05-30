"""
Run 20 LangGraph/LangChain agents with Vorlo tracing.
Some succeed, some fail deliberately to test the full debugging pipeline.
"""

import os
import sys
import time
import json
import random

# Add the src directory so we can import vorlo_trace locally
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.tools import tool
    from langchain_core.messages import HumanMessage
    from langgraph.prebuilt import create_react_agent
except ImportError as exc:
    raise SystemExit(
        "Install the real-agent example dependencies first: "
        "pip install langchain-openai langgraph"
    ) from exc

import vorlo_trace

# ─── Configuration ───────────────────────────────────────────────────────────

VORLO_API_KEY = os.getenv("VORLO_API_KEY")
LLM_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.siemens.com/llm/v1")
LLM_MODEL = os.getenv("OPENAI_MODEL", "qwen-3.6-27b")
VORLO_SERVER_URL = os.getenv("VORLO_SERVER_URL", "https://vorlo-server-production.up.railway.app")
VORLO_VERIFY_SSL = os.getenv("VORLO_VERIFY_SSL", "").lower() not in {"0", "false", "no", "off"}

# ─── Tool Definitions (mix of sensors and actuators, some that fail) ─────────

@tool
def get_customer(customer_id: str) -> str:
    """Look up customer details by ID."""
    time.sleep(0.1)
    return json.dumps({
        "id": customer_id,
        "name": "Jane Doe",
        "email": "jane@example.com",
        "plan": "enterprise",
        "balance": 4500.00
    })

@tool
def search_orders(query: str) -> str:
    """Search for orders matching a query."""
    time.sleep(0.15)
    return json.dumps([
        {"order_id": "ord_8821", "amount": 299.99, "status": "shipped"},
        {"order_id": "ord_8822", "amount": 1249.00, "status": "pending"},
    ])

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to a recipient. FAILS with timeout."""
    time.sleep(0.05)
    raise TimeoutError("Connection timed out after 30s — smtp.gmail.com did not respond")

@tool
def charge_card(customer_id: str, amount: float) -> str:
    """Charge a customer's card. FAILS with Stripe auth error."""
    time.sleep(0.05)
    raise Exception("HTTP 403: Your API key does not have the required scope 'charges:write'. You are using a read-only key rk_test_xxxx.")

@tool
def update_crm(customer_id: str, field: str, value: str) -> str:
    """Update a field in the CRM for a customer."""
    time.sleep(0.12)
    return json.dumps({"updated": True, "customer_id": customer_id, "field": field, "new_value": value})

@tool
def check_inventory(product_id: str) -> str:
    """Check inventory levels for a product."""
    time.sleep(0.08)
    return json.dumps({"product_id": product_id, "available": 142, "warehouse": "US-West-2"})

@tool
def track_shipment(tracking_id: str) -> str:
    """Track a shipment by tracking number."""
    time.sleep(0.1)
    return json.dumps({
        "tracking_id": tracking_id,
        "status": "in_transit",
        "location": "Memphis, TN",
        "eta": "2026-06-02"
    })

@tool
def generate_invoice(order_id: str, amount: float) -> str:
    """Generate an invoice PDF for an order."""
    time.sleep(0.2)
    return json.dumps({"invoice_id": "inv_9912", "pdf_url": "/invoices/inv_9912.pdf", "amount": amount})

@tool
def sync_salesforce(account_id: str) -> str:
    """Sync data with Salesforce. FAILS with token expired."""
    time.sleep(0.05)
    raise Exception("HTTP 401: INVALID_SESSION_ID — Session expired or invalid. Your Salesforce OAuth token has expired after 24 hours.")

@tool
def send_slack_message(channel: str, message: str) -> str:
    """Send a message to a Slack channel."""
    time.sleep(0.08)
    return json.dumps({"ok": True, "channel": channel, "ts": "1717020000.001"})

@tool
def query_database(sql: str) -> str:
    """Execute a read-only database query."""
    time.sleep(0.15)
    return json.dumps({"rows": [{"id": 1, "name": "Widget A", "revenue": 52000}, {"id": 2, "name": "Widget B", "revenue": 31000}]})

@tool
def upload_file(filename: str, destination: str) -> str:
    """Upload a file to cloud storage. FAILS with permission denied."""
    time.sleep(0.05)
    raise PermissionError("HTTP 403: Access denied. The service account does not have 'storage.objects.create' permission on bucket 'prod-reports'.")

@tool
def create_pdf_report(title: str, data: str) -> str:
    """Generate a PDF report from data."""
    time.sleep(0.25)
    return json.dumps({"report_id": "rpt_445", "pages": 3, "url": "/reports/rpt_445.pdf"})

@tool
def book_calendar_event(title: str, date: str, attendees: str) -> str:
    """Book a calendar event with attendees."""
    time.sleep(0.1)
    return json.dumps({"event_id": "evt_221", "confirmed": True, "date": date})

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    time.sleep(0.05)
    return json.dumps({"city": city, "temp_c": 22, "condition": "partly cloudy", "humidity": 45})

@tool
def translate_text(text: str, target_language: str) -> str:
    """Translate text to a target language."""
    time.sleep(0.1)
    return json.dumps({"translated": f"[{target_language}] {text[:50]}...", "confidence": 0.94})

@tool
def analyze_sentiment(text: str) -> str:
    """Analyze sentiment of text."""
    time.sleep(0.08)
    return json.dumps({"sentiment": "positive", "score": 0.87, "keywords": ["great", "excellent", "recommend"]})

@tool
def create_gmail_draft(to: str, subject: str, body: str) -> str:
    """Create a Gmail draft. FAILS with rate limit."""
    time.sleep(0.05)
    raise Exception("HTTP 429: Rate limit exceeded. Gmail API allows 25 requests per minute per user. Retry after 42 seconds.")

@tool
def export_data(format: str, filters: str) -> str:
    """Export data to a file. FAILS with missing field."""
    time.sleep(0.05)
    raise KeyError("customer_segment")

@tool
def build_report(report_type: str, period: str) -> str:
    """Build an analytics report for a time period."""
    time.sleep(0.2)
    return json.dumps({
        "report_type": report_type,
        "period": period,
        "summary": {"total_revenue": 142000, "active_users": 891, "churn_rate": 0.03}
    })

@tool
def list_recent_tickets(status: str) -> str:
    """List recent support tickets by status."""
    time.sleep(0.1)
    return json.dumps([
        {"ticket_id": "TKT-901", "subject": "Login issues", "priority": "high"},
        {"ticket_id": "TKT-902", "subject": "Billing question", "priority": "medium"},
    ])

@tool
def fetch_analytics(metric: str, days: int) -> str:
    """Fetch analytics data for a metric over N days."""
    time.sleep(0.12)
    return json.dumps({"metric": metric, "days": days, "values": [100, 120, 115, 140, 135, 160, 155]})


# ─── Agent Scenarios ─────────────────────────────────────────────────────────

SCENARIOS = [
    {
        "name": "customer-lookup-agent",
        "task": "Look up customer cus_ABC123 and check their details",
        "tools": [get_customer],
        "should_fail": False,
    },
    {
        "name": "order-search-agent",
        "task": "Search for all pending orders for the enterprise plan",
        "tools": [search_orders],
        "should_fail": False,
    },
    {
        "name": "email-notification-agent",
        "task": "Send an email to john@company.com with subject 'Order Update' about their shipment",
        "tools": [send_email],
        "should_fail": True,
    },
    {
        "name": "payment-processing-agent",
        "task": "Charge customer cus_ABC123 amount 299.99 for their order",
        "tools": [charge_card, get_customer],
        "should_fail": True,
    },
    {
        "name": "crm-update-agent",
        "task": "Update customer cus_XYZ789 plan field to 'enterprise' in the CRM",
        "tools": [update_crm, get_customer],
        "should_fail": False,
    },
    {
        "name": "inventory-check-agent",
        "task": "Check inventory levels for product SKU-1001",
        "tools": [check_inventory],
        "should_fail": False,
    },
    {
        "name": "shipment-tracker-agent",
        "task": "Track shipment with tracking number 1Z999AA10123456784",
        "tools": [track_shipment],
        "should_fail": False,
    },
    {
        "name": "invoice-generator-agent",
        "task": "Generate an invoice for order ord_8821 with amount 299.99",
        "tools": [generate_invoice, search_orders],
        "should_fail": False,
    },
    {
        "name": "salesforce-sync-agent",
        "task": "Sync account ACC-5501 data with Salesforce",
        "tools": [sync_salesforce],
        "should_fail": True,
    },
    {
        "name": "slack-notifier-agent",
        "task": "Send a message to #engineering channel saying 'Deployment complete'",
        "tools": [send_slack_message],
        "should_fail": False,
    },
    {
        "name": "database-analyst-agent",
        "task": "Query the database for top products by revenue",
        "tools": [query_database],
        "should_fail": False,
    },
    {
        "name": "file-upload-agent",
        "task": "Upload the monthly-report.csv file to cloud storage at /reports/2026/",
        "tools": [upload_file],
        "should_fail": True,
    },
    {
        "name": "report-builder-agent",
        "task": "Create a PDF report titled 'Q2 Revenue Summary' with the latest revenue data",
        "tools": [create_pdf_report, query_database],
        "should_fail": False,
    },
    {
        "name": "calendar-booking-agent",
        "task": "Book a team standup meeting for June 2 2026 with alice@co.com and bob@co.com",
        "tools": [book_calendar_event],
        "should_fail": False,
    },
    {
        "name": "weather-lookup-agent",
        "task": "Get the current weather in Munich, Germany",
        "tools": [get_weather],
        "should_fail": False,
    },
    {
        "name": "translation-agent",
        "task": "Translate 'Hello, how can I help you today?' to German",
        "tools": [translate_text],
        "should_fail": False,
    },
    {
        "name": "sentiment-analysis-agent",
        "task": "Analyze the sentiment of this review: 'This product is absolutely fantastic, best purchase I ever made!'",
        "tools": [analyze_sentiment],
        "should_fail": False,
    },
    {
        "name": "gmail-draft-agent",
        "task": "Create a Gmail draft to boss@company.com with subject 'Weekly Update' about project status",
        "tools": [create_gmail_draft],
        "should_fail": True,
    },
    {
        "name": "data-export-agent",
        "task": "Export data in CSV format with filter for last 30 days",
        "tools": [export_data],
        "should_fail": True,
    },
    {
        "name": "analytics-report-agent",
        "task": "Build a monthly analytics report for May 2026 covering revenue and user growth metrics",
        "tools": [build_report, fetch_analytics, list_recent_tickets],
        "should_fail": False,
    },
]


# ─── Agent Runner ────────────────────────────────────────────────────────────

def create_llm():
    """Create LLM client pointing to Siemens API."""
    return ChatOpenAI(
        model=LLM_MODEL,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        temperature=0.1,
        max_tokens=1024,
    )


def run_agent(scenario: dict, agent_number: int):
    """Run a single agent scenario with Vorlo tracing."""
    name = scenario["name"]
    task = scenario["task"]
    tools = scenario["tools"]
    should_fail = scenario["should_fail"]

    print(f"\n{'='*70}")
    print(f"  Agent {agent_number:02d}/20: {name}")
    print(f"  Task: {task}")
    print(f"  Tools: {[t.name for t in tools]}")
    print(f"  Expected: {'FAIL ✗' if should_fail else 'SUCCESS ✓'}")
    print(f"{'='*70}")

    # Initialize Vorlo for this agent (new session per agent)
    vorlo_trace.init(
        api_key=VORLO_API_KEY,
        server_url=VORLO_SERVER_URL,
        agent_name=name,
        verify_ssl=VORLO_VERIFY_SSL,
    )
    handler = vorlo_trace.get_handler()

    # Create the LLM and agent
    llm = create_llm()

    # Use langgraph's create_react_agent
    agent = create_react_agent(llm, tools)

    start = time.time()
    try:
        result = agent.invoke(
            {"messages": [HumanMessage(content=task)]},
            config={"callbacks": [handler], "recursion_limit": 10},
        )
        elapsed = (time.time() - start) * 1000
        # Get last message content
        messages = result.get("messages", [])
        output = str(messages[-1].content)[:200] if messages else "No output"
        print(f"  Result: {output}")
        print(f"  Duration: {elapsed:.0f}ms")
        if should_fail:
            print(f"  ⚠ Expected failure but got success")
        else:
            print(f"  ✓ Success as expected")
    except Exception as e:
        elapsed = (time.time() - start) * 1000
        print(f"  ✗ Error: {str(e)[:150]}")
        print(f"  Duration: {elapsed:.0f}ms")
        if should_fail:
            print(f"  ✓ Failed as expected (testing error capture)")
        else:
            print(f"  ⚠ Unexpected failure!")

    # Give sender time to flush
    time.sleep(0.5)

    # Reset handler for next agent
    vorlo_trace._handler = None

    return {"name": name, "should_fail": should_fail, "elapsed_ms": elapsed}


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    if not VORLO_API_KEY:
        raise SystemExit("Set VORLO_API_KEY before running this script.")
    if not LLM_API_KEY:
        raise SystemExit("Set OPENAI_API_KEY or LLM_API_KEY before running this script.")

    print("\n" + "╔" + "═"*68 + "╗")
    print("║" + " VORLO 20-AGENT TEST RUN ".center(68) + "║")
    print("║" + " Testing full observability pipeline ".center(68) + "║")
    print("╚" + "═"*68 + "╝")
    print(f"\n  LLM: {LLM_MODEL} via {LLM_BASE_URL}")
    print(f"  Vorlo Server: {VORLO_SERVER_URL}")
    print(f"  Agents: 20 (14 success, 6 failures)")
    print(f"  Tracing: vorlo-trace SDK v0.1.0\n")

    results = []
    for i, scenario in enumerate(SCENARIOS, 1):
        try:
            result = run_agent(scenario, i)
            results.append(result)
        except Exception as e:
            print(f"  !! Runner error: {e}")
            results.append({"name": scenario["name"], "should_fail": scenario["should_fail"], "elapsed_ms": 0})

        # Small delay between agents to avoid rate limiting
        time.sleep(1)

    # Summary
    print("\n\n" + "="*70)
    print(" SUMMARY ".center(70, "="))
    print("="*70)
    successes = sum(1 for r in results if not r["should_fail"])
    failures = sum(1 for r in results if r["should_fail"])
    total_time = sum(r["elapsed_ms"] for r in results)
    print(f"  Total agents: {len(results)}")
    print(f"  Expected successes: {successes}")
    print(f"  Expected failures: {failures}")
    print(f"  Total execution time: {total_time:.0f}ms")
    print(f"\n  Check dashboard: https://vorlo.dev")
    print(f"  Or check API: curl -H 'Authorization: Bearer {VORLO_API_KEY[:20]}...' \\")
    print(f"    https://vorlo-server-production.up.railway.app/v1/sessions?page=1&status=all")
    print("="*70)


if __name__ == "__main__":
    main()

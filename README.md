# vorlo-trace

The debugger for AI agents. Add 2 lines of code to see exactly which step of your agent failed, the plain-English root cause, and the exact fix — in 30 seconds.

Works with **LangChain** and the **OpenAI Agents SDK**.

## Quick Start

```bash
pip install vorlo-trace
```

### LangChain

```python
import vorlo_trace
from langchain.agents import create_tool_calling_agent, AgentExecutor

# 1. Initialize Vorlo (or set VORLO_API_KEY env var)
vorlo_trace.init(api_key="vrlo_your_key_here", agent_name="my-agent")

# 2. Add the handler to your agent
handler = vorlo_trace.get_handler()
result = agent_executor.invoke(
    {"input": "Check my latest emails"},
    config={"callbacks": [handler]}
)
```

### OpenAI Agents SDK

```python
import vorlo_trace
from agents import Agent, Runner

# 1. Initialize Vorlo (or set VORLO_API_KEY env var)
vorlo_trace.init(api_key="vrlo_your_key_here", agent_name="my-agent")

# 2. Instrument once — every Runner.run() is traced from here on
vorlo_trace.instrument_openai_agents()

result = await Runner.run(agent, "Check my latest emails")
```

Tool calls, handoffs between agents, and triggered guardrails all appear
as numbered steps — failures carry a plain-English diagnosis and fix.

That's it. Open [vorlo.dev](https://vorlo.dev) to see every step your agent took, with:

- **Step Replay** — chronological view of every tool call
- **Root Cause** — plain English explanation, not raw HTTP errors
- **Fix Hints** — specific, actionable steps to resolve the issue
- **LLM Reasoning** — why the agent made each decision

## What You See

**Without Vorlo:**
```
ToolError: 403
```

**With Vorlo:**
```
Stripe token expired after retry chain caused stale auth context.
Your Stripe OAuth token was issued 23h ago — Stripe tokens expire at 24h.
Fix: Reconnect your Stripe account in Settings → Integrations.
```

## How It Works

Vorlo wraps your agent's callback system. It captures:

1. Every tool call (input, output, latency)
2. Every LLM reasoning step (chain of thought)
3. Every error (translated to plain English with fix hints)
4. Cross-step context (what data flowed between steps)

All sends are **fire-and-forget** — Vorlo never slows or crashes your agent, even if the Vorlo server is unreachable.

## Features

- **2-line integration** — works with any LangChain agent or the OpenAI Agents SDK
- **Sensor/Actuator classification** — reads vs writes, clearly labeled
- **Cross-step root cause** — understands data flow between steps
- **OTel-compatible** — trace_id, span_id for enterprise export
- **Zero impact** — async sends, daemon threads, silent failures

## MCP server — debug from inside your editor

The package ships an MCP server (`vorlo-mcp`), so Claude Code / Cursor / any
MCP client can ask Vorlo *"why did my last run fail?"* without leaving the editor:

```bash
pip install vorlo-trace
claude mcp add vorlo --env VORLO_API_KEY=vrlo_... -- vorlo-mcp
# or without installing first:
claude mcp add vorlo --env VORLO_API_KEY=vrlo_... -- uvx --from vorlo-trace vorlo-mcp
```

Tools: `why_did_my_last_run_fail`, `get_session_diagnosis`,
`list_recent_sessions`, `get_failure_clusters`, `propose_fix`, `confirm_fix`.

**The autofix loop:** ask your coding assistant to fix your agent — it calls
`propose_fix` (a ready-to-apply briefing: diagnosis, confidence, context,
instructions), applies the fix in your code, re-runs, and reports back with
`confirm_fix`. A confirmed fix is promoted to **verified** and served to every
developer who hits the same failure next.

## Configuration

```python
vorlo_trace.init(
    api_key="vrlo_...",             # or set VORLO_API_KEY env var
    server_url="https://...",       # or set VORLO_SERVER_URL env var
    agent_name="order-processor",   # shown in dashboard
    verify_ssl=True,                 # or set VORLO_VERIFY_SSL=false for local proxy testing
)
```

## Environment Variables

| Variable | Description |
|---|---|
| `VORLO_API_KEY` | Your Vorlo API key |
| `VORLO_SERVER_URL` | Custom server URL (default: production) |
| `VORLO_VERIFY_SSL` | Set to `false` only for local corporate proxy testing |
| `VORLO_DEBUG` | Set to `1` for debug logging to stderr |

## License

MIT — see [LICENSE](LICENSE)

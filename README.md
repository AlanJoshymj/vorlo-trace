# vorlo-trace

> "LangSmith tells you what your agent did. Langfuse shows you where it happened.
> **Vorlo tells you exactly WHY it failed and what to fix — in 30 seconds.**"

AI agent observability SDK. Add 2 lines of code to see exactly which step of your agent failed, the plain-English root cause, and a specific fix hint.

## Quick Start

```bash
pip install vorlo-trace
```

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

- **2-line integration** — works with any LangChain agent
- **Sensor/Actuator classification** — reads vs writes, clearly labeled
- **Cross-step root cause** — understands data flow between steps
- **OTel-compatible** — trace_id, span_id for enterprise export
- **Zero impact** — async sends, daemon threads, silent failures

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

"""
Vorlo Trace — AI agent observability SDK.

Add 2 lines of code to see exactly why your agent failed:

    import vorlo_trace
    vorlo_trace.init(api_key="vrlo_...", agent_name="my-agent")

    # Then pass the handler to your agent:
    agent.invoke({"input": "..."}, config={"callbacks": [vorlo_trace.get_handler()]})

    # Or use the convenience wrapper:
    vorlo_trace.trace(agent, input="...")
"""
from __future__ import annotations

import os
from typing import Any, Callable, Optional

from vorlo_trace.handler import VorloHandler

__version__ = "0.3.0"
__all__ = ["init", "get_handler", "trace", "VorloHandler"]

# Module-level singleton — one handler per process
_handler: Optional[VorloHandler] = None

# Default server URL — points to Vorlo production server
_DEFAULT_SERVER_URL = "https://vorlo-server-production.up.railway.app"


def init(
    api_key: Optional[str] = None,
    server_url: Optional[str] = None,
    agent_name: str = "default",
    verify_ssl: Optional[bool] = None,
    redact: Optional[Callable[[str], str]] = None,
) -> VorloHandler:
    """
    Initialize the Vorlo trace SDK.

    Args:
        api_key: Your Vorlo API key. Falls back to VORLO_API_KEY env var.
        server_url: Vorlo server URL. Falls back to VORLO_SERVER_URL env var,
                    then to the default production URL.
        agent_name: A name for this agent (shown in the dashboard).
        verify_ssl: Whether to verify TLS certificates. Defaults to True.
                Can be overridden with VORLO_VERIFY_SSL=false for local
                corporate proxy testing.
        redact: Optional callback applied to every captured string (tool
                inputs/outputs, errors, reasoning) BEFORE it leaves the
                process — use it to scrub PII or secrets. If it raises, the
                content is dropped rather than shipped raw.

    Returns:
        The VorloHandler instance (also stored as module singleton).

    Raises:
        ValueError: If no API key is provided and VORLO_API_KEY is not set.
    """
    global _handler

    resolved_key = api_key or os.environ.get("VORLO_API_KEY", "")
    if not resolved_key:
        raise ValueError(
            "Vorlo API key is required. Pass api_key='vrlo_...' to init() "
            "or set the VORLO_API_KEY environment variable."
        )

    resolved_url = (
        server_url
        or os.environ.get("VORLO_SERVER_URL", "")
        or _DEFAULT_SERVER_URL
    )
    resolved_verify_ssl = verify_ssl
    if resolved_verify_ssl is None:
        verify_env = os.environ.get("VORLO_VERIFY_SSL", "").lower()
        resolved_verify_ssl = verify_env not in ("0", "false", "no", "off")

    _handler = VorloHandler(
        server_url=resolved_url,
        api_key=resolved_key,
        agent_name=agent_name,
        verify_ssl=resolved_verify_ssl,
        redact=redact,
    )
    return _handler


def get_handler() -> VorloHandler:
    """
    Get the initialized VorloHandler singleton.

    Returns:
        The VorloHandler instance.

    Raises:
        RuntimeError: If init() has not been called yet.
    """
    if _handler is None:
        raise RuntimeError(
            "Vorlo is not initialized. Call vorlo_trace.init(api_key='vrlo_...') first."
        )
    return _handler


def trace(agent: Any, input: Optional[str] = None, **kwargs: Any) -> Any:
    """
    Convenience wrapper — runs an agent with Vorlo tracing enabled.

    Args:
        agent: A LangChain agent (or any object with an `invoke` method).
        input: The input string to pass to the agent.
        **kwargs: Additional keyword arguments passed to agent.invoke().

    Returns:
        The agent's output.

    Raises:
        RuntimeError: If init() has not been called yet.
    """
    handler = get_handler()

    # Build the invoke config with the Vorlo handler in callbacks
    config = kwargs.pop("config", {})
    callbacks = config.get("callbacks", [])
    callbacks.append(handler)
    config["callbacks"] = callbacks

    # Determine the input format — LangChain agents expect a dict
    if input is not None:
        invoke_input: Any = {"input": input}
    else:
        invoke_input = kwargs.pop("invoke_input", {"input": ""})

    return agent.invoke(invoke_input, config=config, **kwargs)

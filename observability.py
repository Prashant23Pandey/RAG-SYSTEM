"""
Langfuse observability integration for the Agentic RAG pipeline.

Provides automatic tracing of every node execution, decision edge,
and prompt payload without modifying core pipeline logic.

Environment variables:
  LANGFUSE_PUBLIC_KEY   – public key from Langfuse cloud/self-hosted
  LANGFUSE_SECRET_KEY   – secret key
  LANGFUSE_HOST         – defaults to https://cloud.langfuse.com
  OLLAMA_BASE_URL       – used for component latency tagging
"""

import os
import time
import functools
import logging
from typing import Any, Callable

logger = logging.getLogger("rag.observability")

# ---------------------------------------------------------------------------
# Lazy-initialised Langfuse client
# ---------------------------------------------------------------------------
_langfuse_client = None


def get_langfuse():
    """Return a singleton Langfuse client (or None if not configured)."""
    global _langfuse_client
    if _langfuse_client is not None:
        return _langfuse_client

    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
    host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")

    if not public_key or not secret_key:
        logger.warning(
            "LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set – "
            "tracing disabled"
        )
        return None

    try:
        from langfuse import Langfuse

        _langfuse_client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
        )
        logger.info("Langfuse client initialised (host=%s)", host)
    except Exception as exc:
        logger.error("Failed to init Langfuse: %s", exc)
        _langfuse_client = None

    return _langfuse_client


def flush():
    client = get_langfuse()
    if client:
        client.flush()


# ---------------------------------------------------------------------------
# Trace wrapper for LangGraph node functions
# ---------------------------------------------------------------------------
def trace_node(node_name: str) -> Callable:
    """
    Decorator that wraps a LangGraph node function with Langfuse tracing.

    Usage:
        @trace_node("retrieve")
        def retrieve_node(state):
            ...
    """

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(state: dict, *args, **kwargs) -> dict:
            client = get_langfuse()
            if client is None:
                return fn(state, *args, **kwargs)

            generation = state.get("generation", "")
            trace_name = f"rag.{node_name}"
            tags = ["rag", node_name]
            metadata = {
                "question": state.get("question", "")[:200],
                "loop_count": state.get("loop_count", 0),
                "web_search_needed": state.get("web_search_needed", False),
            }

            trace = client.trace(
                name=trace_name,
                input={"question": state.get("question", "")[:500]},
                tags=tags,
                metadata=metadata,
            )

            t0 = time.perf_counter()
            try:
                result = fn(state, *args, **kwargs)
                elapsed_ms = (time.perf_counter() - t0) * 1000

                trace.output = {
                    "result_keys": list(result.keys()) if isinstance(result, dict) else [],
                    "documents_count": len(result.get("documents", []))
                    if isinstance(result, dict)
                    else 0,
                }
                trace.update(metadata={"latency_ms": round(elapsed_ms, 1)})

                client.span(
                    trace_id=trace.id,
                    name=f"{node_name}.latency",
                    input={},
                    output={"ms": round(elapsed_ms, 1)},
                    metadata={"component": node_name},
                )

                return result

            except Exception as exc:
                elapsed_ms = (time.perf_counter() - t0) * 1000
                trace.update(
                    level="ERROR",
                    status_message=str(exc),
                    metadata={"latency_ms": round(elapsed_ms, 1)},
                )
                raise

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Generic latency decorator for any callable
# ---------------------------------------------------------------------------
def measure_latency(label: str) -> Callable:
    """
    Decorator that logs latency and token usage for any function.

    Usage:
        @measure_latency("embedding_generation")
        def get_embeddings():
            ...
    """

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs) -> Any:
            client = get_langfuse()
            t0 = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                logger.debug("%s latency: %.1fms", label, elapsed_ms)

                if client:
                    client.span(
                        name=f"perf.{label}",
                        input={},
                        output={"latency_ms": round(elapsed_ms, 1)},
                    )

                return result

            except Exception as exc:
                elapsed_ms = (time.perf_counter() - t0) * 1000
                logger.error("%s failed after %.1fms: %s", label, elapsed_ms, exc)
                raise

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Token counting helper (approximate)
# ---------------------------------------------------------------------------
def log_token_usage(
    trace_id: str,
    prompt_tokens: int,
    completion_tokens: int,
    model: str,
):
    client = get_langfuse()
    if not client:
        return
    client.generation(
        trace_id=trace_id,
        name="token_usage",
        model=model,
        usage={
            "promptTokens": prompt_tokens,
            "completionTokens": completion_tokens,
        },
    )

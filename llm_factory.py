"""
LLM Abstraction Layer — Unified interface for Ollama and vLLM.

Provides a factory function that returns the appropriate LangChain chat model
based on the LLM_BACKEND environment variable.

Environment variables:
  LLM_BACKEND        — "ollama" (default) or "vllm"
  OLLAMA_BASE_URL    — Ollama endpoint (default: http://localhost:11434)
  VLLM_BASE_URL      — vLLM endpoint (default: http://localhost:8000)
  GENERATION_MODEL   — Model name for generation
  GRADING_MODEL      — Model name for grading
  REWRITER_MODEL     — Model name for rewriting
"""

import os
import logging
from typing import Optional

logger = logging.getLogger("rag.llm_factory")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LLM_BACKEND = os.environ.get("LLM_BACKEND", "ollama").lower()
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000")

DEFAULT_MODELS = {
    "ollama": {
        "generation": "llama3.2",
        "grading": "llama3.2",
        "rewriting": "llama3.2",
        "embedding": "nomic-embed-text",
    },
    "vllm": {
        "generation": "meta-llama/Llama-3.2-3B-Instruct",
        "grading": "meta-llama/Llama-3.2-3B-Instruct",
        "rewriting": "meta-llama/Llama-3.2-3B-Instruct",
        "embedding": "nomic-ai/nomic-embed-text-v1",
    },
}


def get_backend() -> str:
    """Return the active LLM backend name."""
    return LLM_BACKEND


# ---------------------------------------------------------------------------
# Ollama LLM
# ---------------------------------------------------------------------------
def _get_ollama_llm(
    model: Optional[str] = None,
    temperature: float = 0,
    base_url: Optional[str] = None,
):
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=model or os.environ.get(
            "GENERATION_MODEL", DEFAULT_MODELS["ollama"]["generation"]
        ),
        base_url=base_url or OLLAMA_BASE_URL,
        temperature=temperature,
    )


# ---------------------------------------------------------------------------
# vLLM LLM (OpenAI-compatible API)
# ---------------------------------------------------------------------------
def _get_vllm_llm(
    model: Optional[str] = None,
    temperature: float = 0,
    base_url: Optional[str] = None,
):
    from langchain_openai import ChatOpenAI

    url = (base_url or VLLM_BASE_URL).rstrip("/")
    if not url.endswith("/v1"):
        url = f"{url}/v1"

    return ChatOpenAI(
        model=model or os.environ.get(
            "GENERATION_MODEL", DEFAULT_MODELS["vllm"]["generation"]
        ),
        openai_api_base=url,
        openai_api_key="not-needed",
        temperature=temperature,
        max_tokens=1024,
    )


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------
def get_llm(
    role: str = "generation",
    temperature: float = 0,
    model_override: Optional[str] = None,
):
    """
    Return a LangChain chat model for the specified role.

    Args:
        role: One of "generation", "grading", "rewriting"
        temperature: Sampling temperature
        model_override: Force a specific model name

    Returns:
        ChatOllama or ChatOpenAI instance (OpenAI-compatible for vLLM)
    """
    backend = LLM_BACKEND

    if model_override:
        model = model_override
    else:
        models = DEFAULT_MODELS.get(backend, DEFAULT_MODELS["ollama"])
        model = models.get(role, models["generation"])

    if backend == "vllm":
        logger.debug("Using vLLM backend — model=%s, role=%s", model, role)
        return _get_vllm_llm(model=model, temperature=temperature)
    else:
        logger.debug("Using Ollama backend — model=%s, role=%s", model, role)
        return _get_ollama_llm(model=model, temperature=temperature)


def get_embedding_model():
    """Return the appropriate embedding model."""
    backend = LLM_BACKEND

    if backend == "vllm":
        from langchain_openai import OpenAIEmbeddings

        url = VLLM_BASE_URL.rstrip("/")
        if not url.endswith("/v1"):
            url = f"{url}/v1"

        return OpenAIEmbeddings(
            model=DEFAULT_MODELS["vllm"]["embedding"],
            openai_api_base=url,
            openai_api_key="not-needed",
        )
    else:
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(
            model=os.environ.get(
                "EMBEDDING_MODEL", DEFAULT_MODELS["ollama"]["embedding"]
            ),
            base_url=OLLAMA_BASE_URL,
        )


def get_generation_llm(temperature: float = 0, model: Optional[str] = None):
    """Convenience: LLM for answer generation."""
    return get_llm(role="generation", temperature=temperature, model_override=model)


def get_grading_llm(temperature: float = 0, model: Optional[str] = None):
    """Convenience: LLM for document grading."""
    return get_llm(role="grading", temperature=temperature, model_override=model)


def get_rewriting_llm(temperature: float = 0, model: Optional[str] = None):
    """Convenience: LLM for query rewriting."""
    return get_llm(role="rewriting", temperature=temperature, model_override=model)

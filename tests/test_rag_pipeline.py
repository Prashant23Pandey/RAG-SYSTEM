"""
Pytest regression suite for the Agentic RAG pipeline.

Tests cover:
  - Unit tests for individual nodes (grade_documents, transform_query)
  - System guardrails (latency, PII detection, hallucination)
  - End-to-end integration tests

Run:
    pytest tests/test_rag_pipeline.py -v
    pytest tests/test_rag_pipeline.py -v -m slow   # skip slow tests
"""

import os
import sys
import re
import json
import time
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------
pytestmark = [
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
]

SLOW = pytest.mark.slow


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_doc(content: str, source: str = "test.txt") -> Document:
    return Document(page_content=content, metadata={"source": source})


PII_PATTERNS = {
    "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "phone": re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"),
}


def _contains_pii(text: str) -> list[str]:
    """Return list of PII types found in text."""
    found = []
    for pii_type, pattern in PII_PATTERNS.items():
        if pattern.search(text):
            found.append(pii_type)
    return found


# ===========================================================================
# UNIT TESTS: grade_documents
# ===========================================================================
class TestGradeDocuments:
    """Unit tests for the document grading node."""

    def test_irrelevant_document_flagged_as_no(self):
        """An irrelevant document should be graded as 'no'."""
        from grader import grade_document

        irrelevant_doc = (
            "The capital of France is Paris, which is known for the Eiffel Tower."
        )
        question = "What is LangGraph used for?"
        result = grade_document(question, irrelevant_doc)
        assert result.get("score", "").lower() == "no"

    def test_relevant_document_flagged_as_yes(self):
        """A relevant document should be graded as 'yes'."""
        from grader import grade_document

        relevant_doc = (
            "LangGraph is a library for building stateful, multi-actor "
            "applications with LLMs, built on top of LangChain."
        )
        question = "What is LangGraph used for?"
        result = grade_document(question, relevant_doc)
        assert result.get("score", "").lower() == "yes"

    def test_grade_documents_filters_correctly(self):
        """grade_documents should return relevance flags for each doc."""
        from grader import grade_documents

        relevant = _make_doc(
            "LangGraph helps build agent workflows with state management."
        )
        irrelevant = _make_doc(
            "The stock market closed at record highs today."
        )
        docs = [relevant, irrelevant]
        results = grade_documents("What is LangGraph?", docs)
        assert len(results) == 2
        assert results[0]["relevant"] is True
        assert results[1]["relevant"] is False

    def test_grade_documents_handles_empty_list(self):
        """Grading an empty document list should return empty."""
        from grader import grade_documents

        results = grade_documents("test question", [])
        assert results == []


# ===========================================================================
# UNIT TESTS: transform_query
# ===========================================================================
class TestTransformQuery:
    """Unit tests for the query rewriting node."""

    def test_rewrite_returns_string(self):
        """rewrite_query should return a non-empty string."""
        from rewriter import rewrite_query

        result = rewrite_query("What is LangGraph?")
        assert isinstance(result, str)
        assert len(result.strip()) > 0

    def test_rewrite_preserves_intent(self):
        """Rewritten query should contain key terms from original."""
        from rewriter import rewrite_query

        result = rewrite_query("How does ChromaDB store embeddings?")
        # Should contain at least one keyword from original
        keywords = ["chromadb", "embed", "vector", "store"]
        assert any(kw in result.lower() for kw in keywords)


# ===========================================================================
# SYSTEM GUARDRAILS
# ===========================================================================
class TestGuardrails:
    """System-level safety and performance tests."""

    def test_latency_under_threshold_vector_query(self):
        """Vector-only queries should complete within 3 seconds."""
        from workflow import compile_workflow

        app = compile_workflow()
        initial_state = {
            "question": "What is LangGraph?",
            "documents": [],
            "generation": "",
            "web_search_needed": False,
            "loop_count": 0,
        }

        t0 = time.perf_counter()
        try:
            result = app.invoke(initial_state)
            elapsed = time.perf_counter() - t0
            # Allow generous timeout for CI (5s), stricter locally (3s)
            threshold = float(os.environ.get("LATENCY_THRESHOLD", "5.0"))
            assert elapsed < threshold, (
                f"Pipeline took {elapsed:.2f}s, exceeds {threshold}s threshold"
            )
        except Exception as exc:
            # If Ollama is not running, skip gracefully
            if "connect" in str(exc).lower() or "refused" in str(exc).lower():
                pytest.skip("Ollama not available – skipping latency test")
            raise

    def test_no_pii_in_generated_answer(self):
        """Generated answers must not contain PII."""
        from workflow import compile_workflow

        app = compile_workflow()
        initial_state = {
            "question": "What is LangGraph?",
            "documents": [],
            "generation": "",
            "web_search_needed": False,
            "loop_count": 0,
        }

        try:
            result = app.invoke(initial_state)
            answer = result.get("generation", "")
            pii_found = _contains_pii(answer)
            assert not pii_found, f"PII detected in answer: {pii_found}"
        except Exception as exc:
            if "connect" in str(exc).lower() or "refused" in str(exc).lower():
                pytest.skip("Ollama not available – skipping PII test")
            raise

    def test_answer_not_empty(self):
        """The pipeline should always produce a non-empty answer."""
        from workflow import compile_workflow

        app = compile_workflow()
        initial_state = {
            "question": "What is LangGraph?",
            "documents": [],
            "generation": "",
            "web_search_needed": False,
            "loop_count": 0,
        }

        try:
            result = app.invoke(initial_state)
            answer = result.get("generation", "")
            assert len(answer.strip()) > 0, "Empty answer returned"
        except Exception as exc:
            if "connect" in str(exc).lower() or "refused" in str(exc).lower():
                pytest.skip("Ollama not available – skipping empty answer test")
            raise


# ===========================================================================
# INTEGRATION TESTS
# ===========================================================================
class TestIntegration:
    """End-to-end integration tests."""

    def test_workflow_compiles(self):
        """The LangGraph workflow should compile without errors."""
        from workflow import compile_workflow

        app = compile_workflow()
        assert app is not None
        assert hasattr(app, "invoke")

    def test_vector_store_queryable(self):
        """ChromaDB should be queryable (may be empty)."""
        from vector_store import query_vectorstore

        try:
            docs = query_vectorstore("test query", k=1)
            assert isinstance(docs, list)
        except Exception as exc:
            if "connect" in str(exc).lower():
                pytest.skip("ChromaDB not available")
            raise

    def test_full_pipeline_end_to_end(self):
        """Full pipeline should run and return a generation."""
        from workflow import compile_workflow

        app = compile_workflow()
        initial_state = {
            "question": "What is LangGraph?",
            "documents": [],
            "generation": "",
            "web_search_needed": False,
            "loop_count": 0,
        }

        try:
            result = app.invoke(initial_state)
            assert "generation" in result
            assert isinstance(result["generation"], str)
        except Exception as exc:
            if "connect" in str(exc).lower() or "refused" in str(exc).lower():
                pytest.skip("Ollama not available – skipping E2E test")
            raise


# ===========================================================================
# RUNNER
# ===========================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

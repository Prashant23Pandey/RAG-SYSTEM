"""
Hybrid Search + Cross-Encoder Reranking Pipeline.

Combines:
  - Dense vector search (ChromaDB + nomic-embed-text)
  - Sparse keyword search (BM25Retriever)
  - Reciprocal Rank Fusion (RRF) to merge results
  - Cross-Encoder reranking (FlashRank or sentence-transformers) for precision
"""

import os
import logging
from typing import Optional

import numpy as np
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.retrievers import BM25Retriever

logger = logging.getLogger("rag.hybrid_search")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")
CHROMA_DIR = os.environ.get(
    "CHROMA_DIR", os.path.join(os.path.dirname(__file__), "chroma_db")
)
COLLECTION_NAME = "rag_documents"

RRF_K = int(os.environ.get("RRF_K", "60"))  # RRF constant (default 60)
VECTOR_TOP_K = int(os.environ.get("VECTOR_TOP_K", "10"))
BM25_TOP_K = int(os.environ.get("BM25_TOP_K", "10"))
RERANK_TOP_N = int(os.environ.get("RERANK_TOP_N", "3"))
RERANKER_MODEL = os.environ.get(
    "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


# ---------------------------------------------------------------------------
# Embeddings helper
# ---------------------------------------------------------------------------
def _get_embeddings():
    return OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)


# ---------------------------------------------------------------------------
# Vector store access
# ---------------------------------------------------------------------------
def _get_vectorstore(persist_dir: str = CHROMA_DIR) -> Chroma:
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=_get_embeddings(),
        collection_name=COLLECTION_NAME,
    )


def _get_all_documents(persist_dir: str = CHROMA_DIR) -> list[Document]:
    """Fetch every document currently in ChromaDB (needed for BM25 index)."""
    vs = _get_vectorstore(persist_dir)
    collection = vs._collection
    results = collection.get(include=["documents", "metadatas"])
    docs = []
    for content, meta in zip(results["documents"], results["metadatas"]):
        docs.append(Document(page_content=content, metadata=meta or {}))
    return docs


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------
def reciprocal_rank_fusion(
    ranked_lists: list[list[Document]],
    k: int = RRF_K,
) -> list[Document]:
    """
    Merge multiple ranked lists using Reciprocal Rank Fusion.

    Args:
        ranked_lists: list of ranked document lists from different retrievers
        k: RRF constant (higher = less weight to top ranks)

    Returns:
        Fused and deduplicated list of documents sorted by RRF score.
    """
    fused_scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list):
            doc_id = doc.page_content[:120]
            score = 1.0 / (k + rank + 1)
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + score
            if doc_id not in doc_map:
                doc_map[doc_id] = doc

    sorted_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
    return [doc_map[did] for did in sorted_ids]


# ---------------------------------------------------------------------------
# Cross-Encoder Reranking
# ---------------------------------------------------------------------------
_reranker = None


def _get_reranker():
    """Lazy-load the cross-encoder reranker."""
    global _reranker
    if _reranker is not None:
        return _reranker

    try:
        from sentence_transformers import CrossEncoder

        _reranker = CrossEncoder(RERANKER_MODEL, max_length=512)
        logger.info("Loaded cross-encoder reranker: %s", RERANKER_MODEL)
    except ImportError:
        logger.warning(
            "sentence-transformers not installed – reranking disabled. "
            "Install with: pip install sentence-transformers"
        )
        _reranker = None

    return _reranker


def rerank_documents(
    query: str,
    documents: list[Document],
    top_n: int = RERANK_TOP_N,
) -> list[Document]:
    """
    Rerank documents using a cross-encoder model.

    Returns the top_n documents sorted by relevance score.
    Falls back to returning the first top_n if reranker unavailable.
    """
    if not documents:
        return []

    reranker = _get_reranker()
    if reranker is None:
        return documents[:top_n]

    pairs = [(query, doc.page_content) for doc in documents]
    scores = reranker.predict(pairs, show_progress_bar=False)

    scored_docs = list(zip(documents, scores.tolist()))
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    reranked = []
    for doc, score in scored_docs[:top_n]:
        doc.metadata["rerank_score"] = round(score, 4)
        reranked.append(doc)

    return reranked


# ---------------------------------------------------------------------------
# BM25 Retriever
# ---------------------------------------------------------------------------
def _build_bm25_retriever(
    documents: list[Document], k: int = BM25_TOP_K
) -> BM25Retriever:
    """Build a BM25 keyword retriever from the document corpus."""
    if not documents:
        return None

    retriever = BM25Retriever.from_documents(documents, k=k)
    return retriever


# ---------------------------------------------------------------------------
# Main hybrid search function
# ---------------------------------------------------------------------------
def hybrid_search(
    query: str,
    persist_dir: str = CHROMA_DIR,
    vector_k: int = VECTOR_TOP_K,
    bm25_k: int = BM25_TOP_K,
    rerank_n: int = RERANK_TOP_N,
    rrf_k: int = RRF_K,
) -> list[Document]:
    """
    Execute hybrid search: Vector + BM25 + RRF + Cross-Encoder Reranking.

    Steps:
      1. Dense vector search (ChromaDB similarity_search)
      2. Sparse BM25 keyword search
      3. Merge with Reciprocal Rank Fusion
      4. Rerank merged results with cross-encoder
      5. Return top_n high-precision chunks with scores

    Returns:
        List of Document objects with rerank_score in metadata.
    """
    # --- Step 1: Vector search ---
    vs = _get_vectorstore(persist_dir)
    vector_results = vs.similarity_search(query, k=vector_k)
    logger.debug("Vector search returned %d results", len(vector_results))

    # --- Step 2: BM25 search ---
    all_docs = _get_all_documents(persist_dir)
    bm25_retriever = _build_bm25_retriever(all_docs, k=bm25_k)
    bm25_results = []
    if bm25_retriever is not None:
        try:
            bm25_results = bm25_retriever.invoke(query)
            logger.debug("BM25 search returned %d results", len(bm25_results))
        except Exception as e:
            logger.warning("BM25 search failed: %s", e)

    # --- Step 3: Reciprocal Rank Fusion ---
    fused = reciprocal_rank_fusion(
        [vector_results, bm25_results], k=rrf_k
    )
    logger.debug("RRF fused %d unique documents", len(fused))

    # --- Step 4: Cross-Encoder Reranking ---
    reranked = rerank_documents(query, fused, top_n=rerank_n)
    logger.debug("Reranked to top %d documents", len(reranked))

    return reranked


# ---------------------------------------------------------------------------
# Convenience: query that returns (docs, scores)
# ---------------------------------------------------------------------------
def hybrid_search_with_scores(
    query: str, **kwargs
) -> list[tuple[Document, float]]:
    """Return (document, score) pairs from hybrid search."""
    docs = hybrid_search(query, **kwargs)
    return [(doc, doc.metadata.get("rerank_score", 0.0)) for doc in docs]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.DEBUG)

    question = sys.argv[1] if len(sys.argv) > 1 else "What is LangGraph?"
    print(f"Query: {question}\n")

    results = hybrid_search(question, rerank_n=5)
    for i, doc in enumerate(results, 1):
        score = doc.metadata.get("rerank_score", "N/A")
        source = doc.metadata.get("source", "unknown")
        print(f"--- Result {i} (score={score}, source={source}) ---")
        print(doc.page_content[:200])
        print()

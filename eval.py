"""
Offline RAG Evaluation Suite using Ragas.

Runs a ground-truth test dataset through the Agentic RAG pipeline and
computes key RAG metrics: Faithfulness, Answer Relevancy, Context Precision,
and Context Recall.

Usage:
    python eval.py                          # run with defaults
    python eval.py --dataset test_dataset.json
    python eval.py --threshold 0.6          # fail if any metric < threshold
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.dirname(__file__))

from workflow import compile_workflow
from vector_store import ingest_directory, get_vectorstore, CHROMA_DIR

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_DATASET = os.path.join(os.path.dirname(__file__), "test_dataset.json")
DEFAULT_THRESHOLD = 0.5
REPORT_DIR = os.path.join(os.path.dirname(__file__), "eval_reports")


# ---------------------------------------------------------------------------
# Run pipeline for a single question and return context + answer
# ---------------------------------------------------------------------------
def run_pipeline(app, question: str) -> dict:
    """Execute the RAG pipeline and return answer + retrieved context."""
    initial_state = {
        "question": question,
        "documents": [],
        "generation": "",
        "web_search_needed": False,
        "loop_count": 0,
    }

    try:
        result = app.invoke(initial_state)
    except Exception as exc:
        return {
            "answer": f"[Pipeline error: {exc}]",
            "contexts": [],
            "source_docs": [],
            "error": str(exc),
        }

    docs = result.get("documents", [])
    contexts = [doc.page_content for doc in docs]
    source_docs = [doc.metadata.get("source", "unknown") for doc in docs]

    return {
        "answer": result.get("generation", ""),
        "contexts": contexts,
        "source_docs": source_docs,
        "error": None,
    }


# ---------------------------------------------------------------------------
# Build Ragas evaluation dataset
# ---------------------------------------------------------------------------
def build_ragas_dataset(test_data: list, pipeline_results: list) -> list:
    """Format data for Ragas evaluation."""
    eval_data = []
    for item, res in zip(test_data, pipeline_results):
        eval_data.append(
            {
                "question": item["question"],
                "answer": res["answer"],
                "contexts": res["contexts"],
                "ground_truth": item["ground_truth_answer"],
                "source_docs": item.get("expected_source_docs", []),
                "retrieved_docs": res["source_docs"],
            }
        )
    return eval_data


# ---------------------------------------------------------------------------
# Metric computations (Ragas-style, computed locally)
# ---------------------------------------------------------------------------
def _llm_as_judge(question: str, answer: str, context: str = "") -> float:
    """Use the local Ollama LLM as a judge. Returns 0.0-1.0 score."""
    from langchain_ollama import ChatOllama
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.environ.get("GENERATION_MODEL", "llama3.2")

    llm = ChatOllama(model=model, base_url=base_url, temperature=0)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an evaluation judge. Rate the quality on a scale of "
                "0.0 to 1.0. Output ONLY a number between 0.0 and 1.0.",
            ),
            (
                "human",
                "{question}\n{answer}\n{context}",
            ),
        ]
    )
    chain = prompt | llm | StrOutputParser()
    try:
        raw = chain.invoke(
            {"question": question, "answer": answer, "context": context}
        )
        score = float(raw.strip())
        return max(0.0, min(1.0, score))
    except Exception:
        return 0.0


def compute_faithfulness(answer: str, contexts: list[str]) -> float:
    """
    Faithfulness: can the answer be derived from the provided contexts?
    Checks each claim in the answer against the contexts.
    """
    if not contexts:
        return 0.0
    context_blob = "\n".join(contexts)
    score = _llm_as_judge(
        "Is the following answer fully supported by the context? "
        "Score 1.0 if fully supported, 0.0 if completely hallucinated.",
        answer,
        context_blob,
    )
    return score


def compute_answer_relevancy(question: str, answer: str) -> float:
    """Answer Relevancy: does the answer address the question?"""
    score = _llm_as_judge(
        "Does this answer directly address the question? "
        "Score 1.0 if perfectly relevant, 0.0 if irrelevant.",
        answer,
        question,
    )
    return score


def compute_context_precision(
    question: str, ground_truth: str, contexts: list[str]
) -> float:
    """Context Precision: are the retrieved contexts relevant to the question?"""
    if not contexts:
        return 0.0
    context_blob = "\n".join(contexts)
    score = _llm_as_judge(
        f"Are these contexts relevant to answering this question? "
        f"Ground truth answer: {ground_truth}",
        context_blob,
        question,
    )
    return score


def compute_context_recall(ground_truth: str, contexts: list[str]) -> float:
    """Context Recall: do the contexts contain information from the ground truth?"""
    if not contexts:
        return 0.0
    context_blob = "\n".join(contexts)
    score = _llm_as_judge(
        "Does the context contain the information needed to support this "
        "ground truth answer? Score 1.0 if fully covered.",
        ground_truth,
        context_blob,
    )
    return score


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------
def run_evaluation(
    dataset_path: str = DEFAULT_DATASET,
    threshold: float = DEFAULT_THRESHOLD,
    ingest_sample: bool = True,
) -> dict:
    """Run the full evaluation pipeline and return results."""
    print("=" * 70)
    print("  RAG Evaluation Suite (Ragas-style)")
    print("=" * 70)

    # Load test dataset
    print(f"\n[1] Loading test dataset from: {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    print(f"    Loaded {len(test_data)} test cases")

    # Optionally ingest sample docs
    if ingest_sample:
        sample_dir = os.path.join(os.path.dirname(__file__), "sample_docs")
        if os.path.isdir(sample_dir):
            print("\n[2] Ensuring documents are ingested...")
            try:
                ingest_directory(sample_dir)
            except Exception as exc:
                print(f"    Ingestion skipped: {exc}")

    # Compile workflow
    print("\n[3] Compiling workflow...")
    app = compile_workflow()

    # Run pipeline on each question
    print("\n[4] Running pipeline on test cases...")
    pipeline_results = []
    for i, item in enumerate(test_data, 1):
        q = item["question"]
        print(f"    [{i}/{len(test_data)}] {q[:60]}...")
        t0 = time.perf_counter()
        res = run_pipeline(app, q)
        elapsed = time.perf_counter() - t0
        res["latency_s"] = round(elapsed, 2)
        pipeline_results.append(res)
        if res["error"]:
            print(f"           ERROR: {res['error']}")
        else:
            print(f"           OK ({elapsed:.2f}s)")

    # Compute metrics per test case
    print("\n[5] Computing evaluation metrics...")
    scores = []
    for item, res in zip(test_data, pipeline_results):
        question = item["question"]
        answer = res["answer"]
        contexts = res["contexts"]
        ground_truth = item["ground_truth_answer"]

        faithfulness = compute_faithfulness(answer, contexts)
        relevancy = compute_answer_relevancy(question, answer)
        precision = compute_context_precision(question, ground_truth, contexts)
        recall = compute_context_recall(ground_truth, contexts)

        entry = {
            "question": question,
            "category": item.get("category", "unknown"),
            "latency_s": res["latency_s"],
            "error": res["error"],
            "metrics": {
                "faithfulness": round(faithfulness, 3),
                "answer_relevancy": round(relevancy, 3),
                "context_precision": round(precision, 3),
                "context_recall": round(recall, 3),
            },
            "source_docs": res.get("source_docs", []),
            "expected_docs": item.get("expected_source_docs", []),
        }
        scores.append(entry)

    # Aggregate
    metric_keys = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    averages = {}
    for key in metric_keys:
        vals = [s["metrics"][key] for s in scores]
        averages[key] = round(sum(vals) / len(vals), 3) if vals else 0.0

    # Identify failures
    failures = [
        s for s in scores
        if any(s["metrics"][k] < threshold for k in metric_keys)
        or s["error"]
    ]

    # Report
    report = {
        "dataset": dataset_path,
        "total_cases": len(test_data),
        "threshold": threshold,
        "averages": averages,
        "failures_count": len(failures),
        "cases": scores,
    }

    # Print summary
    print("\n" + "=" * 70)
    print("  EVALUATION RESULTS")
    print("=" * 70)
    for key in metric_keys:
        val = averages[key]
        status = "PASS" if val >= threshold else "FAIL"
        print(f"  {key:<22s}: {val:.3f}  [{status}] (threshold={threshold})")
    print(f"\n  Failures: {len(failures)}/{len(test_data)}")
    if failures:
        print("\n  Failed test cases:")
        for f in failures:
            print(f"    - {f['question'][:70]}")
            print(f"      Metrics: {f['metrics']}")
    print("=" * 70)

    # Save report
    os.makedirs(REPORT_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(REPORT_DIR, f"eval_report_{ts}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n  Report saved to: {report_path}")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="RAG Evaluation Suite")
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="Path to test_dataset.json",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Minimum metric score to pass (default: 0.5)",
    )
    parser.add_argument(
        "--no-ingest",
        action="store_true",
        help="Skip document ingestion step",
    )
    args = parser.parse_args()

    report = run_evaluation(
        dataset_path=args.dataset,
        threshold=args.threshold,
        ingest_sample=not args.no_ingest,
    )

    # Exit with non-zero if any failures
    if report["failures_count"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

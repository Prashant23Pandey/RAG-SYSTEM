"""
Developer CLI for the Agentic RAG system.

Commands:
    ingest   — Ingest documents into ChromaDB
    query    — Run a test query and show agent trajectory
    clear-db — Clear the vector database
    status   — Show system status and health

Usage:
    python rag_cli.py ingest --path ./docs
    python rag_cli.py query "What is LangGraph?"
    python rag_cli.py clear-db
    python rag_cli.py status
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Optional

try:
    import typer
    from typer import Typer
except ImportError:
    print("Error: typer is required. Install with: pip install typer")
    sys.exit(1)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Typer(
    name="rag-cli",
    help="Developer CLI for the Agentic RAG system.",
    add_completion=False,
)

# Color helpers
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _info(msg: str):
    print(f"{CYAN}[INFO]{RESET} {msg}")


def _success(msg: str):
    print(f"{GREEN}[OK]{RESET} {msg}")


def _warn(msg: str):
    print(f"{YELLOW}[WARN]{RESET} {msg}")


def _error(msg: str):
    print(f"{RED}[ERROR]{RESET} {msg}")


# ---------------------------------------------------------------------------
# INGEST
# ---------------------------------------------------------------------------
@app.command()
def ingest(
    path: str = typer.Argument(..., help="File or directory to ingest"),
    tenant_id: str = typer.Option("default", "--tenant", "-t", help="Tenant ID"),
    persist_dir: Optional[str] = typer.Option(
        None, "--persist-dir", "-d", help="ChromaDB directory"
    ),
):
    """Ingest documents into the ChromaDB vector store."""
    from ingest import ingest_file, ingest_directory, SUPPORTED_EXTENSIONS

    target = os.path.abspath(path)

    if not os.path.exists(target):
        _error(f"Path not found: {target}")
        raise typer.Exit(1)

    chroma_dir = persist_dir or os.environ.get(
        "CHROMA_DIR", os.path.join(os.path.dirname(__file__), "chroma_db")
    )

    _info(f"Target: {target}")
    _info(f"ChromaDB: {chroma_dir}")
    _info(f"Tenant: {tenant_id}")
    print()

    t0 = time.perf_counter()

    if os.path.isdir(target):
        results = ingest_directory(target, chroma_dir, tenant_id)
        total = sum(r["chunks_added"] for r in results)
        _success(f"Ingested {len(results)} file(s), {total} chunks total")
    elif os.path.isfile(target):
        result = ingest_file(target, chroma_dir, tenant_id)
        _success(
            f"Ingested {result['filename']} — {result['chunks_added']} chunks"
        )
    else:
        _error(f"Not a valid file or directory: {target}")
        raise typer.Exit(1)

    elapsed = time.perf_counter() - t0
    _info(f"Completed in {elapsed:.2f}s")


# ---------------------------------------------------------------------------
# QUERY
# ---------------------------------------------------------------------------
@app.command()
def query(
    question: str = typer.Argument(..., help="Question to ask the RAG system"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show full trace"),
):
    """Run a query and display the agent trajectory."""
    from workflow import compile_workflow

    _info(f"Question: {question}")
    _info("Compiling workflow...")
    app = compile_workflow()

    initial_state = {
        "question": question,
        "documents": [],
        "generation": "",
        "web_search_needed": False,
        "loop_count": 0,
        "guard_flag": "",
    }

    print()
    _info("Running pipeline...")
    print()

    t0 = time.perf_counter()
    steps = []
    final_answer = ""

    try:
        for step_event in app.stream(initial_state, stream_mode="updates"):
            for node_name, update in step_event.items():
                if node_name == "__end__":
                    continue

                elapsed = (time.perf_counter() - t0) * 1000
                step_info = _format_step(node_name, update)
                steps.append((node_name, step_info, elapsed))

                print(f"  {BOLD}[{node_name}]{RESET} {step_info} ({elapsed:.0f}ms)")

                if node_name == "generate":
                    final_answer = update.get("generation", "")

    except Exception as exc:
        _error(f"Pipeline error: {exc}")
        raise typer.Exit(1)

    total_time = time.perf_counter() - t0

    # Print answer
    print()
    print(f"  {BOLD}{'='*60}{RESET}")
    print(f"  {BOLD}ANSWER{RESET}")
    print(f"  {'='*60}")
    print(f"  {final_answer}")
    print(f"  {BOLD}{'='*60}{RESET}")
    print()

    # Summary
    web_search_used = any(s[0] == "web_search" for s in steps)
    docs_retrieved = 0
    for name, info, _ in steps:
        if name == "retrieve" and "reranked" in info:
            try:
                docs_retrieved = int(info.split()[2])
            except (IndexError, ValueError):
                pass

    print(f"  {CYAN}Summary:{RESET}")
    print(f"    Steps executed:  {len(steps)}")
    print(f"    Documents used:  {docs_retrieved}")
    print(f"    Web search:      {'Yes' if web_search_used else 'No'}")
    print(f"    Total time:      {total_time:.2f}s")
    print()

    if verbose:
        print(f"  {BOLD}Full Trajectory:{RESET}")
        for i, (name, info, t) in enumerate(steps, 1):
            print(f"    {i}. [{name}] {info} ({t:.0f}ms)")
        print()


def _format_step(node_name: str, update: dict) -> str:
    if node_name == "input_guard":
        flag = update.get("guard_flag", "")
        if flag:
            return f"BLOCKED: {flag}"
        return "Input cleared"
    if node_name == "retrieve":
        docs = update.get("documents", [])
        return f"{len(docs)} reranked chunk(s) returned"
    if node_name == "grade_documents":
        needed = update.get("web_search_needed", False)
        if needed:
            return "No relevant documents — web search flagged"
        docs = update.get("documents", [])
        return f"{len(docs)} relevant document(s)"
    if node_name == "transform_query":
        q = update.get("question", "")
        return f"Rewritten: {q[:80]}"
    if node_name == "web_search":
        docs = update.get("documents", [])
        return f"{len(docs)} web result(s)"
    if node_name == "generate":
        gen = update.get("generation", "")
        return f"Generated {len(gen)} chars"
    return str(update)[:100]


# ---------------------------------------------------------------------------
# CLEAR-DB
# ---------------------------------------------------------------------------
@app.command(name="clear-db")
def clear_db(
    persist_dir: Optional[str] = typer.Option(
        None, "--persist-dir", "-d", help="ChromaDB directory"
    ),
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
):
    """Clear the ChromaDB vector store."""
    chroma_dir = persist_dir or os.environ.get(
        "CHROMA_DIR", os.path.join(os.path.dirname(__file__), "chroma_db")
    )

    if not os.path.isdir(chroma_dir):
        _warn(f"ChromaDB directory not found: {chroma_dir}")
        raise typer.Exit(0)

    if not confirm:
        typer.confirm(
            f"This will delete all data in {chroma_dir}. Continue?",
            abort=True,
        )

    import shutil

    try:
        shutil.rmtree(chroma_dir)
        _success(f"Cleared ChromaDB at: {chroma_dir}")
    except Exception as exc:
        _error(f"Failed to clear database: {exc}")
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# STATUS
# ---------------------------------------------------------------------------
@app.command()
def status():
    """Show system status and component health."""
    print(f"\n  {BOLD}Agentic RAG — System Status{RESET}\n")

    # Check Ollama
    ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        import requests

        resp = requests.get(f"{ollama_url}/api/tags", timeout=5)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            model_names = [m["name"] for m in models]
            _success(f"Ollama connected ({ollama_url})")
            _info(f"Models: {', '.join(model_names) or 'none'}")
        else:
            _error(f"Ollama returned status {resp.status_code}")
    except Exception:
        _error(f"Ollama not reachable at {ollama_url}")

    # Check ChromaDB
    chroma_dir = os.environ.get(
        "CHROMA_DIR", os.path.join(os.path.dirname(__file__), "chroma_db")
    )
    if os.path.isdir(chroma_dir):
        _success(f"ChromaDB directory exists: {chroma_dir}")
        try:
            from vector_store import get_vectorstore

            vs = get_vectorstore(chroma_dir)
            count = vs._collection.count()
            _info(f"Documents in store: {count}")
        except Exception as e:
            _warn(f"Could not read ChromaDB: {e}")
    else:
        _warn(f"ChromaDB directory not found: {chroma_dir}")

    # Check Langfuse
    langfuse_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    if langfuse_key:
        _success("Langfuse configured")
    else:
        _warn("Langfuse not configured (set LANGFUSE_PUBLIC_KEY)")

    # Check reranker
    try:
        from sentence_transformers import CrossEncoder

        model_name = os.environ.get(
            "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        _success(f"Reranker available: {model_name}")
    except ImportError:
        _warn("sentence-transformers not installed — reranking disabled")

    # Check guardrails
    input_guard = os.environ.get("INPUT_GUARD", "true").lower() == "true"
    output_guard = os.environ.get("OUTPUT_GUARD", "true").lower() == "true"
    pii_mask = os.environ.get("PII_MASK", "true").lower() == "true"
    _info(
        f"Guardrails: input={input_guard}, output={output_guard}, pii={pii_mask}"
    )

    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app()

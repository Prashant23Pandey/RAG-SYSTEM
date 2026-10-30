import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from workflow import compile_workflow
from vector_store import ingest_directory


def setup_sample_documents():
    sample_dir = os.path.join(os.path.dirname(__file__), "sample_docs")
    os.makedirs(sample_dir, exist_ok=True)

    langgraph_file = os.path.join(sample_dir, "langgraph_info.txt")
    if not os.path.exists(langgraph_file):
        with open(langgraph_file, "w", encoding="utf-8") as f:
            f.write(
                "LangGraph is a library for building stateful, multi-actor applications "
                "with LLMs, built on top of LangChain. It extends LangChain expression "
                "language with the ability to coordinate multiple chains (or actors) across "
                "multiple steps of computation in a cyclic manner. This is particularly "
                "useful for building agent workflows where different components need to "
                "make decisions based on the current state of the system.\n\n"
                "Key features of LangGraph include:\n"
                "- State management across multiple nodes\n"
                "- Conditional edges and branching logic\n"
                "- Support for human-in-the-loop workflows\n"
                "- Cyclic graph execution for iterative processes\n"
                "- Integration with LangSmith for observability"
            )

    chromadb_file = os.path.join(sample_dir, "chromadb_info.txt")
    if not os.path.exists(chromadb_file):
        with open(chromadb_file, "w", encoding="utf-8") as f:
            f.write(
                "ChromaDB is an open-source embedding database that makes it easy to "
                "build AI applications with embeddings. It provides fast, accurate vector "
                "search capabilities.\n\n"
                "ChromaDB supports:\n"
                "- Multiple embedding functions including OpenAI and local models\n"
                "- Persistent and in-memory storage options\n"
                "- Metadata filtering for precise queries\n"
                "- Easy integration with LangChain and other frameworks\n\n"
                "To use ChromaDB with LangChain, you can use the Chroma class from "
                "langchain_community.vectorstores."
            )

    ollama_file = os.path.join(sample_dir, "ollama_info.txt")
    if not os.path.exists(ollama_file):
        with open(ollama_file, "w", encoding="utf-8") as f:
            f.write(
                "Ollama allows you to run large language models locally on your machine. "
                "It supports a wide range of models including Llama 3.2, Mistral, and many "
                "others.\n\n"
                "Ollama provides:\n"
                "- Simple API for model interaction\n"
                "- Local model hosting for privacy\n"
                "- Embedding models like nomic-embed-text\n"
                "- Easy installation and setup\n\n"
                "To install Ollama, visit https://ollama.com and follow the installation "
                "instructions for your operating system."
            )

    return sample_dir


def run_test(app, question: str, test_name: str):
    print("\n" + "=" * 60)
    print(f"  TEST: {test_name}")
    print("=" * 60)
    print(f"Question: {question}")
    print("-" * 60)

    initial_state = {
        "question": question,
        "documents": [],
        "generation": "",
        "web_search_needed": False,
        "loop_count": 0,
    }

    try:
        result = app.invoke(initial_state)

        print("\n" + "-" * 60)
        print("FINAL RESULT:")
        print("-" * 60)
        print(f"Question: {result.get('question', 'N/A')}")
        print(f"\nGeneration:\n{result.get('generation', 'N/A')}")
        print(f"\nDocuments used: {len(result.get('documents', []))}")
        print(f"Web search was needed: {result.get('web_search_needed', False)}")
    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()


def main():
    print("Agentic RAG System - End-to-End Test")
    print("=" * 60)

    print("\n[1] Setting up sample documents...")
    sample_dir = setup_sample_documents()

    print("\n[2] Ingesting documents into vector store...")
    try:
        ingest_directory(sample_dir)
    except Exception as e:
        print(f"Error ingesting documents: {e}")
        print("Make sure Ollama is running with: ollama serve")
        return

    print("\n[3] Compiling workflow...")
    try:
        app = compile_workflow()
    except Exception as e:
        print(f"Error compiling workflow: {e}")
        return

    print("\n[4] Running test queries...")

    in_domain_query = "What is LangGraph and what are its key features?"
    run_test(app, in_domain_query, "In-Domain Query (Vector Store)")

    out_of_domain_query = "What are the latest developments in quantum computing in 2026?"
    run_test(app, out_of_domain_query, "Out-of-Domain Query (Web Search Fallback)")

    print("\n" + "=" * 60)
    print("  ALL TESTS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

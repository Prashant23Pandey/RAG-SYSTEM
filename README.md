<div align="center">

# :brain: Agentic RAG

**A production-grade, self-hosted Retrieval-Augmented Generation system with autonomous agent routing, hybrid search, and full observability.**

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-00bfff?style=for-the-badge)](LICENSE)
[![Build Status](https://img.shields.io/badge/Build-Passing-brightgreen?style=for-the-badge)](https://github.com/yourusername/agentic-rag/actions)
[![Code Style: Black](https://img.shields.io/badge/Code%20Style-Black-000000?style=for-the-badge)](https://github.com/psf/black)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-7b2ff7?style=for-the-badge&logo=langchain&logoColor=white)](https://github.com/langchain-ai/langgraph)

```
  ____                    _ ____    ____ _     ___ _   _ _____
 |  _ \ __ _ _ __   __ _| | __ )  / ___| |   |_ _| \ | | ____|
 | |_) / _` | '_ \ / _` | |  _ \| |   | |    | ||  \| |  _|
 |  _ < (_| | | | | (_| | | |_) | |___| |___ | || |\  | |___
 |_| \_\__,_|_| |_|\__,_|_|____/ \____|_____|___|_| \_|_____|
```

</div>

---

## :zap: Highlights

- ** :arrows_counterclockwise ** **Autonomous Agent Loop** — Self-correcting RAG with automatic query rewriting and web search fallback
- ** ** **Hybrid Search** — BM25 + ChromaDB vector search fused with Reciprocal Rank Fusion (RRF)
- ** :shield: **Production Guardrails** — Input injection detection, PII masking, output hallucination checking
- ** :mag: **Cross-Encoder Reranking** — Local `ms-marco-MiniLM-L-6-v2` reranker for high-precision retrieval
- ** :bar_chart: **Full Observability** — Langfuse tracing on every node execution, latency, and token usage
- ** :lock: **100% Local & Private** — Zero external API calls. Ollama + ChromaDB. Your data never leaves your machine.
- ** :test_tube: **Evaluation Suite** — Ragas-style faithfulness, relevancy, precision, and recall metrics

---

## :building_construction: Architecture

```
                          +-----------------+
                          |    User Input   |
                          +--------+--------+
                                   |
                          +--------v--------+
                          |   Input Guard   |  <-- PII Masking + Injection Detection
                          +--------+--------+
                                   |
                 +-----------------v------------------+
                 |         LangGraph StateGraph        |
                 |                                     |
                 |  +---------+    +----------------+  |
                 |  |Retrieve |--->|Grade Documents |  |
                 |  |(Hybrid)|    +--------+-------+  |
                 |  +----+----+             |          |
                 |       |           [relevant?]       |
                 |       |          /         \        |
                 |       |        YES          NO      |
                 |       |         |            |      |
                 |  +----v----+    |    +-------v-----+|
                 |  |Generate |<---+    |Transform    ||
                 |  +----+----+         |Query        ||
                 |       |              +-------+-----+|
                 |       |                      |      |
                 |  +----v----+           +-----v----+ |
                 |  | Output  |           |Web Search| |
                 |  | Guard   |           +-----+----+ |
                 |  +----+----+                 |      |
                 +-------+---------------------+------+
                         |
              +----------v----------+
              |                     |
     +--------v--------+  +--------v--------+
     |     Ollama      |  |    ChromaDB     |
     |  (llama3.2 +    |  |  (nomic-embed + |
     |  nomic-embed)   |  |   BM25 + RRF)  |
     +-----------------+  +-----------------+
              |                     |
     +--------v--------+  +--------v--------+
     |   FastAPI       |  |   Langfuse      |
     |   :8000         |  |   :3000         |
     +--------+--------+  +-----------------+
              |
     +--------v--------+
     |   Streamlit     |
     |   :8501         |
     +-----------------+
```

### Pipeline Flow

```
User Query
    │
    ▼
[Input Guard] ─── PII masking, injection detection, toxicity check
    │
    ▼
[Hybrid Retrieve] ─── ChromaDB (dense) + BM25 (sparse) → RRF → Cross-Encoder Rerank
    │
    ▼
[Grade Documents] ─── LLM judges each chunk for relevance
    │
    ├─ [Relevant] ────────────────────────────┐
    │                                          ▼
    │                                     [Generate]
    │                                          │
    │                                     [Output Guard]
    │                                          │
    │                                     Final Answer
    │
    └─ [Not Relevant] ──► [Transform Query] ──► [Web Search] ──► [Generate]
```

---

## :rocket: Quick Start

### Option 1: Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/agentic-rag.git
cd agentic-rag

# Start all services
docker compose up --build

# Wait for models to download (~2GB first time)
# Then open:
#   Streamlit UI:  http://localhost:8501
#   FastAPI docs:  http://localhost:8000/docs
#   Langfuse:      http://localhost:3000
```

### Option 2: Local Setup

```bash
# Clone and create virtual environment
git clone https://github.com/yourusername/agentic-rag.git
cd agentic-rag
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Start Ollama (in a separate terminal)
ollama serve
ollama pull llama3.2
ollama pull nomic-embed-text

# Ingest sample documents
python main.py

# Start the API server
python app.py

# Start the Streamlit UI (in another terminal)
streamlit run ui.py
```

### Option 3: CLI

```bash
# Ingest documents
python rag_cli.py ingest --path ./my_documents

# Query directly from terminal
python rag_cli.py query "What is LangGraph?"

# Clear the vector database
python rag_cli.py clear-db
```

---

## :wrench: Configuration

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API endpoint |
| `GENERATION_MODEL` | `llama3.2` | LLM for answer generation |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model for vectors |
| `CHROMA_DIR` | `./chroma_db` | ChromaDB persistence directory |
| `MAX_REWRITE_LOOPS` | `2` | Max query rewrite attempts |
| `VECTOR_TOP_K` | `10` | Vector search result count |
| `BM25_TOP_K` | `10` | BM25 search result count |
| `RERANK_TOP_N` | `3` | Final reranked result count |
| `RRF_K` | `60` | Reciprocal Rank Fusion constant |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder reranker |
| `INPUT_GUARD` | `true` | Enable input safety guard |
| `OUTPUT_GUARD` | `true` | Enable output hallucination check |
| `PII_MASK` | `true` | Enable PII redaction |
| `LANGFUSE_PUBLIC_KEY` | *(empty)* | Langfuse public key |
| `LANGFUSE_SECRET_KEY` | *(empty)* | Langfuse secret key |
| `LANGFUSE_HOST` | `http://localhost:3000` | Langfuse dashboard URL |

---

## :package: Project Structure

```
agentic-rag/
├── app.py                  # FastAPI backend (REST API + SSE streaming)
├── nodes.py                # LangGraph node functions (retrieve, grade, generate...)
├── workflow.py             # LangGraph state graph definition
├── hybrid_search.py        # BM25 + Vector + RRF + Cross-Encoder reranking
├── grader.py               # LLM-based document relevance grading
├── rewriter.py             # Query rewriting for search optimization
├── web_search.py           # DuckDuckGo web search fallback
├── guardrails.py           # Input/output safety + PII sanitization
├── vector_store.py         # ChromaDB vector store management
├── observability.py        # Langfuse tracing integration
├── ingest.py               # Multi-format document ingestion ETL
├── ui.py                   # Streamlit interactive dashboard
├── eval.py                 # Ragas-style evaluation suite
├── rag_cli.py              # Developer CLI tool
├── main.py                 # Quick-start script (ingest + test)
├── test_dataset.json       # Ground-truth evaluation data
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container image definition
├── docker-compose.yml      # Multi-service orchestration
├── .env.example            # Environment variable template
├── CONTRIBUTING.md         # Contribution guidelines
├── LICENSE                 # MIT License
├── DEPLOYMENT.md           # Phase 10 deployment guide
├── tests/
│   └── test_rag_pipeline.py
├── .github/
│   └── workflows/
│       └── rag_ci.yml      # CI/CD pipeline
├── eval_reports/           # Generated evaluation reports
│
│   # Phase 9: Multi-Agent & Fine-Tuning
├── multi_agent.py          # Supervisor → Researcher/Code/Synthesizer subgraphs
├── finetune.py             # QLoRA fine-tuning with Unsloth
├── llm_factory.py          # Ollama/vLLM abstraction layer
├── vllm_server.py          # vLLM deployment utilities
├── benchmark.py            # Async RPS benchmarking
│
│   # Phase 10: Cloud & Edge Deployment
├── k8s_deploy.py           # Kubernetes deployment script
├── helm/                   # Helm chart for K8s
│   └── rag-system/
├── serverless/             # Serverless GPU deployments
│   ├── modal_serverless.py # Modal deployment
│   └── runpod_handler.py   # RunPod handler
├── edge/                   # Edge/On-Device RAG
│   ├── export_onnx.py      # ONNX model exporter
│   └── edge-rag.js         # Browser RAG implementation
```

---

## :test_tube: Testing

```bash
# Run unit tests
pytest tests/ -v -k "TestGradeDocuments or TestTransformQuery"

# Run guardrail tests
pytest tests/ -v -k "TestGuardrails"

# Run full test suite
pytest tests/ -v

# Run evaluation suite
python eval.py --threshold 0.5

# Run evaluation without re-ingesting
python eval.py --no-ingest --threshold 0.6
```

---

## :bar_chart: Evaluation Metrics

The evaluation suite computes 4 key RAG metrics:

| Metric | Description | Target |
|--------|-------------|--------|
| **Faithfulness** | Is the answer grounded in retrieved context? | >= 0.7 |
| **Answer Relevancy** | Does the answer address the question? | >= 0.7 |
| **Context Precision** | Are retrieved chunks relevant to the question? | >= 0.6 |
| **Context Recall** | Do chunks contain the ground-truth information? | >= 0.6 |

---

## :handshake: Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## :scroll: License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## :raising_hand: Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) — LLM orchestration framework
- [LangGraph](https://github.com/langchain-ai/langgraph) — Stateful agent graphs
- [Ollama](https://ollama.com) — Local LLM runtime
- [ChromaDB](https://github.com/chroma-core/chroma) — Embedding database
- [Langfuse](https://langfuse.com) — Open-source observability
- [Ragas](https://github.com/explodinggradients/ragas) — RAG evaluation framework
- [Sentence Transformers](https://www.sbert.net) — Cross-encoder reranking

---

<div align="center">

**Built with :heart: for the open-source AI community**

</div>

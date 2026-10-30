import os
from typing import TypedDict, Annotated, List
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from operator import add

from hybrid_search import hybrid_search
from grader import grade_documents
from rewriter import rewrite_query
from web_search import web_search
from guardrails import run_input_guards, run_output_guard, sanitize_for_external_use
from observability import trace_node, measure_latency, flush

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
GENERATION_MODEL = os.environ.get("GENERATION_MODEL", "llama3.2")
MAX_REWRITE_LOOPS = int(os.environ.get("MAX_REWRITE_LOOPS", "2"))


class GraphState(TypedDict):
    question: str
    documents: Annotated[List[Document], add]
    generation: str
    web_search_needed: bool
    loop_count: int
    guard_flag: str


@trace_node("input_guard")
def input_guard_node(state: GraphState) -> dict:
    """Run input safety guards before processing."""
    print("---INPUT GUARD---")
    question = state["question"]

    sanitized, result = run_input_guards(question)

    if not result.passed:
        print(f"---INPUT BLOCKED: {result.reason}---")
        return {
            "guard_flag": f"BLOCKED: {result.reason}",
            "generation": (
                "I'm sorry, but I cannot process this request. "
                f"Reason: {result.reason}"
            ),
        }

    print(f"---INPUT CLEARED (PII masked if present)---")
    return {"question": sanitized, "guard_flag": ""}


@trace_node("retrieve")
def retrieve_node(state: GraphState) -> dict:
    """Hybrid search: Vector + BM25 + RRF + Cross-Encoder Reranking."""
    print("---RETRIEVE (Hybrid)---")
    question = state["question"]
    documents = hybrid_search(
        question,
        vector_k=10,
        bm25_k=10,
        rerank_n=5,
    )
    print(f"---Retrieved {len(documents)} reranked chunk(s)---")
    return {"documents": documents}


@trace_node("grade_documents")
def grade_documents_node(state: GraphState) -> dict:
    print("---GRADE DOCUMENTS---")
    question = state["question"]
    documents = state["documents"]

    relevant_docs = grade_documents(question, documents)
    filtered = [item["document"] for item in relevant_docs if item["relevant"]]

    if len(filtered) == 0:
        print("---NO RELEVANT DOCUMENTS FOUND---")
        return {"web_search_needed": True}
    else:
        print(f"---{len(filtered)} RELEVANT DOCUMENT(S) FOUND---")
        return {"documents": filtered, "web_search_needed": False}


@trace_node("transform_query")
def transform_query_node(state: GraphState) -> dict:
    print("---TRANSFORM QUERY---")
    question = state["question"]
    loop_count = state.get("loop_count", 0)

    if loop_count >= MAX_REWRITE_LOOPS:
        print(f"---MAX REWRITE LOOPS ({MAX_REWRITE_LOOPS}) REACHED---")
        return {"web_search_needed": True, "loop_count": loop_count}

    new_question = rewrite_query(question)
    print(f"Original: {question}")
    print(f"Rewritten: {new_question}")
    return {"question": new_question, "loop_count": loop_count + 1}


@trace_node("web_search")
def web_search_node(state: GraphState) -> dict:
    print("---WEB SEARCH---")
    question = state["question"]
    sanitized_q = sanitize_for_external_use(question)
    web_docs = web_search(sanitized_q, max_results=3)
    return {"documents": web_docs}


@trace_node("generate")
def generate_node(state: GraphState) -> dict:
    """Generate answer with output guard."""
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    llm = ChatOllama(
        model=GENERATION_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Use the provided context to answer "
                "the user's question. If the context does not contain enough "
                "information, say so honestly. Be concise and accurate. "
                "Never fabricate information not present in the context.",
            ),
            (
                "human",
                "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:",
            ),
        ]
    )

    context = "\n\n".join([doc.page_content for doc in documents])
    chain = prompt | llm | StrOutputParser()
    generation = chain.invoke({"context": context, "question": question})

    # Output guard
    contexts = [doc.page_content for doc in documents]
    guard_result = run_output_guard(question, generation, contexts)
    if not guard_result.passed:
        print(f"---OUTPUT GUARD FAILED: {guard_result.reason}---")
        generation = (
            "I'm sorry, but I'm unable to provide a reliable answer to this "
            "question based on the available information. Please try "
            "rephrasing your question."
        )

    flush()
    return {"generation": generation}


def decide_next_step(state: GraphState) -> str:
    print("---DECIDE NEXT STEP---")
    # Check if input guard blocked the request
    if state.get("guard_flag", "").startswith("BLOCKED"):
        return "generate"
    web_search_needed = state.get("web_search_needed", False)
    if web_search_needed:
        return "web_search"
    else:
        return "generate"

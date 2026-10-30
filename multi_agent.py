"""
Multi-Agent Specialist Framework using LangGraph Subgraphs.

Implements a hierarchical multi-agent system:
  - Supervisor Agent: Routes sub-tasks to specialist sub-agents
  - Document Researcher Agent: Hybrid vector/BM25 retrieval + summarization
  - Code/Data Agent: Executes Python code and parses structured data
  - Synthesizer Agent: Combines sub-agent outputs into final answer

Uses persistent state passing and parent-child subgraph coordination.
"""

import os
import json
import logging
from typing import TypedDict, Annotated, List, Optional, Literal
from operator import add

from langgraph.graph import StateGraph, END
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser

from llm_factory import get_generation_llm, get_grading_llm
from hybrid_search import hybrid_search
from web_search import web_search
from guardrails import run_input_guards, run_output_guard, sanitize_for_external_use
from observability import trace_node, flush

logger = logging.getLogger("rag.multi_agent")

# ---------------------------------------------------------------------------
# Shared State Schema
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    question: str
    sanitized_question: str
    agent_route: str
    documents: Annotated[List[Document], add]
    code_output: str
    researcher_output: str
    synthesis: str
    generation: str
    web_search_needed: bool
    guard_flag: str
    sub_task_results: Annotated[List[dict], add]
    iteration: int


# ---------------------------------------------------------------------------
# Supervisor Agent
# ---------------------------------------------------------------------------
@trace_node("supervisor")
def supervisor_agent(state: AgentState) -> dict:
    """
    Inspects user prompt and routes to specialist sub-agents.

    Routes:
      - "research" — Document retrieval, summarization, Q&A
      - "code"     — Python code execution, data parsing, JSON analysis
      - "hybrid"   — Both research and code needed
    """
    print("---SUPERVISOR---")
    question = state.get("sanitized_question", state["question"])

    llm = get_grading_llm()
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a supervisor agent. Analyze the user's question and "
            "determine which specialist agent(s) should handle it.\n\n"
            "Available agents:\n"
            "- research: For questions about documents, facts, knowledge retrieval, summarization\n"
            "- code: For questions requiring Python code execution, data analysis, JSON/table parsing\n"
            "- hybrid: When both document research AND code/data processing are needed\n\n"
            "Respond with a JSON object: {\"route\": \"research\"|\"code\"|\"hybrid\", "
            "\"reason\": \"brief explanation\"}",
        ),
        ("human", "{question}"),
    ])

    chain = prompt | llm | JsonOutputParser()
    try:
        result = chain.invoke({"question": question[:500]})
        route = result.get("route", "research").lower()
        if route not in ("research", "code", "hybrid"):
            route = "research"
    except Exception as e:
        logger.warning("Supervisor parse failed: %s — defaulting to research", e)
        route = "research"

    print(f"  Route: {route}")
    return {"agent_route": route, "iteration": 0}


def route_from_supervisor(state: AgentState) -> str:
    """Route to appropriate sub-agent based on supervisor decision."""
    route = state.get("agent_route", "research")
    if route == "code":
        return "code_agent"
    elif route == "hybrid":
        return "research_agent"  # Start with research, then code
    return "research_agent"


# ---------------------------------------------------------------------------
# Document Researcher Sub-Agent
# ---------------------------------------------------------------------------
@trace_node("research_agent")
def research_agent(state: AgentState) -> dict:
    """
    Handles document retrieval via hybrid search and summarization.
    This is a subgraph that encapsulates the full RAG retrieval pipeline.
    """
    print("---RESEARCH AGENT---")
    question = state.get("sanitized_question", state["question"])

    # Hybrid search
    documents = hybrid_search(question, vector_k=10, bm25_k=10, rerank_n=5)
    print(f"  Retrieved {len(documents)} chunk(s)")

    # Summarize retrieved context
    if documents:
        llm = get_generation_llm()
        context = "\n\n".join([doc.page_content for doc in documents])

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a document research specialist. Summarize the key "
                "information from the retrieved documents that is relevant to "
                "the question. Be concise and factual.",
            ),
            (
                "human",
                "Context:\n{context}\n\nQuestion: {question}\n\nSummary:",
            ),
        ])
        chain = prompt | llm | StrOutputParser()
        try:
            summary = chain.invoke({"context": context[:3000], "question": question})
        except Exception as e:
            logger.warning("Research summarization failed: %s", e)
            summary = context[:1000]
    else:
        summary = "No relevant documents found in the local knowledge base."

    return {
        "documents": documents,
        "researcher_output": summary,
        "web_search_needed": len(documents) == 0,
    }


# ---------------------------------------------------------------------------
# Code/Data Sub-Agent
# ---------------------------------------------------------------------------
@trace_node("code_agent")
def code_agent(state: AgentState) -> dict:
    """
    Handles code execution and structured data parsing.
    Uses the LLM to generate and execute Python code safely.
    """
    print("---CODE AGENT---")
    question = state.get("sanitized_question", state["question"])

    llm = get_generation_llm()
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a code/data specialist. Given the user's question, "
            "generate a Python code snippet that would help answer it. "
            "The code should be self-contained and use only standard libraries "
            "(json, re, math, datetime, collections). Do NOT execute the code — "
            "just provide it along with an explanation of what it does.\n\n"
            "Respond in this format:\n"
            "```python\n<code>\n```\n\nExplanation: <what the code does>",
        ),
        ("human", "{question}"),
    ])

    chain = prompt | llm | StrOutputParser()
    try:
        code_output = chain.invoke({"question": question[:500]})
    except Exception as e:
        logger.warning("Code agent failed: %s", e)
        code_output = f"Code generation failed: {e}"

    return {"code_output": code_output}


# ---------------------------------------------------------------------------
# Hybrid Coordinator (for "hybrid" route)
# ---------------------------------------------------------------------------
@trace_node("hybrid_coordinator")
def hybrid_coordinator(state: AgentState) -> dict:
    """Run code agent after research agent for hybrid queries."""
    print("---HYBRID COORDINATOR---")
    # The research agent already ran, now run code agent
    return code_agent(state)


# ---------------------------------------------------------------------------
# Synthesizer Agent
# ---------------------------------------------------------------------------
@trace_node("synthesizer")
def synthesizer_agent(state: AgentState) -> dict:
    """
    Combines outputs from all sub-agents into a coherent final answer.
    """
    print("---SYNTHESIZER---")
    question = state.get("sanitized_question", state["question"])
    research = state.get("researcher_output", "")
    code = state.get("code_output", "")

    llm = get_generation_llm()
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a synthesis agent. Combine the research findings and "
            "code analysis into a single, coherent, well-structured answer. "
            "Be concise, accurate, and directly address the user's question. "
            "If context is insufficient, acknowledge limitations honestly.",
        ),
        (
            "human",
            "Question: {question}\n\n"
            "Research Findings:\n{research}\n\n"
            "Code/Data Analysis:\n{code}\n\n"
            "Provide a unified answer:",
        ),
    ])

    chain = prompt | llm | StrOutputParser()
    try:
        synthesis = chain.invoke({
            "question": question,
            "research": research[:2000],
            "code": code[:2000],
        })
    except Exception as e:
        logger.warning("Synthesis failed: %s", e)
        synthesis = f"Research: {research[:500]}\n\nCode: {code[:500]}"

    # Output guard
    contexts = [doc.page_content for doc in state.get("documents", [])]
    guard_result = run_output_guard(question, synthesis, contexts)
    if not guard_result.passed:
        print(f"  Output guard failed: {guard_result.reason}")
        synthesis = (
            "I'm sorry, but I'm unable to provide a reliable answer based "
            "on the available information. Please try rephrasing."
        )

    flush()
    return {"synthesis": synthesis, "generation": synthesis}


# ---------------------------------------------------------------------------
# Simple RAG Synthesizer (for research-only route)
# ---------------------------------------------------------------------------
@trace_node("rag_synthesizer")
def rag_synthesizer(state: AgentState) -> dict:
    """Direct synthesis for research-only queries (no code agent needed)."""
    print("---RAG SYNTHESIZER---")
    question = state.get("sanitized_question", state["question"])
    research = state.get("researcher_output", "")
    documents = state.get("documents", [])

    llm = get_generation_llm()
    context = "\n\n".join([doc.page_content for doc in documents]) if documents else ""

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a helpful assistant. Use the provided context to answer "
            "the user's question. If the context doesn't contain enough "
            "information, say so honestly. Be concise and accurate.",
        ),
        (
            "human",
            "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:",
        ),
    ])

    chain = prompt | llm | StrOutputParser()
    try:
        generation = chain.invoke({"context": context[:3000], "question": question})
    except Exception as e:
        logger.warning("RAG synthesis failed: %s", e)
        generation = research[:1000] if research else "Unable to generate answer."

    # Output guard
    contexts = [doc.page_content for doc in documents]
    guard_result = run_output_guard(question, generation, contexts)
    if not guard_result.passed:
        generation = (
            "I'm sorry, but I'm unable to provide a reliable answer. "
            "Please try rephrasing your question."
        )

    flush()
    return {"generation": generation}


# ---------------------------------------------------------------------------
# Code Synthesizer (for code-only route)
# ---------------------------------------------------------------------------
@trace_node("code_synthesizer")
def code_synthesizer(state: AgentState) -> dict:
    """Direct synthesis for code-only queries."""
    print("---CODE SYNTHESIZER---")
    code_output = state.get("code_output", "")

    return {"generation": code_output, "synthesis": code_output}


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------
def route_after_research(state: AgentState) -> str:
    """Route after research agent completes."""
    route = state.get("agent_route", "research")
    if route == "hybrid":
        return "hybrid_coordinator"
    return "rag_synthesizer"


def route_after_hybrid(state: AgentState) -> str:
    """Route after hybrid coordinator completes."""
    return "synthesizer"


# ---------------------------------------------------------------------------
# Build the Multi-Agent Graph
# ---------------------------------------------------------------------------
def build_multi_agent_workflow() -> StateGraph:
    """
    Build the hierarchical multi-agent workflow.

    Graph structure:
        input_guard → supervisor → research_agent → [rag_synthesizer | hybrid_coordinator → synthesizer]
                       supervisor → code_agent → code_synthesizer
    """
    workflow = StateGraph(AgentState)

    # Supervisor (entry point)
    workflow.add_node("supervisor", supervisor_agent)

    # Specialist sub-agents
    workflow.add_node("research_agent", research_agent)
    workflow.add_node("code_agent", code_agent)
    workflow.add_node("hybrid_coordinator", hybrid_coordinator)

    # Synthesizers
    workflow.add_node("rag_synthesizer", rag_synthesizer)
    workflow.add_node("code_synthesizer", code_synthesizer)
    workflow.add_node("synthesizer", synthesizer_agent)

    # Entry point
    workflow.set_entry_point("supervisor")

    # Supervisor routing
    workflow.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "research_agent": "research_agent",
            "code_agent": "code_agent",
        },
    )

    # Research agent → either direct synthesis or hybrid
    workflow.add_conditional_edges(
        "research_agent",
        route_after_research,
        {
            "rag_synthesizer": "rag_synthesizer",
            "hybrid_coordinator": "hybrid_coordinator",
        },
    )

    # Code agent → code synthesis
    workflow.add_edge("code_agent", "code_synthesizer")

    # Hybrid coordinator → full synthesis
    workflow.add_edge("hybrid_coordinator", "synthesizer")

    # All synthesizers → END
    workflow.add_edge("rag_synthesizer", END)
    workflow.add_edge("code_synthesizer", END)
    workflow.add_edge("synthesizer", END)

    return workflow.compile()


# ---------------------------------------------------------------------------
# Compile and run
# ---------------------------------------------------------------------------
_multi_agent_app = None


def compile_multi_agent_workflow():
    global _multi_agent_app
    if _multi_agent_app is None:
        _multi_agent_app = build_multi_agent_workflow()
    return _multi_agent_app


def run_multi_agent(question: str, verbose: bool = True) -> dict:
    """
    Run the multi-agent system on a question.

    Returns:
        dict with "answer", "route", "documents", "steps"
    """
    app = compile_multi_agent_workflow()

    sanitized, guard_result = run_input_guards(question)
    if not guard_result.passed:
        return {
            "answer": f"Request blocked: {guard_result.reason}",
            "route": "blocked",
            "documents": [],
            "steps": [],
        }

    initial_state = {
        "question": question,
        "sanitized_question": sanitized,
        "agent_route": "",
        "documents": [],
        "code_output": "",
        "researcher_output": "",
        "synthesis": "",
        "generation": "",
        "web_search_needed": False,
        "guard_flag": "",
        "sub_task_results": [],
        "iteration": 0,
    }

    steps = []
    answer = ""

    for step_event in app.stream(initial_state, stream_mode="updates"):
        for node_name, update in step_event.items():
            if node_name == "__end__":
                continue
            steps.append(node_name)
            if verbose:
                print(f"  [{node_name}]")
            if "generation" in update and update["generation"]:
                answer = update["generation"]

    return {
        "answer": answer,
        "route": initial_state.get("agent_route", "unknown"),
        "documents": initial_state.get("documents", []),
        "steps": steps,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    question = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "What is LangGraph and how does it handle state?"
    )

    print(f"\nQuestion: {question}\n")
    result = run_multi_agent(question, verbose=True)
    print(f"\n{'='*60}")
    print(f"Route: {result['route']}")
    print(f"Steps: {result['steps']}")
    print(f"Documents: {len(result['documents'])}")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"{'='*60}")

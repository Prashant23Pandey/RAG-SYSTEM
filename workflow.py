from langgraph.graph import StateGraph, END
from nodes import (
    GraphState,
    input_guard_node,
    retrieve_node,
    grade_documents_node,
    transform_query_node,
    web_search_node,
    generate_node,
    decide_next_step,
)


def build_workflow() -> StateGraph:
    workflow = StateGraph(GraphState)

    # Phase 7: Input guard is the entry point
    workflow.add_node("input_guard", input_guard_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("transform_query", transform_query_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("generate", generate_node)

    workflow.set_entry_point("input_guard")

    # If input guard blocks, skip straight to generate (returns refusal)
    workflow.add_conditional_edges(
        "input_guard",
        lambda state: "generate"
        if state.get("guard_flag", "").startswith("BLOCKED")
        else "retrieve",
        {
            "generate": "generate",
            "retrieve": "retrieve",
        },
    )

    workflow.add_edge("retrieve", "grade_documents")

    workflow.add_conditional_edges(
        "grade_documents",
        lambda state: "transform_query"
        if state.get("web_search_needed", False)
        else "generate",
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )

    workflow.add_conditional_edges(
        "transform_query",
        lambda state: "web_search"
        if state.get("web_search_needed", False)
        else "retrieve",
        {
            "web_search": "web_search",
            "retrieve": "retrieve",
        },
    )

    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()


def compile_workflow():
    app = build_workflow()
    return app


if __name__ == "__main__":
    app = compile_workflow()
    print("Workflow compiled successfully!")
    print("\nGraph structure:")
    print(app.get_graph().draw_ascii())

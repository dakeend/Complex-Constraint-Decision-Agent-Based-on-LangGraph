from langgraph.graph import END, START, StateGraph

from .nodes import (
    check_missing_info,
    generate_answer,
    parse_query,
    retrieve_products,
    score_candidates,
)
from .state import AgentState


def should_ask_clarification(state: AgentState) -> str:
    if state.get("missing_info"):
        return "end"
    return "retrieve_products"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("parse_query", parse_query)
    graph.add_node("check_missing_info", check_missing_info)
    graph.add_node("retrieve_products", retrieve_products)
    graph.add_node("score_candidates", score_candidates)
    graph.add_node("generate_answer", generate_answer)

    graph.add_edge(START, "parse_query")
    graph.add_edge("parse_query", "check_missing_info")
    graph.add_conditional_edges(
        "check_missing_info",
        should_ask_clarification,
        {
            "end": END,
            "retrieve_products": "retrieve_products",
        },
    )
    graph.add_edge("retrieve_products", "score_candidates")
    graph.add_edge("score_candidates", "generate_answer")
    graph.add_edge("generate_answer", END)
    return graph.compile()

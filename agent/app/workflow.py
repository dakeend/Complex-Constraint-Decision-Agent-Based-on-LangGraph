"""项目计划完整流程 LangGraph 编排"""
from langgraph.graph import END, START, StateGraph
import os

from .nodes import (
    route_public_search_platform,
    deep_search,
    dual_path_retrieval,
    evidence_extraction,
    fallback_similar,
    generate_final_output,
    initial_screen_and_consistency,
    parse_query,
    preference_recall_and_rewrite,
    pdd_mapping,
    web_search_agent_zhihu,
    web_search_agent_xiaohongshu,
    web_search_agent_tieba,
    web_search_agent_weibo,
    web_search_agent_general,
)
from .state import AgentState


def should_deep_search(state: AgentState) -> str:
    """结果不一致或证据不足时二次深搜"""
    if state.get("need_deep_search"):
        return "deep_search"
    return "pdd_mapping"


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("parse_query", parse_query)
    graph.add_node("preference_recall", preference_recall_and_rewrite)
    graph.add_node("search_zhihu", web_search_agent_zhihu)
    graph.add_node("search_xiaohongshu", web_search_agent_xiaohongshu)
    graph.add_node("search_tieba", web_search_agent_tieba)
    graph.add_node("search_weibo", web_search_agent_weibo)
    graph.add_node("search_general", web_search_agent_general)
    graph.add_node("dual_retrieval", dual_path_retrieval)
    graph.add_node("evidence_extraction", evidence_extraction)
    graph.add_node("initial_screen", initial_screen_and_consistency)
    graph.add_node("deep_search", deep_search)
    graph.add_node("pdd_mapping", pdd_mapping)
    graph.add_node("fallback", fallback_similar)
    graph.add_node("generate_output", generate_final_output)

    graph.add_edge(START, "parse_query")
    graph.add_edge("parse_query", "preference_recall")
    graph.add_conditional_edges(
        "preference_recall",
        route_public_search_platform,
        {
            "zhihu": "search_zhihu",
            "xiaohongshu": "search_xiaohongshu",
            "tieba": "search_tieba",
            "weibo": "search_weibo",
            "general": "search_general",
        },
    )
    graph.add_edge("search_zhihu", "dual_retrieval")
    graph.add_edge("search_xiaohongshu", "dual_retrieval")
    graph.add_edge("search_tieba", "dual_retrieval")
    graph.add_edge("search_weibo", "dual_retrieval")
    graph.add_edge("search_general", "dual_retrieval")
    graph.add_edge("dual_retrieval", "evidence_extraction")
    graph.add_edge("evidence_extraction", "initial_screen")
    graph.add_conditional_edges(
        "initial_screen",
        should_deep_search,
        {"deep_search": "deep_search", "pdd_mapping": "pdd_mapping"},
    )
    graph.add_edge("deep_search", "pdd_mapping")
    graph.add_edge("pdd_mapping", "fallback")
    graph.add_edge("fallback", "generate_output")
    graph.add_edge("generate_output", END)

    return graph.compile()

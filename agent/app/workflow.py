"""项目计划完整流程 LangGraph 编排"""
from langgraph.graph import END, START, StateGraph
import os

from .nodes import (
    summarize_public_search,
    extract_llm_recommendations,
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
    graph.add_node("summarize_search", summarize_public_search)
    graph.add_node("extract_llm", extract_llm_recommendations)
    graph.add_node("evidence_extraction", evidence_extraction)
    graph.add_node("initial_screen", initial_screen_and_consistency)
    graph.add_node("deep_search", deep_search)
    graph.add_node("pdd_mapping", pdd_mapping)
    graph.add_node("fallback", fallback_similar)
    graph.add_node("generate_output", generate_final_output)

    graph.add_edge(START, "parse_query")
    graph.add_edge("parse_query", "preference_recall")
    # 并行触发所有平台搜索节点（每个平台节点内部用 LLM 并行压缩），最后汇总再进入后续检索/评价
    graph.add_edge("preference_recall", "search_zhihu")
    graph.add_edge("preference_recall", "search_xiaohongshu")
    graph.add_edge("preference_recall", "search_tieba")
    graph.add_edge("preference_recall", "search_weibo")
    graph.add_edge("preference_recall", "search_general")
    graph.add_edge("search_zhihu", "dual_retrieval")
    graph.add_edge("search_xiaohongshu", "dual_retrieval")
    graph.add_edge("search_tieba", "dual_retrieval")
    graph.add_edge("search_weibo", "dual_retrieval")
    graph.add_edge("search_general", "dual_retrieval")
    graph.add_edge("dual_retrieval", "summarize_search")
    graph.add_edge("summarize_search", "extract_llm")
    graph.add_edge("extract_llm", "evidence_extraction")
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

"""Agent 状态定义，支持项目计划中的完整流程"""
from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    # -1. 对话上下文（用于总控 Agent 做上下文理解）
    #   结构示例：[{"role": "user" | "assistant", "content": "..."}, ...]
    history: list[dict[str, Any]]

    # 0. 总控 Agent 输出（需求充分性判断 + 加工后的需求）
    processed_requirement: str  # 充分时由 LLM 加工后的需求描述，供下游节点使用

    # 1. 总控 Agent
    query_text: str
    extracted_constraints: dict[str, Any]
    missing_info: list[str]
    clarifying_questions: list[str]
    info_sufficient: bool

    # 2. 偏好召回与 Query 重写
    user_preferences: dict[str, Any]
    rewritten_query: str

    # 3. 双路检索
    content_search_results: list[dict[str, Any]]
    review_search_results: list[dict[str, Any]]

    # 4. 证据抽取与候选商品
    extracted_evidence: list[dict[str, Any]]
    candidate_products: list[dict[str, Any]]
    normalized_candidates: list[dict[str, Any]]

    # 5. 初筛排序与一致性
    scored_candidates: list[dict[str, Any]]
    is_consistent: bool
    need_deep_search: bool

    # 6. Top 候选二次深搜
    deep_search_evidence: list[dict[str, Any]]

    # 7. 多多进宝映射
    pdd_mapped_products: list[dict[str, Any]]
    has_exact_match: bool

    # 8. 兜底
    fallback_products: list[dict[str, Any]]

    # 9. 最终输出
    final_recommendation: dict[str, Any] | None
    recommendation_reason: list[str]
    risk_explanations: list[str]
    purchase_links: list[dict[str, str]]
    candidates: list[dict[str, Any]]

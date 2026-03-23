"""Agent 状态定义，支持项目计划中的完整流程"""
from typing import Any, TypedDict, Annotated
import operator


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
    # 并行搜索节点会同时写入该字段；用 reducer 做 list 累加合并，避免互相覆盖
    content_search_results: Annotated[list[dict[str, Any]], operator.add]
    # 每个平台节点对搜索结果做 LLM 压缩后的摘要（并行累加）
    platform_summaries: Annotated[list[dict[str, Any]], operator.add]
    review_search_results: list[dict[str, Any]]
    # LLM 从 web_search_summary 解析出的推荐商品列表（按推荐顺序），作为主推荐依据
    llm_recommended_products: list[str]
    # 推荐商品及推荐原因 [{name, reason}, ...]
    llm_recommended_with_reasons: list[dict[str, Any]]

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
    # 深度检索 Top5 全部结果 [{name, reason, purchase_url}, ...]
    top5_recommendations: list[dict[str, Any]]

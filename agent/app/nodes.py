"""项目计划完整流程节点"""
import json
from pathlib import Path
import os
from typing import Optional

from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from .state import AgentState
from .services.content_search import search_public_content
from .services.evidence import extract_evidence, deduplicate_candidates
from .services.pdd_client import (
    search_goods,
    map_candidates_to_pdd,
    get_promotion_links,
)
from .services.preferences import recall_preferences, rewrite_query
from .services.quality_evaluation import (
    PLATFORM_WEIGHTS,
    compute_cross_platform_scores,
    build_product_snippets,
    llm_quality_score,
    compute_combined_score,
    deep_search_for_candidates,
    llm_final_rerank,
)


DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "products.json"
os.environ.setdefault("TAVILY_API_KEY", os.getenv("TAVILY_API_KEY", ""))
model = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.environ["GROQ_API_KEY"],
)
search_tool=TavilySearch(result=5)
tools=[search_tool]

class RequirementCheckResult(BaseModel):
    """总控 Agent：需求充分性判断的结构化输出"""
    sufficient: bool = Field(description="用户需求是否已充分，足以进行商品推荐")
    clarifying_questions: Optional[list[str]] = Field(
        default=None,
        description="当需求不充分时，需要向用户追问的问题列表；充分时可为空",
    )
    processed_requirement: Optional[str] = Field(
        default=None,
        description="当需求充分时，对用户需求加工后的摘要（保留预算、场景、偏好等关键约束），供下游检索与推荐使用；不充分时可为空",
    )

class WebSearchResult(BaseModel):
    """LLM 根据搜索结果整合后的输出"""
    source_from: str = Field(description="搜索来源，如知乎、小红书")
    search_result: str = Field(description="对搜索结果进行整合，推荐前五的商品或结论")


class ParsedConstraintsResult(BaseModel):
    """需求结构化抽取结果"""
    budget_min: int | None = Field(default=None, description="预算下限（元）")
    budget_max: int | None = Field(default=None, description="预算上限（元）")
    usage_scenarios: list[str] = Field(default_factory=list, description="使用场景列表")
    brand_preference: list[str] = Field(default_factory=list, description="品牌偏好")
    brand_avoid: list[str] = Field(default_factory=list, description="排除品牌")
    need_portability: bool = Field(default=False, description="是否强调便携/轻薄")
    need_dedicated_gpu: bool = Field(default=False, description="是否强调独显/游戏性能")
    keyword: str = Field(default="笔记本", description="商品关键词/品类")


# 总控 Agent 专用：不绑定工具，只做充分性判断 + 需求加工
orchestrator_llm = model.with_structured_output(RequirementCheckResult)
# 仅用于「先搜再答」：根据搜索结果文本做整合，不绑定 search tool
web_search_summary_llm = model.with_structured_output(WebSearchResult)
constraint_parser_llm = model.with_structured_output(ParsedConstraintsResult)


ORCHESTRATOR_PROMPT = """你是一个商品推荐流程的总控 Agent。

你可以看到用户与系统之前的多轮对话历史，请在理解当前轮需求时综合考虑这些上下文，而不是仅看当前一句话。

对话历史（从早到晚）：
{history}

当前用户输入：
{query}

请仅根据「综合后的当前需求」判断其是否「充分」——即是否足以开始做推荐（例如至少能推断出品类/预算/场景中的一部分）。

判断标准：
- 充分：能推断出大致品类（如笔记本、洗面奶）且至少有一项约束（预算、场景、偏好、排除等），可据此进行检索与推荐。
- 不充分：完全无法推断品类，或信息过于模糊无法形成有效检索条件。

请严格按以下规则输出：
1. 若判断为「充分」：将 sufficient 设为 true，在 processed_requirement 中写出一段加工后的需求描述（保留预算、使用场景、偏好、排除项等关键信息，便于下游检索）；clarifying_questions 可为空列表或 null。
2. 若判断为「不充分」：将 sufficient 设为 false，在 clarifying_questions 中列出需要用户补充的问题（如「您的预算大概多少？」「主要用途是办公、编程还是游戏？」）；processed_requirement 可为 null。
"""

CONSTRAINT_PARSER_PROMPT = """你是商品需求结构化抽取器。请将用户需求转为结构化约束字段。

对话历史（从早到晚）：
{history}

用户最终需求文本：
{query}

抽取规则：
1. keyword：优先识别真实品类；若无法判断则给「笔记本」。
2. budget_min/budget_max：把「5000左右/大概5000/5000以内/最多5000」等口语金额都解析为数值（单位元）。
3. usage_scenarios：尽量抽取，如办公、编程、游戏、设计、学习、出差便携等；可多选。
4. brand_preference / brand_avoid：根据偏好词和排除词抽取品牌。
5. need_portability：有“轻薄/便携/出差带着走”等诉求时为 true，否则 false。
6. need_dedicated_gpu：有“独显/游戏/3D建模/大型渲染”等诉求时为 true，否则 false。

只输出结构化字段，不要补充解释。
"""


def controlbot(state: AgentState) -> AgentState:
    """
    总控 Agent：用大模型判断需求是否充分，不调用任何工具。
    - 充分：产出加工后的需求 processed_requirement，供下游节点使用。
    - 不充分：产出需要补充的问题 clarifying_questions，供追问并结束。
    """
    query = state["query_text"]
    history = state.get("history", [])
    # 只截取最近若干轮，避免 prompt 过长
    recent_history = history[-8:] if isinstance(history, list) else []
    history_text_lines = []
    for m in recent_history:
        role = m.get("role", "user")
        content = m.get("content", "")
        history_text_lines.append(f"{role}: {content}")
    history_text = "\n".join(history_text_lines) or "（无历史对话，当前为第一轮）"

    prompt = ORCHESTRATOR_PROMPT.format(history=history_text, query=query)
    result = orchestrator_llm.invoke(prompt)

    return {
        "info_sufficient": result.sufficient,
        "clarifying_questions": result.clarifying_questions or [],
        "processed_requirement": result.processed_requirement or "",
        "missing_info": [] if result.sufficient else ["requirement_check"],
    }

def _web_search_agent_impl(
    state: AgentState,
    platform_key: str,
    platform_label: str,
    prompt_instruction: str,
) -> AgentState:
    """
    通用 Web 搜索 Agent 实现：按平台搜索 → 拼成文本 → LLM 整合为「推荐前五」。
    platform_key: content_search 的 platform 参数（zhihu/xiaohongshu/tieba/weibo/general）
    platform_label: 展示用名称（知乎/小红书/百度贴吧/微博/综合多平台）
    prompt_instruction: 填入 prompt 的整合说明，如「根据以下知乎搜索结果…」
    """
    query = _get_working_query(state)
    raw_results = search_public_content(query, platform=platform_key)
    search_list = raw_results[:10]

    search_text = "\n\n".join(
        f"[{i+1}] {r.get('title', '')}\n{r.get('snippet', '')}"
        for i, r in enumerate(search_list)
    )
    prompt = f"""{prompt_instruction}

    用户需求：{query}

    搜索结果：
    {search_text or '（暂无结果）'}

    请填写 source_from 为「{platform_label}」，在 search_result 中写出整合后的推荐结论（前五名或主要推荐）。"""
    summary = web_search_summary_llm.invoke(prompt)

    return {
        "content_search_results": search_list,
        "web_search_summary": summary.search_result,
        "web_search_source": summary.source_from,
    }


def web_search_agent_zhihu(state: AgentState) -> AgentState:
    """
    搜索 Agent（限定知乎）：在知乎内搜索，再把搜索结果送给 LLM 整合成「推荐前五」。
    """
    return _web_search_agent_impl(
        state,
        platform_key="zhihu",
        platform_label="知乎",
        prompt_instruction="根据以下知乎搜索结果，整合并给出「推荐前五」的商品或结论（与用户需求相关），字符长度每个商品限定在100字。",
    )


def web_search_agent_xiaohongshu(state: AgentState) -> AgentState:
    """
    搜索 Agent（限定小红书）：在小红书内搜索，再把搜索结果送给 LLM 整合成「推荐前五」。
    """
    return _web_search_agent_impl(
        state,
        platform_key="xiaohongshu",
        platform_label="小红书",
        prompt_instruction="根据以下小红书搜索结果，整合并给出「推荐前五」的商品或结论（与用户需求相关），字符长度每个商品限定在100字。",
    )


def web_search_agent_tieba(state: AgentState) -> AgentState:
    """
    搜索 Agent（限定百度贴吧）：在百度贴吧内搜索，再把搜索结果送给 LLM 整合成「推荐前五」。
    """
    return _web_search_agent_impl(
        state,
        platform_key="tieba",
        platform_label="百度贴吧",
        prompt_instruction="根据以下百度贴吧搜索结果，整合并给出「推荐前五」的商品或结论（与用户需求相关），字符长度每个商品限定在100字。",
    )


def web_search_agent_weibo(state: AgentState) -> AgentState:
    """
    搜索 Agent（限定微博）：在微博内搜索，再把搜索结果送给 LLM 整合成「推荐前五」。
    """
    return _web_search_agent_impl(
        state,
        platform_key="weibo",
        platform_label="微博",
        prompt_instruction="根据以下微博搜索结果，整合并给出「推荐前五」的商品或结论（与用户需求相关），字符长度每个商品限定在100字。",
    )


def web_search_agent_general(state: AgentState) -> AgentState:
    """
    搜索 Agent（不限定条件）：在多平台（知乎、小红书、贴吧、微博等）综合搜索，
    再把搜索结果送给 LLM 整合成「推荐前五」。
    """
    return _web_search_agent_impl(
        state,
        platform_key="general",
        platform_label="综合多平台",
        prompt_instruction="根据以下多平台（知乎、小红书、百度贴吧、微博等）搜索结果，整合并给出「推荐前五」的商品或结论（与用户需求相关），字符长度每个商品限定在100字。",
    )


def rout_ask(state: AgentState) -> str:
    """
    根据总控 Agent 的充分性判断做路由（对应流程图「需求信息是否充分」）：
    - 充分 → 将加工后的需求传给下一节点执行（continue）
    - 不充分 → 返回需要补充的内容，结束（clarify）
    """
    if state.get("info_sufficient"):
        return "continue"
    return "clarify"


def route_public_search_platform(state: AgentState) -> str:
    """
    公开内容搜索路由：
    - 若用户明确限定平台（知乎/小红书/贴吧/微博），走对应搜索节点
    - 若未限定平台，走综合搜索节点（general）
    """
    query = _get_working_query(state)
    query_lower = query.lower()

    if "知乎" in query or "zhihu" in query_lower:
        return "zhihu"
    if "小红书" in query or "xiaohongshu" in query_lower or "rednote" in query_lower:
        return "xiaohongshu"
    if "贴吧" in query or "tieba" in query_lower:
        return "tieba"
    if "微博" in query or "weibo" in query_lower:
        return "weibo"
    return "general"



def _local_products_fallback(constraints: dict) -> list[dict]:
    """无 PDD 时使用本地 products.json 兜底"""
    products = _load_products()
    out = []
    for p in products:
        if constraints.get("brand_avoid") and p.get("brand") in constraints["brand_avoid"]:
            continue
        if constraints.get("budget_max") and p.get("price", 0) > constraints["budget_max"]:
            continue
        out.append({
            "product_id": p.get("product_id", ""),
            "name": p.get("name", ""),
            "brand": p.get("brand", ""),
            "price": p.get("price", 0),
            "goods_sign": None,
            "evidence_count": 1,
            "matched_constraints": [],
            "violated_constraints": [],
            "risks": [],
            "purchase_url": None,
        })
    return out


def _load_products() -> list[dict]:
    if DATA_PATH.exists():
        with DATA_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    return []


def _extract_constraints(query: str) -> dict:
    constraints = {
        "budget_max": None,
        "usage_scenarios": [],
        "brand_avoid": [],
        "need_portability": "轻" in query or "便携" in query,
        "need_dedicated_gpu": "独显" in query or "游戏" in query,
        "keyword": "笔记本",
    }
    if "编程" in query or "代码" in query:
        constraints["usage_scenarios"].append("编程")
    if "游戏" in query:
        constraints["usage_scenarios"].append("游戏")
    if "洗面奶" in query or "洗面" in query:
        constraints["keyword"] = "洗面奶"
    if "联想" in query and ("不要" in query or "排除" in query):
        constraints["brand_avoid"].append("联想")
    for token in query.replace("元", " ").replace("以内", " ").split():
        if token.isdigit() and 3000 <= int(token) <= 30000:
            constraints["budget_max"] = int(token)
            break
    return constraints


def _get_working_query(state: AgentState) -> str:
    """下游节点统一入口：优先使用总控加工后的需求，否则用原始 query_text。"""
    return (state.get("processed_requirement") or "").strip() or state["query_text"]


def parse_query(state: AgentState) -> AgentState:
    """意图解析：使用 LLM 直接结构化抽取约束。"""
    query = _get_working_query(state)
    history = state.get("history", [])
    recent_history = history[-8:] if isinstance(history, list) else []
    history_text_lines = []
    for m in recent_history:
        role = m.get("role", "user")
        content = m.get("content", "")
        history_text_lines.append(f"{role}: {content}")
    history_text = "\n".join(history_text_lines) or "（无历史对话，当前为第一轮）"

    prompt = CONSTRAINT_PARSER_PROMPT.format(history=history_text, query=query)
    try:
        parsed = constraint_parser_llm.invoke(prompt)
        return {
            "extracted_constraints": {
                "budget_min": parsed.budget_min,
                "budget_max": parsed.budget_max,
                "usage_scenarios": parsed.usage_scenarios,
                "brand_preference": parsed.brand_preference,
                "brand_avoid": parsed.brand_avoid,
                "need_portability": parsed.need_portability,
                "need_dedicated_gpu": parsed.need_dedicated_gpu,
                "keyword": parsed.keyword or "笔记本",
            },
            "missing_info": [],
            "clarifying_questions": [],
            "info_sufficient": True,
        }
    except Exception:
        # 兜底：当结构化抽取失败时退回规则解析，避免流程中断
        return {
            "extracted_constraints": _extract_constraints(query),
            "missing_info": [],
            "clarifying_questions": [],
            "info_sufficient": True,
        }


def check_missing_info(state: AgentState) -> AgentState:
    """判断需求信息是否充分"""
    constraints = state["extracted_constraints"]
    missing = []
    questions = []
    if not constraints.get("budget_max"):
        missing.append("budget_max")
        questions.append("你的预算上限大概是多少？")
    if not constraints.get("usage_scenarios") and "笔记本" in constraints.get("keyword", "笔记本"):
        missing.append("usage_scenarios")
        questions.append("你主要是拿来做编程、办公还是游戏？")
    return {
        "missing_info": missing,
        "clarifying_questions": questions,
        "info_sufficient": len(missing) == 0,
    }


def preference_recall_and_rewrite(state: AgentState) -> AgentState:
    """2. 用户偏好召回与 Query 重写（基于加工后的需求或原始 query）"""
    query = _get_working_query(state)
    user_id = state.get("user_id")
    prefs = recall_preferences(user_id or "default")
    rewritten = rewrite_query(query, prefs)
    return {"user_preferences": prefs, "rewritten_query": rewritten or query}


def dual_path_retrieval(state: AgentState) -> AgentState:
    """3. 双路检索：公开内容 + 公共评价数据"""
    query = state.get("rewritten_query") or state["query_text"]
    constraints = state["extracted_constraints"]
    keyword = constraints.get("keyword", "笔记本")

    # 若上游已执行过平台搜索节点，则复用其结果；否则走默认综合检索。
    content_results = state.get("content_search_results") or search_public_content(query)
    review_results = []
    try:
        pdd_items = search_goods(keyword, page=1, page_size=20)
        for item in pdd_items:
            review_results.append({
                "goods_name": item.get("goods_name"),
                "name": item.get("goods_name"),
                "title": item.get("goods_name"),
                "price": item.get("min_group_price"),
                "goods_id": item.get("goods_id"),
                "goods_sign": item.get("goods_sign"),
            })
    except Exception:
        pass

    return {"content_search_results": content_results, "review_search_results": review_results}


# 向后兼容：workflow 里若仍使用 chatbot 名称，可映射到 controlbot
chatbot = controlbot


def evidence_extraction(state: AgentState) -> AgentState:
    """4. 证据抽取与候选商品生成、标准化、去重"""
    content = state.get("content_search_results", [])
    review = state.get("review_search_results", [])
    evidence = extract_evidence(content, review)
    normalized = deduplicate_candidates([{"product_name": e["product_name"], **e} for e in evidence])
    return {"extracted_evidence": evidence, "candidate_products": normalized, "normalized_candidates": normalized}


def initial_screen_and_consistency(state: AgentState) -> AgentState:
    """5. 质量评估初筛（第1层+第2层）：跨平台一致性 + LLM 多维打分"""
    constraints = state["extracted_constraints"]
    evidence = state.get("extracted_evidence", [])
    candidates = state.get("normalized_candidates", [])
    content_results = state.get("content_search_results", [])
    web_search_summary = state.get("web_search_summary", "")
    query = _get_working_query(state)
    keyword = constraints.get("keyword", "笔记本")

    if not candidates:
        candidates = map_candidates_to_pdd([], keyword)
    if not candidates:
        candidates = _local_products_fallback(constraints)

    # 第1层：跨平台一致性 + 平台权重
    layer1 = compute_cross_platform_scores(candidates, PLATFORM_WEIGHTS)
    # 构建每个商品对应的社交媒体片段
    product_snippets = build_product_snippets(layer1, content_results, evidence)
    # 第2层：LLM 质量打分
    layer2 = llm_quality_score(
        layer1,
        product_snippets,
        query,
        web_search_summary,
        keyword,
        lambda p: model.invoke(p),
    )
    # 综合分 = 跨平台分(40%) + LLM质量分(60%)
    scored = []
    for c in layer2:
        base = compute_combined_score(c)
        matched = list(c.get("matched_constraints", []))
        violated = list(c.get("violated_constraints", []))
        if c.get("platform_count", 0) >= 2:
            matched.append("多平台口碑支持")
        # 预算约束（若有价格）
        price = c.get("price") or c.get("price_yuan", 0)
        if isinstance(price, int) and price > 1000:
            price = price / 100
        if isinstance(price, (int, float)) and constraints.get("budget_max"):
            if price <= constraints["budget_max"]:
                base += 15
                matched.append("预算内")
            else:
                violated.append("超预算")
                base -= 10
        scored.append({
            **c,
            "total_score": round(base, 1),
            "matched_constraints": matched,
            "violated_constraints": violated,
        })
    scored.sort(key=lambda x: x.get("total_score", 0), reverse=True)
    top = scored[:5]
    is_consistent = len(top) >= 2 and (top[0].get("total_score", 0) - (top[1].get("total_score", 0) or 0)) < 25
    need_deep = len(scored) >= 5
    return {
        "scored_candidates": scored[:10],
        "is_consistent": is_consistent,
        "need_deep_search": need_deep,
    }


def deep_search(state: AgentState) -> AgentState:
    """6. 质量评估第3层：Top 8 深度检索 + LLM 终排 Top5"""
    scored = state.get("scored_candidates", [])
    query = _get_working_query(state)
    constraints = state.get("extracted_constraints", {})
    keyword = constraints.get("keyword", "笔记本")

    if len(scored) < 5:
        return {"deep_search_evidence": scored[: min(3, len(scored))]}

    # 对 Top 8 做深度检索
    def _search(q: str, platform: str) -> list[dict]:
        return search_public_content(q, platform=platform)

    deep_results = deep_search_for_candidates(
        scored, query, _search, keyword, top_n=8
    )
    # LLM 终排取 Top5
    reranked = llm_final_rerank(
        scored[:8],
        deep_results,
        query,
        keyword,
        lambda p: model.invoke(p),
        top_k=5,
    )
    # 将终排 Top5 置顶，其余保持原顺序
    rest = [c for c in scored if c not in reranked]
    new_scored = reranked + rest[:5]
    return {
        "deep_search_evidence": reranked[:3],
        "scored_candidates": new_scored,
    }


def pdd_mapping(state: AgentState) -> AgentState:
    """7. 多多进宝映射"""
    scored = state.get("scored_candidates", [])
    constraints = state["extracted_constraints"]
    keyword = constraints.get("keyword", "笔记本")
    pdd_mapped = []
    for c in scored[:10]:
        if c.get("goods_sign"):
            pdd_mapped.append({**c})
        else:
            extra = map_candidates_to_pdd([c.get("product_name", c.get("name", ""))], keyword)
            if extra:
                pdd_mapped.append({
                    **extra[0],
                    "total_score": c.get("total_score", 0),
                    "matched_constraints": c.get("matched_constraints", []),
                    "violated_constraints": c.get("violated_constraints", []),
                    "quality_reason": c.get("quality_reason", ""),
                    "evidence_count": c.get("evidence_count", 0),
                    "platform_count": c.get("platform_count", 0),
                })
    if not pdd_mapped:
        pdd_mapped = map_candidates_to_pdd([], keyword)
    if not pdd_mapped:
        pdd_mapped = _local_products_fallback(constraints)
    goods_signs = [p["goods_sign"] for p in pdd_mapped if p.get("goods_sign")]
    if goods_signs:
        links = get_promotion_links(goods_signs)
        link_map = {l["goods_sign"]: l["url"] for l in links}
        for p in pdd_mapped:
            p["purchase_url"] = link_map.get(p.get("goods_sign", ""))
    pdd_mapped.sort(key=lambda x: x.get("total_score", 0), reverse=True)
    return {"pdd_mapped_products": pdd_mapped[:5], "has_exact_match": len(pdd_mapped) > 0}


def fallback_similar(state: AgentState) -> AgentState:
    """8. 无同款时相似商品兜底"""
    pdd_mapped = state.get("pdd_mapped_products", [])
    if pdd_mapped:
        return {"fallback_products": []}
    constraints = state["extracted_constraints"]
    keyword = constraints.get("keyword", "笔记本")
    fallback = map_candidates_to_pdd([], keyword)
    if not fallback:
        fallback = _local_products_fallback(constraints)
    for f in fallback:
        f["is_fallback"] = True
    return {"fallback_products": fallback}


def generate_final_output(state: AgentState) -> AgentState:
    """9. 最终输出"""
    pdd = state.get("pdd_mapped_products", [])
    fallback = state.get("fallback_products", [])
    scored = state.get("scored_candidates", [])

    candidates = pdd if pdd else fallback
    final_rec = candidates[0] if candidates else None
    reasons = []
    risks = []
    links = []

    if final_rec:
        reasons = list(final_rec.get("matched_constraints", [])[:3])
        qr = final_rec.get("quality_reason", "")
        if qr and qr not in reasons:
            reasons.append(qr)
        if final_rec.get("evidence_count", 0) >= 2 and "多平台口碑支持" not in reasons:
            reasons.append("多平台口碑支持")
        if final_rec.get("is_fallback"):
            reasons.insert(0, "（替代推荐：未找到同款，为您推荐相似商品）")
        risks = final_rec.get("violated_constraints", [])
        if final_rec.get("purchase_url"):
            links.append({"name": final_rec.get("name", ""), "url": final_rec["purchase_url"]})
        for c in candidates[1:4]:
            if c.get("purchase_url"):
                links.append({"name": c.get("name", ""), "url": c["purchase_url"]})

    return {
        "final_recommendation": final_rec,
        "recommendation_reason": reasons,
        "risk_explanations": risks,
        "purchase_links": links,
        "candidates": candidates,
    }

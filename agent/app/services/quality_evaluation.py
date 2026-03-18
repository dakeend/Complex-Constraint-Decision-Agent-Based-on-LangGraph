"""社交媒体商品信息质量评估服务

适配非结构化、无销量/评分的社交媒体内容，采用分层评估：
- 第1层：跨平台一致性 + 平台权重 粗筛
- 第2层：LLM 基于文本质量 + 评价知识打多维分
- 第3层：Top N 深度检索 + LLM 终排 Top5
"""

from typing import Callable

# 平台可信度权重（知乎偏理性长文、小红书偏种草、贴吧偏真实吐槽、微博偏碎片）
PLATFORM_WEIGHTS = {
    "zhihu": 1.2,
    "xiaohongshu": 1.0,
    "tieba": 1.1,
    "weibo": 0.8,
    "general": 1.0,
    "mock": 0.5,
    "pdd_review": 0.9,  # PDD 商品数据作为补充
}


def _normalize_name_for_match(name: str) -> str:
    """用于匹配的简易标准化"""
    if not name:
        return ""
    s = (name or "").strip().lower()
    s = "".join(c for c in s if c.isalnum() or c.isspace() or c in "_-")
    return " ".join(s.split())


def _name_overlap(a: str, b: str) -> bool:
    """判断两个商品名是否可能指向同一商品"""
    na = _normalize_name_for_match(a)
    nb = _normalize_name_for_match(b)
    if not na or not nb:
        return False
    return na in nb or nb in na


def compute_cross_platform_scores(
    candidates: list[dict],
    platform_weights: dict[str, float] | None = None,
) -> list[dict]:
    """
    第1层：基于跨平台提及 + 平台权重计算基础置信度分
    - 同一商品在不同平台被提及次数 × 平台权重
    - 去重：同一平台多次提及只计一次
    """
    weights = platform_weights or PLATFORM_WEIGHTS
    result = []
    for c in candidates:
        sources = c.get("sources", [])
        if not sources:
            result.append({**c, "cross_platform_score": 0.0, "platform_count": 0})
            continue
        # 同一平台只计一次，加权求和
        seen_platforms: set[str] = set()
        weighted_sum = 0.0
        for p in sources:
            if p not in seen_platforms:
                seen_platforms.add(p)
                w = weights.get(p, 1.0)
                weighted_sum += w
        result.append({
            **c,
            "cross_platform_score": round(weighted_sum, 2),
            "platform_count": len(seen_platforms),
        })
    return result


def build_product_snippets(
    candidates: list[dict],
    content_results: list[dict],
    evidence: list[dict],
) -> dict[str, list[dict]]:
    """
    为每个候选商品构建其相关的社交媒体片段（用于 LLM 打分）
    返回: {product_name: [{"platform": str, "title": str, "snippet": str}, ...]}
    """
    product_names: set[str] = set()
    product_aliases: dict[str, set[str]] = {}
    for c in candidates:
        name = c.get("product_name", c.get("name", ""))
        if not name:
            continue
        product_names.add(name)
        aliases = set(c.get("original_names", [name]))
        product_aliases[name] = aliases

    out: dict[str, list[dict]] = {p: [] for p in product_names}

    for r in content_results:
        mentioned = r.get("products_mentioned", [])
        platform = r.get("platform", "unknown")
        title = r.get("title", "")
        snippet = r.get("snippet", "")

        for prod_name in product_names:
            aliases = product_aliases.get(prod_name, {prod_name})
            for alias in aliases:
                for m in mentioned:
                    if _name_overlap(alias, m) or _name_overlap(m, alias):
                        out[prod_name].append({
                            "platform": platform,
                            "title": title,
                            "snippet": snippet[:500],  # 限制长度
                        })
                        break
                else:
                    continue
                break

    return out


def _get_evaluation_criteria(keyword: str) -> str:
    """根据品类返回评价维度的提示（可扩展为外部知识库）"""
    if "洗面奶" in keyword or "洗面" in keyword:
        return """评价维度（参考商品评价知识）：
- 成分安全性（氨基酸表活、无皂基、是否刺激）
- 适用肤质（油皮/干皮/敏感肌）
- 清洁力与温和度平衡
- 性价比
- 用户反馈（好评率、复购意愿）"""
    return """评价维度（参考商品评价知识）：
- 性能/配置是否满足需求
- 续航与散热
- 屏幕素质
- 便携性与做工
- 售后与品牌口碑
- 性价比"""


def llm_quality_score(
    candidates: list[dict],
    product_snippets: dict[str, list[dict]],
    query: str,
    web_search_summary: str,
    keyword: str,
    llm_invoke: Callable,
) -> list[dict]:
    """
    第2层：LLM 基于文本质量 + 评价维度对候选批量打分（0-100）
    输入候选已含 cross_platform_score，输出增加 llm_quality_score、quality_reason
    """
    import json

    criteria = _get_evaluation_criteria(keyword)
    items_for_prompt = []
    for c in candidates:
        name = c.get("product_name", c.get("name", ""))
        snippets = product_snippets.get(name, [])
        snippet_text = "\n".join(
            f"[{s['platform']}] {s['title']}\n{s['snippet']}"
            for s in snippets[:5]
        ) or "（无相关社交内容）"
        items_for_prompt.append({"name": name, "snippet_text": snippet_text})

    block = "\n\n---\n\n".join(
        f"【{i+1}】{x['name']}\n相关社交内容：\n{x['snippet_text']}"
        for i, x in enumerate(items_for_prompt)
    )
    names_list = [x["name"] for x in items_for_prompt]

    prompt = f"""你是一个商品质量评估专家。根据用户需求和社交媒体内容，对以下候选商品逐一打分（0-100）。

用户需求：{query}

平台侧 LLM 整合结论（供参考）：
{web_search_summary or '（无）'}

{criteria}

候选商品及社交内容：
{block}

请输出一个 JSON 数组，每个元素为 {{"name": "商品名", "score": 0-100整数, "reason": "一句话理由"}}，顺序与上述候选一致。
"""
    scored = []
    score_list: list[dict] = [{"score": 50, "reason": "未评分"}] * len(candidates)
    try:
        resp = llm_invoke(prompt)
        text = resp.content if hasattr(resp, "content") else str(resp)
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            arr = json.loads(text[start:end])
            for i, item in enumerate(arr):
                if i >= len(score_list):
                    break
                s = item.get("score", 50)
                r = item.get("reason", "")
                score_list[i] = {"score": min(100, max(0, int(s))), "reason": r}
    except Exception:
        pass

    for i, c in enumerate(candidates):
        ds = score_list[i] if i < len(score_list) else {"score": 50, "reason": "解析失败"}
        scored.append({
            **c,
            "llm_quality_score": ds["score"],
            "quality_reason": ds.get("reason", ""),
        })
    return scored


def compute_combined_score(c: dict) -> float:
    """综合跨平台分 + LLM 质量分（各占 50% 归一化后）"""
    cross = c.get("cross_platform_score", 0)
    llm_s = c.get("llm_quality_score", 50)
    # 跨平台分通常 0~4，归一化到 0-100 尺度（假设最多 4 平台 × 1.2）
    cross_norm = min(100, cross * 25)
    return (cross_norm * 0.4 + llm_s * 0.6)


def deep_search_for_candidates(
    candidates: list[dict],
    query: str,
    search_fn: Callable[[str, str], list[dict]],
    keyword: str,
    top_n: int = 8,
) -> dict[str, list[dict]]:
    """
    第3层：对 Top N 候选做深度检索，获取更多测评/讨论内容
    返回: {product_name: [{"title", "snippet", "platform"}, ...]}
    """
    result: dict[str, list[dict]] = {}
    for c in candidates[:top_n]:
        name = c.get("product_name", c.get("name", ""))
        if not name:
            continue
        search_query = f"{keyword} {name} 评测 使用体验 推荐"
        try:
            hits = search_fn(search_query, "general")
            items = []
            for h in hits[:5]:
                items.append({
                    "title": h.get("title", ""),
                    "snippet": (h.get("snippet", "") or h.get("content", ""))[:600],
                    "platform": h.get("platform", "general"),
                })
            result[name] = items
        except Exception:
            result[name] = []
    return result


def llm_final_rerank(
    top_candidates: list[dict],
    deep_results: dict[str, list[dict]],
    query: str,
    keyword: str,
    llm_invoke: Callable,
    top_k: int = 5,
) -> list[dict]:
    """
    第3层：基于深度检索结果，LLM 做最终排序，输出 Top K
    """
    if len(top_candidates) <= top_k:
        return top_candidates

    lines = []
    for i, c in enumerate(top_candidates, 1):
        name = c.get("product_name", c.get("name", ""))
        deep = deep_results.get(name, [])
        deep_text = "\n".join(
            f"- {d['title']}: {d['snippet'][:200]}..."
            for d in deep[:3]
        ) or "（无额外深度内容）"
        lines.append(f"""
[候选{i}] {name}
- 综合分: {c.get('total_score', 0):.1f}
- 跨平台支持: {c.get('platform_count', 0)} 个平台
- 深度检索内容:
{deep_text}
""")

    prompt = f"""你是一个商品推荐专家。根据用户需求和各候选的深度检索内容，选出最终 Top {top_k} 推荐。

用户需求：{query}

候选列表（已按初评排序）：
{"".join(lines)}

请输出一个 JSON 数组，仅包含你选中的商品名称（按推荐顺序）：
["商品名1", "商品名2", ...]
"""
    try:
        resp = llm_invoke(prompt)
        text = resp.content if hasattr(resp, "content") else str(resp)
        import json
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            names = json.loads(text[start:end])
        else:
            names = [c.get("product_name", c.get("name", "")) for c in top_candidates[:top_k]]
    except Exception:
        names = [c.get("product_name", c.get("name", "")) for c in top_candidates[:top_k]]

    # 按 names 顺序重排，未在 names 中的按原顺序补足
    order_map = {n: i for i, n in enumerate(names)}
    def rank_key(c):
        name = c.get("product_name", c.get("name", ""))
        return (order_map.get(name, 999), -c.get("total_score", 0))
    sorted_candidates = sorted(top_candidates, key=rank_key)
    return sorted_candidates[:top_k]

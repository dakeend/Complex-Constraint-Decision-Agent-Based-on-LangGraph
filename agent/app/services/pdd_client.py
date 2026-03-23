"""多多进宝 API 封装，供 workflow 调用"""
try:
    # 正常仓库结构：agent/data/dataupdate.py
    from ...data.dataupdate import (
        _extract_pdd_items,
        generate_pdd_promotion_url,
        get_pdd_goods_detail,
        search_pdd_goods,
    )
except Exception:  # pragma: no cover
    # 兼容历史路径（例如被复制到不同包结构时）
    from agent.data.dataupdate import (  # type: ignore
        _extract_pdd_items,
        generate_pdd_promotion_url,
        get_pdd_goods_detail,
        search_pdd_goods,
    )


def search_goods(keyword: str, page: int = 1, page_size: int = 20) -> list[dict]:
    """搜索多多进宝商品"""
    try:
        resp = search_pdd_goods(keyword, page, page_size)
        if "error_response" in resp:
            return []
        return _extract_pdd_items(resp)
    except Exception:
        return []


def find_pdd_match_for_candidate(candidate_name: str, keyword: str) -> dict | None:
    """
    按商品名在多多进宝中查找匹配商品。
    用「商品名 + 品类关键词」搜索，取首个结果作为匹配。
    返回带 goods_sign 的 candidate 格式，未找到则返回 None。
    """
    if not candidate_name or not candidate_name.strip():
        return None
    query = f"{candidate_name.strip()} {keyword}".strip()
    pdd_items = search_goods(query, page=1, page_size=10)
    if not pdd_items:
        return None
    return _pdd_item_to_candidate(pdd_items[0], False)


def search_approximate_goods(
    keyword: str,
    exclude_goods_signs: set[str],
    budget_max: int | None = None,
    max_results: int = 5,
) -> list[dict]:
    """
    在多多进宝中搜索品类关键词，返回近似推荐商品。
    排除已匹配的 goods_sign，可选按预算过滤。
    """
    pdd_items = search_goods(keyword, page=1, page_size=30)
    if not pdd_items:
        return []
    result = []
    for item in pdd_items:
        gs = item.get("goods_sign")
        if gs and gs in exclude_goods_signs:
            continue
        c = _pdd_item_to_candidate(item, False)
        c["is_approximate"] = True
        if budget_max is not None:
            price = c.get("price", 0)
            if price > budget_max:
                continue
        result.append(c)
        if len(result) >= max_results:
            break
    return result


def map_candidates_to_pdd(candidate_names: list[str], keyword: str) -> list[dict]:
    """
    将候选商品名映射到多多进宝商品。
    用 keyword 搜索 PDD 返回通用结果（供兜底场景使用）。
    """
    pdd_items = search_goods(keyword, page=1, page_size=50)
    if not pdd_items:
        return []

    mapped = []
    for item in pdd_items:
        name = item.get("goods_name") or item.get("name") or item.get("title", "")
        if not name:
            continue
        mapped.append(_pdd_item_to_candidate(item, False))
    return mapped[:10]


def get_promotion_links(goods_sign_list: list[str]) -> list[dict[str, str]]:
    """批量获取推广链接"""
    if not goods_sign_list:
        return []
    try:
        resp = generate_pdd_promotion_url(goods_sign_list[:20])
        if "error_response" in resp:
            return []
        r = resp.get("goods_promotion_url_generate_response", {})
        url_list = r.get("goods_promotion_url_list", r.get("url_list", []))
        links = []
        for u in url_list:
            gs = u.get("goods_sign", "")
            url = u.get("url") or u.get("mobile_url") or u.get("short_url", "")
            if gs and url:
                links.append({"goods_sign": gs, "url": url})
        return links
    except Exception:
        return []


def enrich_with_details(goods_sign_list: list[str]) -> dict[str, dict]:
    """获取商品详情（销量等）"""
    if not goods_sign_list:
        return {}
    try:
        resp = get_pdd_goods_detail(goods_sign_list[:20])
        if "error_response" in resp:
            return {}
        r = resp.get("goods_detail_response", resp.get("goods_detail_list", {}))
        details_list = r.get("goods_details", r.get("goods_detail_list", []))
        out = {}
        for d in details_list:
            gs = d.get("goods_sign", "")
            if gs:
                out[gs] = d
        return out
    except Exception:
        return {}


def _pdd_item_to_candidate(item: dict, is_fallback: bool = False) -> dict:
    price = item.get("min_group_price") or item.get("min_normal_price") or 0
    price_yuan = (price / 100) if isinstance(price, int) else price
    return {
        "product_id": str(item.get("goods_id", item.get("goods_sign", ""))),
        "name": item.get("goods_name") or item.get("name") or item.get("title", "未知"),
        "brand": item.get("mall_name", ""),
        "price": int(price_yuan) if isinstance(price_yuan, float) else price_yuan,
        "goods_sign": item.get("goods_sign"),
        "evidence_count": 0,
        "total_score": 0,
        "matched_constraints": [],
        "violated_constraints": [],
        "risks": [],
        "purchase_url": None,
        "is_fallback": is_fallback,
    }

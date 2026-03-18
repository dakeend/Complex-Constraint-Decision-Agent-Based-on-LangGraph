"""公开内容搜索（Tavily：限定知乎、小红书）
需配置环境变量 TAVILY_API_KEY"""
import os
import re

# 常见品牌/型号，用于从内容中抽取商品提及
KNOWN_BRANDS = [
    "华硕", "惠普", "联想", "机械革命", "戴尔", "小米", "华为", "苹果",
    "溪木源", "多芬", "大宝", "珂岸", "百雀羚", "RNW", "理然", "妮维雅",
    "华硕无畏", "惠普战66", "联想小新", "机械革命极光",
]


def _extract_products_from_text(text: str) -> list[str]:
    """从文本中抽取可能提及的商品/品牌"""
    if not text:
        return []
    found = []
    for brand in KNOWN_BRANDS:
        if brand in text:
            found.append(brand)
    return list(dict.fromkeys(found))


# 平台与 Tavily 域名对应（用于限定条件搜索）
PLATFORM_DOMAINS = {
    "zhihu": ["zhihu.com"],
    "xiaohongshu": ["xiaohongshu.com", "xiaohongshu.cn"],
    "tieba": ["tieba.baidu.com"],
    "weibo": ["weibo.com", "weibo.cn"],
    "general": [
        "zhihu.com",
        "xiaohongshu.com",
        "xiaohongshu.cn",
        "tieba.baidu.com",
        "weibo.com",
        "weibo.cn",
    ],
}


def _url_to_platform(url: str) -> str:
    """根据 URL 推断平台标识"""
    url_lower = (url or "").lower()
    if "zhihu" in url_lower:
        return "zhihu"
    if "xiaohongshu" in url_lower:
        return "xiaohongshu"
    if "tieba.baidu" in url_lower:
        return "tieba"
    if "weibo" in url_lower:
        return "weibo"
    return "general"


def _tavily_search(query: str, include_domains: list[str], max_results: int = 10) -> list[dict]:
    """调用 Tavily 搜索，限定域名"""
    try:
        from tavily import TavilyClient
    except ImportError:
        return []

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return []

    client = TavilyClient(api_key=api_key)
    resp = client.search(
        query=query,
        include_domains=include_domains,
        max_results=max_results,
        search_depth="basic",
    )
    if isinstance(resp, dict):
        results = resp.get("results", [])
    else:
        results = getattr(resp, "results", []) or []
    out = []
    for r in results:
        title = r.get("title", "")
        content = r.get("content", "")
        url = r.get("url", "")
        domain = _url_to_platform(url)
        products = _extract_products_from_text(content or title)
        out.append({
            "platform": domain,
            "title": title,
            "snippet": content or title,
            "products_mentioned": products,
            "source": url,
        })
    return out


def _mock_fallback(query: str, keyword: str) -> list[dict]:
    """无 API Key 时的模拟兜底"""
    mock_data = [
        ("2024轻薄本推荐", "华硕无畏Pro15性价比高，RTX4050适合编程和轻度游戏。惠普战66六代更便携。", ["华硕无畏Pro15", "惠普战66"]),
        ("程序员笔记本", "机械革命极光X性能强但偏重，联想小新Pro14 32G内存适合多开IDE。", ["机械革命极光X", "联想小新Pro14"]),
        ("洗面奶推荐", "溪木源、多芬氨基酸洁面口碑好，敏感肌可用。大宝B5性价比高。", ["溪木源", "多芬", "大宝"]),
    ]
    results = []
    for i, (title, snippet, products) in enumerate(mock_data[:5]):
        results.append({
            "platform": "mock",
            "title": title,
            "snippet": snippet,
            "products_mentioned": products,
            "source": f"mock_{i}",
        })
    return results


def search_public_content(query: str, platform: str = "general") -> list[dict]:
    """
    使用 Tavily 搜索公开内容，可按平台限定。
    platform: "zhihu" | "xiaohongshu" | "tieba" | "weibo" | "general"（不限定，多平台）
    需配置 TAVILY_API_KEY，否则退回模拟数据。
    返回格式: [{"platform": str, "title": str, "snippet": str, "products_mentioned": list[str]}]
    """
    domains = PLATFORM_DOMAINS.get(platform, PLATFORM_DOMAINS["general"])
    if not query.strip():
        query = "笔记本电脑 推荐"

    results = _tavily_search(query, include_domains=domains, max_results=10)
    if results:
        # 限定平台时只保留该平台结果；general 保留全部
        if platform != "general":
            results = [r for r in results if (r.get("platform") or "").lower() == platform]
        return results

    keyword = "笔记本" if "洗" not in query and "面奶" not in query else "洗面奶"
    return _mock_fallback(query, keyword)

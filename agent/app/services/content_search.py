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
# 小红书：排除 pgy/e/ci 等商业化/帮助子域，优先主站笔记；贴吧用 tieba.baidu.com
PLATFORM_DOMAINS = {
    "zhihu": ["zhihu.com"],
    "xiaohongshu": ["www.xiaohongshu.com", "xhslink.com", "xiaohongshu.com"],
    "tieba": ["tieba.baidu.com"],
    "weibo": ["weibo.com", "weibo.cn"],
    "general": [
        "zhihu.com",
        "www.xiaohongshu.com",
        "xhslink.com",
        "xiaohongshu.com",
        "tieba.baidu.com",
        "weibo.com",
        "weibo.cn",
    ],
}

# 小红书需排除的子域（商业化/帮助中心，非用户种草笔记）
XIAOHONGSHU_EXCLUDE_PREFIXES = ("pgy.xiaohongshu", "e.xiaohongshu", "ci.xiaohongshu", "ad.xiaohongshu", "partner.xiaohongshu")


def _get_int_env(name: str, default: int) -> int:
    v = (os.getenv(name) or "").strip()
    if not v:
        return default
    try:
        n = int(v)
        return n if n > 0 else default
    except Exception:
        return default


def _compact_snippet(text: str, max_chars: int) -> str:
    """将原文压成适合 prompt 的片段（尽量保留更多原文，不是摘要）。"""
    if not text:
        return ""
    s = re.sub(r"\s+", " ", text).strip()
    if len(s) <= max_chars:
        return s
    # 尝试在句号/分号附近截断，避免硬切
    cut = max(
        s.rfind("。", 0, max_chars),
        s.rfind("；", 0, max_chars),
        s.rfind(";", 0, max_chars),
    )
    if cut >= max_chars * 0.6:
        return s[: cut + 1]
    return s[:max_chars]


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
    # Tavily 的 content 往往是摘要级；尽量请求 raw_content 作为更长的 snippet 来源。
    # 不同版本 SDK 参数可能不同，这里做兼容降级。
    search_depth = (os.getenv("TAVILY_SEARCH_DEPTH") or "advanced").strip() or "advanced"
    try:
        resp = client.search(
            query=query,
            include_domains=include_domains,
            max_results=max_results,
            search_depth=search_depth,
            include_raw_content=True,
        )
    except TypeError:
        resp = client.search(
            query=query,
            include_domains=include_domains,
            max_results=max_results,
            search_depth=search_depth,
        )
    if isinstance(resp, dict):
        results = resp.get("results", [])
    else:
        results = getattr(resp, "results", []) or []
    snippet_max_chars = _get_int_env("SEARCH_SNIPPET_MAX_CHARS", 800)
    out = []
    for r in results:
        title = r.get("title", "")
        content = r.get("content", "") or ""
        raw_content = r.get("raw_content", "") or ""
        url = (r.get("url", "") or "").lower()
        domain = _url_to_platform(url)
        # 小红书：排除商业化/帮助子域，这些不是用户种草笔记
        if domain == "xiaohongshu" and any(p in url for p in XIAOHONGSHU_EXCLUDE_PREFIXES):
            continue
        # 优先 raw_content，为空时用 content（贴吧等平台常无 raw_content）
        snippet_source = raw_content or content or title
        snippet = _compact_snippet(snippet_source, snippet_max_chars)
        raw_stored = raw_content or content  # 无 raw 时存 content，供下游用
        products = _extract_products_from_text((content or raw_content) or title)
        out.append({
            "platform": domain,
            "title": title,
            "snippet": snippet,
            "raw_content": _compact_snippet(raw_stored, max(snippet_max_chars * 2, 1200)) if raw_stored else "",
            "products_mentioned": products,
            "source": r.get("url", ""),
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
        raw = snippet * 3 if len(snippet) < 150 else snippet  # 模拟有正文，满足 min_content_len
        results.append({
            "platform": "mock",
            "title": title,
            "snippet": snippet,
            "raw_content": raw,
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
    # 小红书：加「种草 推荐」提高命中真实笔记的概率
    if platform == "xiaohongshu" and "种草" not in query and "推荐" not in query:
        query = f"{query} 种草 推荐"

    results = _tavily_search(query, include_domains=domains, max_results=10)
    if results:
        # 限定平台时只保留该平台结果；general 保留全部
        if platform != "general":
            results = [r for r in results if (r.get("platform") or "").lower() == platform]
        return results

    keyword = "笔记本" if "洗" not in query and "面奶" not in query else "洗面奶"
    return _mock_fallback(query, keyword)

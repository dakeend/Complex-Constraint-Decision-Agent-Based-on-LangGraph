"""笔记本评测知识库检索服务

使用 FAISS 构建向量索引，支持：
- 按商品名/系列名检索
- 按场景检索
- 按优缺点检索

核心接口：
- retrieve_product_review(product_name) -> dict
- retrieve_by_scenario(scenario) -> list[dict]
- retrieve_similar_reviews(query) -> list[dict]
"""

import json
import re
from pathlib import Path
from typing import Optional

# 数据文件路径（位于 agent/data 目录下）
DATA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "laptop_reviews.json"

# 全局缓存
_reviews: list[dict] = []
_index: Optional["faiss.IndexFlatIP"] = None
_embeddings: Optional[list] = None
_texts: list[str] = []


def _load_reviews() -> list[dict]:
    """加载评测数据"""
    global _reviews
    if _reviews:
        return _reviews

    if not DATA_PATH.exists():
        print(f"警告: 评测数据文件不存在: {DATA_PATH}")
        _reviews = []
        return _reviews

    with DATA_PATH.open("r", encoding="utf-8") as f:
        _reviews = json.load(f)

    return _reviews


def _get_embedding_model():
    """获取嵌入模型（使用简单的文本匹配或真正的嵌入模型）"""
    # 方案1：使用 sentence-transformers（需要安装）
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    except ImportError:
        print("提示: 安装 sentence-transformers 可获得更好的检索效果: pip install sentence-transformers")
        return None


def _build_faiss_index():
    """构建 FAISS 向量索引"""
    global _index, _embeddings, _texts

    reviews = _load_reviews()
    if not reviews:
        return None

    model = _get_embedding_model()
    if model is None:
        # 退回简单文本匹配
        return None

    # 为每个评测生成嵌入文本
    _texts = []
    for r in reviews:
        # 合并所有字段为一个文本
        text_parts = [
            r.get("series_name", ""),
            r.get("brand", ""),
            "优点: " + ", ".join(r.get("pros", [])),
            "缺点: " + ", ".join(r.get("cons", [])),
            "适用场景: " + ", ".join(r.get("suitable_scenarios", [])),
            "不适用场景: " + ", ".join(r.get("unsuitable_scenarios", [])),
            "目标用户: " + ", ".join(r.get("target_users", [])),
            r.get("summary", ""),
        ]
        _texts.append(" ".join(text_parts))

    # 生成嵌入向量
    _embeddings = model.encode(_texts, normalize_embeddings=True)

    # 构建 FAISS 索引
    try:
        import faiss
        dimension = _embeddings.shape[1]
        _index = faiss.IndexFlatIP(dimension)
        _index.add(_embeddings)
    except ImportError:
        print("提示: 安装 faiss-cpu: pip install faiss-cpu")
        _index = None

    return _index


def _normalize_name(name: str) -> str:
    """标准化名称用于匹配"""
    if not name:
        return ""
    s = name.strip().lower()
    s = re.sub(r"\[[^\]]*\]", "", s)
    s = re.sub(r"\s*20\d{2}\s*", "", s)
    s = re.sub(r"\s+", "", s)
    return s


def _name_matches(query: str, series: str) -> bool:
    """判断查询名是否匹配系列名"""
    q = _normalize_name(query)
    s = _normalize_name(series)
    if not q or not s:
        return False
    return q in s or s in q


def retrieve_product_review(product_name: str) -> Optional[dict]:
    """
    按商品名/系列名检索评测信息

    Args:
        product_name: 商品名或系列名（如 "华硕无畏Pro15"、"联想小新Pro14"）

    Returns:
        评测信息字典，若未找到返回 None
    """
    reviews = _load_reviews()
    if not reviews:
        return None

    # 直接匹配
    for r in reviews:
        series = r.get("series_name", "")
        if _name_matches(product_name, series):
            return r

    # 尝试从商品名提取系列名
    # 例如 "华硕无畏Pro15 2024" -> "华硕无畏Pro"
    name_norm = product_name.strip()
    for r in reviews:
        series = r.get("series_name", "")
        brand = r.get("brand", "")
        # 品牌匹配 + 系列关键词匹配
        if brand in name_norm and any(kw in name_norm for kw in series.split()[:2]):
            return r

    return None


def retrieve_by_scenario(scenario: str, top_k: int = 5) -> list[dict]:
    """
    按场景检索适合的笔记本系列

    Args:
        scenario: 使用场景（如 "编程"、"游戏"、"办公"）
        top_k: 返回数量

    Returns:
        适合该场景的评测列表
    """
    reviews = _load_reviews()
    if not reviews:
        return []

    results = []
    scenario_lower = scenario.lower()

    for r in reviews:
        suitable = r.get("suitable_scenarios", [])
        # 检查是否包含该场景关键词
        if any(scenario_lower in s.lower() or s.lower() in scenario_lower for s in suitable):
            results.append(r)

    return results[:top_k]


def retrieve_similar_reviews(query: str, top_k: int = 5) -> list[dict]:
    """
    相似检索：根据查询文本检索最相关的评测

    Args:
        query: 查询文本（如 "5000元 编程 轻薄"）
        top_k: 返回数量

    Returns:
        最相关的评测列表
    """
    reviews = _load_reviews()
    if not reviews:
        return []

    # 优先使用 FAISS 向量检索
    index = _build_faiss_index()
    if index is not None:
        model = _get_embedding_model()
        if model:
            query_embedding = model.encode([query], normalize_embeddings=True)
            distances, indices = index.search(query_embedding, top_k)
            results = [reviews[i] for i in indices[0] if i < len(reviews)]
            return results

    # 退回关键词匹配
    results = []
    query_lower = query.lower()
    query_keywords = set(re.findall(r"\w+", query_lower))

    for r in reviews:
        # 计算关键词匹配分数
        text = " ".join([
            r.get("series_name", ""),
            r.get("brand", ""),
            " ".join(r.get("pros", [])),
            " ".join(r.get("suitable_scenarios", [])),
            " ".join(r.get("target_users", [])),
            r.get("summary", ""),
        ]).lower()

        text_keywords = set(re.findall(r"\w+", text))
        overlap = len(query_keywords & text_keywords)

        if overlap > 0:
            results.append((r, overlap))

    # 按匹配分数排序
    results.sort(key=lambda x: x[1], reverse=True)
    return [r for r, _ in results[:top_k]]


def retrieve_for_candidates(candidates: list[dict]) -> dict[str, dict]:
    """
    为候选商品批量检索评测信息

    Args:
        candidates: 候选商品列表，每项需有 product_name 或 name 字段

    Returns:
        {商品名: 评测信息} 的字典
    """
    result: dict[str, dict] = {}

    for c in candidates:
        name = c.get("product_name") or c.get("name") or c.get("normalized_name", "")
        if not name:
            continue

        review = retrieve_product_review(name)
        if review:
            result[name] = review

    return result


def get_all_reviews() -> list[dict]:
    """获取所有评测数据"""
    return _load_reviews()


# 初始化时预加载
_load_reviews()
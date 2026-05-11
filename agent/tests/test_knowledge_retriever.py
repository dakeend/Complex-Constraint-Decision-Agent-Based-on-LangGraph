"""知识库检索服务测试"""
import pytest
from pathlib import Path

# 确保数据文件存在
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "laptop_reviews.json"


def test_data_file_exists():
    """测试数据文件存在"""
    assert DATA_PATH.exists(), f"评测数据文件不存在: {DATA_PATH}"


def test_retrieve_product_review():
    """测试按商品名检索"""
    from agent.app.services.knowledge_retriever import retrieve_product_review

    # 测试精确匹配
    result = retrieve_product_review("华硕无畏Pro")
    assert result is not None
    assert "华硕" in result.get("brand", "")
    assert len(result.get("pros", [])) > 0

    # 测试模糊匹配（带型号后缀）
    result = retrieve_product_review("华硕无畏Pro15 2024")
    assert result is not None

    # 测试不存在的商品
    result = retrieve_product_review("不存在的笔记本")
    assert result is None


def test_retrieve_by_scenario():
    """测试按场景检索"""
    from agent.app.services.knowledge_retriever import retrieve_by_scenario

    # 测试编程场景
    results = retrieve_by_scenario("编程", top_k=5)
    assert len(results) > 0
    for r in results:
        assert "编程" in " ".join(r.get("suitable_scenarios", []))

    # 测试游戏场景
    results = retrieve_by_scenario("游戏", top_k=5)
    assert len(results) > 0


def test_retrieve_similar_reviews():
    """测试相似检索"""
    from agent.app.services.knowledge_retriever import retrieve_similar_reviews

    results = retrieve_similar_reviews("5000元 轻薄 编程", top_k=5)
    assert len(results) > 0


def test_retrieve_for_candidates():
    """测试批量检索"""
    from agent.app.services.knowledge_retriever import retrieve_for_candidates

    candidates = [
        {"product_name": "华硕无畏Pro15"},
        {"name": "联想小新Pro14"},
        {"product_name": "惠普战66六代"},
    ]

    result = retrieve_for_candidates(candidates)
    assert len(result) >= 2
    assert "华硕无畏Pro15" in result or any("华硕" in k for k in result)


def test_get_all_reviews():
    """测试获取全部评测"""
    from agent.app.services.knowledge_retriever import get_all_reviews

    reviews = get_all_reviews()
    assert len(reviews) > 0
    assert all("series_name" in r for r in reviews)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
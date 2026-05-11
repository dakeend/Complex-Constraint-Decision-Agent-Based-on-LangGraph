"""笔记本评测知识库采集脚本

使用 Tavily API 批量搜索笔记本系列评测内容，
结合 LLM 自动提取结构化信息（优缺点、场景、风险）。

运行方式：
    python agent/data/build_knowledge_base.py

输出：
    agent/data/laptop_reviews.json
"""

import json
import os
import re
from pathlib import Path
from typing import Optional

# 笔记本系列列表（主流品牌的主流系列）
LAPTOP_SERIES = [
    # 华硕
    "华硕无畏Pro", "华硕无畏", "华硕灵耀", "华硕天选", "华硕ROG幻系列",
    # 联想
    "联想小新Pro", "联想小新", "联想拯救者", "联想ThinkBook", "联想YOGA",
    # 惠普
    "惠普战66", "惠普战X", "惠普星系列", "惠普暗影精灵", "惠普OMEN",
    # 戴尔
    "戴尔灵越", "戴尔XPS", "戴尔游匣", "戴尔G系列",
    # 华为
    "华为MateBook", "华为MateBook D", "华为MateBook X",
    # 小米/荣耀
    "小米RedmiBook", "小米笔记本Pro", "荣耀MagicBook",
    # 机械革命
    "机械革命极光", "机械革命蛟龙", "机械革命无界",
    # 宏碁
    "宏碁非凡", "宏碁暗影骑士", "宏碁掠夺者",
    # 神舟
    "神舟战神", "神舟优雅",
    # 雷神
    "雷神911", "雷神ZERO",
    # 其他
    "苹果MacBook Air", "苹果MacBook Pro",
]

OUTPUT_PATH = Path(__file__).resolve().parent / "laptop_reviews.json"


def _get_tavily_client():
    """获取 Tavily 客户端"""
    try:
        from tavily import TavilyClient
    except ImportError:
        raise ImportError("请安装 tavily-python: pip install tavily-python")

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("请设置环境变量 TAVILY_API_KEY")

    return TavilyClient(api_key=api_key)


def _get_llm():
    """获取 LLM 客户端（使用 Groq）"""
    try:
        from langchain_groq import ChatGroq
    except ImportError:
        raise ImportError("请安装 langchain-groq: pip install langchain-groq")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("请设置环境变量 GROQ_API_KEY")

    return ChatGroq(model="llama-3.3-70b-versatile", api_key=api_key)


def search_reviews(series: str, max_results: int = 10) -> list[dict]:
    """使用 Tavily 搜索笔记本系列的评测内容"""
    client = _get_tavily_client()
    query = f"{series} 笔记本 评测 优缺点 使用体验 推荐"

    try:
        resp = client.search(
            query=query,
            include_domains=["zhihu.com", "www.xiaohongshu.com", "tieba.baidu.com"],
            max_results=max_results,
            search_depth="advanced",
            include_raw_content=True,
        )
    except TypeError:
        # 兼容不同版本 SDK
        resp = client.search(
            query=query,
            include_domains=["zhihu.com", "www.xiaohongshu.com", "tieba.baidu.com"],
            max_results=max_results,
        )

    if isinstance(resp, dict):
        results = resp.get("results", [])
    else:
        results = getattr(resp, "results", []) or []

    return results


EXTRACT_PROMPT = """你是笔记本评测信息提取专家。请从以下搜索结果中提取结构化评测信息。

笔记本系列：{series}

搜索结果：
{search_content}

请提取以下信息并以 JSON 格式输出，包含以下字段：
- series_name: 系列名称
- brand: 品牌
- pros: 优点列表（3-5条）
- cons: 缺点列表（3-5条）
- suitable_scenarios: 适用场景列表
- unsuitable_scenarios: 不适用场景列表
- target_users: 目标用户列表
- risk_points: 风险点列表
- summary: 一句话总结

要求：
1. pros 和 cons 各提取 3-5 条，要有实质性内容（如散热、屏幕、性能等）
2. scenarios 要具体（如"编程开发"、"轻度游戏"、"学生日常"）
3. risk_points 是该系列常见问题或投诉点
4. 如果搜索结果信息不足，可基于品牌定位和系列特点合理推断
5. 只输出 JSON，不要其他文字"""


def extract_review_with_llm(series: str, search_results: list[dict]) -> Optional[dict]:
    """使用 LLM 从搜索结果提取结构化评测信息"""
    llm = _get_llm()

    # 合并搜索结果内容
    content_parts = []
    for i, r in enumerate(search_results[:5], 1):
        title = r.get("title", "")
        content = r.get("content", "") or r.get("raw_content", "") or ""
        if content:
            content_parts.append(f"[{i}] {title}\n{content[:800]}")

    search_content = "\n\n".join(content_parts) or f"（无搜索结果，请根据 {series} 系列的一般定位推断）"

    prompt = EXTRACT_PROMPT.format(series=series, search_content=search_content)

    try:
        resp = llm.invoke(prompt)
        text = resp.content if hasattr(resp, "content") else str(resp)

        # 提取 JSON
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(text[start:end])
            # 确保必要字段存在
            data.setdefault("series_name", series)
            data.setdefault("brand", _extract_brand(series))
            data.setdefault("pros", [])
            data.setdefault("cons", [])
            data.setdefault("suitable_scenarios", [])
            data.setdefault("unsuitable_scenarios", [])
            data.setdefault("target_users", [])
            data.setdefault("risk_points", [])
            data.setdefault("summary", "")
            return data
    except Exception as e:
        print(f"  提取失败: {e}")
        return None

    return None


def _extract_brand(series: str) -> str:
    """从系列名提取品牌"""
    brands = ["华硕", "联想", "惠普", "戴尔", "华为", "小米", "荣耀", "机械革命", "宏碁", "神舟", "雷神", "苹果"]
    for b in brands:
        if b in series:
            return b
    return "其他"


def collect_all_reviews(series_list: list[str] = None, start_index: int = 0) -> list[dict]:
    """批量采集所有系列的评测信息

    Args:
        series_list: 要采集的系列列表，默认使用 LAPTOP_SERIES
        start_index: 开始采集的索引（从0开始），用于断点续采
    """
    if series_list is None:
        series_list = LAPTOP_SERIES

    # 加载已有数据（用于断点续采）
    existing_reviews: dict[str, dict] = {}
    if OUTPUT_PATH.exists():
        with OUTPUT_PATH.open("r", encoding="utf-8") as f:
            existing_data = json.load(f)
            for r in existing_data:
                name = r.get("series_name", "")
                if name:
                    existing_reviews[name] = r
        print(f"已加载 {len(existing_reviews)} 条已有数据")

    reviews = []
    total = len(series_list)

    print(f"开始采集 {total - start_index} 个笔记本系列的评测信息（从第 {start_index + 1} 个开始）...")

    for i, series in enumerate(series_list, 1):
        if i - 1 < start_index:
            # 跳过已采集的系列，使用已有数据
            if series in existing_reviews:
                reviews.append(existing_reviews[series])
            continue

        print(f"\n[{i}/{total}] 采集: {series}")

        # 检查是否已有完整数据（有优缺点信息）
        if series in existing_reviews:
            existing = existing_reviews[series]
            if existing.get("pros") and existing.get("cons"):
                print(f"  已有完整数据，跳过")
                reviews.append(existing)
                continue

        # 搜索
        print("  搜索评测内容...")
        search_results = search_reviews(series, max_results=8)

        if search_results:
            print(f"  找到 {len(search_results)} 条结果")
        else:
            print("  未找到搜索结果")

        # LLM 提取
        print("  LLM 提取结构化信息...")
        review = extract_review_with_llm(series, search_results)

        if review:
            reviews.append(review)
            print(f"  成功提取: {len(review.get('pros', []))} 个优点, {len(review.get('cons', []))} 个缺点")
        else:
            # 失败时创建基础记录
            reviews.append({
                "series_name": series,
                "brand": _extract_brand(series),
                "pros": [],
                "cons": [],
                "suitable_scenarios": [],
                "unsuitable_scenarios": [],
                "target_users": [],
                "risk_points": [],
                "summary": f"{series}系列（待补充详细信息）",
            })
            print("  提取失败，创建基础记录")

        # 每采集5个保存一次，避免丢失数据
        if i % 5 == 0:
            save_reviews(reviews + [existing_reviews[s] for s in existing_reviews if s not in {r.get("series_name") for r in reviews}], OUTPUT_PATH)
            print(f"  [自动保存] 已保存 {len(reviews)} 条记录")

    return reviews


def save_reviews(reviews: list[dict], output_path: Path = None):
    """保存评测信息到 JSON 文件"""
    if output_path is None:
        output_path = OUTPUT_PATH

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(reviews, f, ensure_ascii=False, indent=2)

    print(f"\n已保存 {len(reviews)} 条评测信息到: {output_path}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="笔记本评测知识库采集脚本")
    parser.add_argument("--start", type=int, default=0, help="开始采集的索引（从0开始），默认0")
    parser.add_argument("--retry-empty", action="store_true", help="仅重新采集数据为空的系列")
    args = parser.parse_args()

    # 加载环境变量
    from dotenv import load_dotenv
    load_dotenv()

    # 检查环境变量
    if not os.getenv("TAVILY_API_KEY"):
        print("错误: 请设置 TAVILY_API_KEY 环境变量")
        return
    if not os.getenv("GROQ_API_KEY"):
        print("错误: 请设置 GROQ_API_KEY 环境变量")
        return

    # 采集
    reviews = collect_all_reviews(start_index=args.start)

    # 保存
    save_reviews(reviews)

    print("\n采集完成！")


if __name__ == "__main__":
    main()
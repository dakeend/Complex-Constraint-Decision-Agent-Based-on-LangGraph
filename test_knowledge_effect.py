"""知识库效果测试脚本

测试场景：5000元以内编程笔记本推荐
预期结果：
1. 推荐列表包含知识库评测的系列
2. 推荐理由包含知识库的优缺点信息
"""

import requests
import json
import time

BASE_URL = "http://127.0.0.1:8001"

def test_recommendation():
    """测试推荐接口"""

    # 测试提示词
    query = "5000元以内，适合编程开发的轻薄笔记本，最好性价比高一点"

    print("=" * 50)
    print("测试提示词:")
    print(query)
    print("=" * 50)

    # 创建会话
    try:
        resp = requests.post(f"{BASE_URL}/api/sessions", timeout=5)
        session_id = resp.json().get("session_id", "test_session")
        print(f"\n会话ID: {session_id}")
    except Exception as e:
        print(f"创建会话失败: {e}")
        session_id = "test_session"

    # 发送推荐请求
    print("\n发送推荐请求...")
    start_time = time.time()

    try:
        resp = requests.post(
            f"{BASE_URL}/api/chat",
            json={"session_id": session_id, "message": query},
            timeout=600  # 推荐流程可能较长，增加到10分钟
        )
        elapsed_time = time.time() - start_time

        if resp.status_code == 200:
            result = resp.json()
            print(f"\n响应时间: {elapsed_time:.1f} 秒")
            print("\n" + "=" * 50)
            print("推荐结果:")
            print("=" * 50)

            # 解析结果
            top5 = result.get("top5_recommendations", [])
            if top5:
                print(f"\nTop 5 推荐:")
                for i, item in enumerate(top5, 1):
                    name = item.get("name", "未知")
                    reason = item.get("reason", "无理由")
                    print(f"\n{i}. {name}")
                    print(f"   推荐理由: {reason}")

            # 最终推荐
            final = result.get("final_recommendation", {})
            if final:
                print(f"\n首选推荐: {final.get('name', final.get('product_name', '未知'))}")
                reasons = result.get("recommendation_reason", [])
                if reasons:
                    print(f"推荐理由: {'; '.join(reasons)}")

            # 检查知识库信息是否被使用
            print("\n" + "=" * 50)
            print("知识库效果检查:")
            print("=" * 50)

            # 加载知识库数据
            from agent.app.services.knowledge_retriever import get_all_reviews
            reviews = get_all_reviews()
            review_names = {r.get("series_name", "") for r in reviews}

            matched = 0
            for item in top5:
                name = item.get("name", "")
                # 检查是否匹配知识库中的系列
                for series_name in review_names:
                    if series_name in name or name in series_name:
                        matched += 1
                        print(f"✓ '{name}' 匹配知识库系列 '{series_name}'")
                        break

            print(f"\n知识库命中率: {matched}/{len(top5)} ({matched/len(top5)*100:.0f}%)")

            # 检查推荐理由是否包含知识库内容
            print("\n推荐理由分析:")
            for item in top5:
                reason = item.get("reason", "")
                name = item.get("name", "")
                # 检查是否包含典型知识库关键词
                kb_keywords = ["屏幕", "性能", "便携", "性价比", "散热", "续航", "轻薄", "做工"]
                found_keywords = [kw for kw in kb_keywords if kw in reason]
                if found_keywords:
                    print(f"  '{name}' 理由包含知识库关键词: {found_keywords}")

        else:
            print(f"请求失败: {resp.status_code}")
            print(resp.text)

    except requests.exceptions.Timeout:
        print(f"请求超时 (>{time.time() - start_time:.1f}秒)")
    except Exception as e:
        print(f"请求异常: {e}")


if __name__ == "__main__":
    test_recommendation()
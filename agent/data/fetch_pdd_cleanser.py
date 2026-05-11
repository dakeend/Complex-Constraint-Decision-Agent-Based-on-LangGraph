"""
拼多多多多进宝 - 洗面奶商品数据（评论数、好评相关、标签、类目、销量、推广链接）

在 dataupdate 基础上，搜索化妆品-洗面奶，并提取：
- 评论数 (goods_eval_count)
- 描述分 (desc_txt) - 可反映好评程度
- 物流分 (lgst_txt)、服务分 (serv_txt)
- 商品标签 (opt_name)
- 商品类目 (category_name)
- 销量 (sell_count / sales_tip)
- 推广链接 (promotion_url)
"""
import json
import os
import sys
import time

# 复用同目录下 dataupdate 的搜索逻辑
from dataupdate import (
    search_pdd_goods,
    _extract_pdd_items,
    get_pdd_goods_detail,
    generate_pdd_promotion_url,
)


def _extract_review_fields(item: dict) -> dict:
    """从商品项中提取评论数、好评相关、标签、类目等字段"""
    # 评价数：多个可能的字段名
    eval_count = (
        item.get("goods_eval_count")
        or item.get("eval_count")
        or item.get("evaluation_count")
    )
    # 销量：搜索接口可能返回
    sales = item.get("sales_tip") or item.get("sell_count") or item.get("sold") or item.get("sales_num") or item.get("sold_quantity")
    # 描述分（与好评相关，通常 0-5 或百分比）
    desc_txt = item.get("desc_txt")
    lgst_txt = item.get("lgst_txt")
    serv_txt = item.get("serv_txt")
    # 标签：可能是 opt_name、opt_ids 或 opt_id
    opt_name = item.get("opt_name")
    if not opt_name and item.get("opt_ids"):
        opt_name = str(item["opt_ids"])
    if not opt_name:
        opt_name = item.get("opt_id") or ""
    # 类目
    category_name = item.get("category_name") or item.get("cat_name")
    if not category_name and item.get("cat_ids"):
        category_name = str(item["cat_ids"])
    category_id = item.get("category_id") or item.get("cat_id")

    return {
        "goods_id": item.get("goods_id"),
        "goods_name": item.get("goods_name") or item.get("name") or item.get("title", "未知"),
        "price_yuan": (item.get("min_group_price") or item.get("min_normal_price") or 0) / 100,
        "mall_name": item.get("mall_name") or "",
        "评论数": eval_count,
        "描述分": desc_txt,
        "物流分": lgst_txt,
        "服务分": serv_txt,
        "商品标签": opt_name,
        "类目ID": category_id,
        "类目名称": category_name,
        "goods_sign": item.get("goods_sign") or item.get("goodsSign"),
        "销量": sales,
        "推广链接": None,
    }


def main() -> int:
    keyword = os.getenv("PDD_CLEANSER_KEYWORD", "保湿霜")
    page = int(os.getenv("PDD_PAGE", "1"))
    page_size = int(os.getenv("PDD_PAGE_SIZE", "20"))

    print(f"搜索关键词: {keyword}")
    print()

    try:
        result = search_pdd_goods(keyword, page, page_size)
    except Exception as exc:
        print(f"调用失败: {exc}")
        return 1

    items = _extract_pdd_items(result)
    if not items:
        print("未解析到商品列表，输出原始响应：")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    extracted = [_extract_review_fields(i) for i in items]
    goods_signs = [e["goods_sign"] for e in extracted if e.get("goods_sign")]

    # 1. 批量获取商品详情（销量）
    if goods_signs:
        try:
            for i in range(0, len(goods_signs), 20):
                chunk = goods_signs[i : i + 20]
                detail_resp = get_pdd_goods_detail(chunk)
                if "error_response" in detail_resp:
                    print("商品详情接口返回错误:", detail_resp["error_response"].get("sub_msg", ""))
                else:
                    details = (
                        detail_resp.get("goods_detail_response", {}) or
                        detail_resp.get("goods_detail_list", {})
                    )
                    detail_list = details.get("goods_details") or details.get("goods_detail_list") or []
                    for j, d in enumerate(detail_list):
                        idx = i + j
                        if idx < len(extracted):
                            sold = d.get("sell_count") or d.get("sales_tip") or d.get("sold") or d.get("sales_num")
                            if sold is not None:
                                extracted[idx]["销量"] = sold
                time.sleep(0.5)
        except Exception as e:
            print("获取商品详情失败:", e)

        # 2. 批量生成推广链接
        try:
            for i in range(0, len(goods_signs), 20):
                chunk = goods_signs[i : i + 20]
                promo_resp = generate_pdd_promotion_url(chunk)
                if "error_response" in promo_resp:
                    print("推广链接接口返回错误:", promo_resp["error_response"].get("sub_msg", ""))
                else:
                    resp_key = "goods_promotion_url_generate_response"
                    resp = promo_resp.get(resp_key, {})
                    url_list = resp.get("goods_promotion_url_list") or resp.get("url_list") or []
                    for j, u in enumerate(url_list):
                        idx = i + j
                        if idx < len(extracted):
                            url = u.get("url") or u.get("mobile_url") or u.get("short_url") or u.get("mobile_short_url")
                            extracted[idx]["推广链接"] = url
                time.sleep(0.5)
        except Exception as e:
            print("生成推广链接失败:", e)

    print("=" * 70)
    print(f"共 {len(extracted)} 条洗面奶相关商品")
    print("=" * 70)
    for idx, row in enumerate(extracted, 1):
        print(f"\n【{idx}】 {row['goods_name'][:50]}{'...' if len(row['goods_name']) > 50 else ''}")
        print(f"     价格: ¥{row['price_yuan']:.2f}  |  店铺: {row['mall_name']}  |  销量: {row['销量']}")
        print(f"     评论数: {row['评论数']}  |  描述分: {row['描述分']}  |  物流分: {row['物流分']}  |  服务分: {row['服务分']}")
        print(f"     类目: {row['类目名称']}  |  标签: {row['商品标签']}")
        if row.get("推广链接"):
            link = row["推广链接"]
            print(f"     推广链接: {link[:70]}{'...' if len(link) > 70 else ''}")

    out_path = os.path.join(os.path.dirname(__file__), "pdd_cleanser_output.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(extracted, f, ensure_ascii=False, indent=2)
    print(f"\n已保存至: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

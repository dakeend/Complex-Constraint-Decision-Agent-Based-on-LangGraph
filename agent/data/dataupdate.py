import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from typing import Any
from urllib import parse, request


PDD_API_URL = "https://gw-api.pinduoduo.com/api/router"
JD_API_URL = "https://api.jd.com/routerjson"
DEFAULT_PDD_PID = "44151340_314628407"


def _md5_upper(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest().upper()


def _post_form(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    encoded = parse.urlencode(payload).encode("utf-8")
    req = request.Request(url, data=encoded, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    with request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _jd_sign(params: dict[str, Any], secret: str) -> str:
    pieces: list[str] = []
    for key, value in sorted(params.items(), key=lambda item: item[0]):
        if key == "sign" or value is None or value == "":
            continue
        pieces.append(f"{key}{value}")
    return _md5_upper(f"{secret}{''.join(pieces)}{secret}")


def _pdd_sign(params: dict[str, Any], secret: str) -> str:
    pieces: list[str] = []
    for key, value in sorted(params.items(), key=lambda item: item[0]):
        if key == "sign" or value is None or value == "":
            continue
        pieces.append(f"{key}{value}")
    return _md5_upper(f"{secret}{''.join(pieces)}{secret}")


def search_pdd_goods(keyword: str, page: int, page_size: int) -> dict[str, Any]:
    client_id = os.getenv("PDD_CLIENT_ID")
    client_secret = os.getenv("PDD_CLIENT_SECRET")
    access_token = os.getenv("PDD_ACCESS_TOKEN")
    pid = os.getenv("PDD_PID", DEFAULT_PDD_PID)

    if not client_id or not client_secret:
        raise RuntimeError(
            "缺少拼多多凭证。请先设置环境变量 PDD_CLIENT_ID 和 PDD_CLIENT_SECRET。"
        )
    if not access_token:
        raise RuntimeError(
            "缺少 PDD_ACCESS_TOKEN。多多进宝搜索通常还需要 access token，"
            "仅有 client_id/client_secret 通常不足以直接发起真实搜索。"
        )

    payload: dict[str, Any] = {
        "type": "pdd.ddk.goods.search",
        "client_id": client_id,
        "access_token": access_token,
        "timestamp": int(time.time()),
        "data_type": "JSON",
        "keyword": keyword,
        "page": page,
        "page_size": page_size,
        "pid": pid,
        "with_coupon": "true",
    }
    payload["sign"] = _pdd_sign(payload, client_secret)
    return _post_form(PDD_API_URL, payload)


def get_pdd_goods_detail(goods_sign_list: list[str]) -> dict[str, Any]:
    """商品详情，用于获取销量等字段。goods_sign_list 最多约 20 个。"""
    client_id = os.getenv("PDD_CLIENT_ID")
    client_secret = os.getenv("PDD_CLIENT_SECRET")
    access_token = os.getenv("PDD_ACCESS_TOKEN")
    pid = os.getenv("PDD_PID", DEFAULT_PDD_PID)

    if not client_id or not client_secret or not access_token:
        raise RuntimeError("缺少 PDD 凭证，请设置 PDD_CLIENT_ID、PDD_CLIENT_SECRET、PDD_ACCESS_TOKEN")

    payload: dict[str, Any] = {
        "type": "pdd.ddk.goods.detail",
        "client_id": client_id,
        "access_token": access_token,
        "timestamp": int(time.time()),
        "data_type": "JSON",
        "goods_sign_list": json.dumps(goods_sign_list, ensure_ascii=False),
        "pid": pid,
    }
    payload["sign"] = _pdd_sign(payload, client_secret)
    return _post_form(PDD_API_URL, payload)


def generate_pdd_promotion_url(goods_sign_list: list[str]) -> dict[str, Any]:
    """生成商品推广链接。goods_sign_list 最多约 20 个。"""
    client_id = os.getenv("PDD_CLIENT_ID")
    client_secret = os.getenv("PDD_CLIENT_SECRET")
    access_token = os.getenv("PDD_ACCESS_TOKEN")
    pid = os.getenv("PDD_PID", DEFAULT_PDD_PID)

    if not client_id or not client_secret or not access_token:
        raise RuntimeError("缺少 PDD 凭证，请设置 PDD_CLIENT_ID、PDD_CLIENT_SECRET、PDD_ACCESS_TOKEN")

    payload: dict[str, Any] = {
        "type": "pdd.ddk.goods.promotion.url.generate",
        "client_id": client_id,
        "access_token": access_token,
        "timestamp": int(time.time()),
        "data_type": "JSON",
        "goods_sign_list": json.dumps(goods_sign_list, ensure_ascii=False),
        "p_id": pid,
        "generate_short_url": "true",
    }
    payload["sign"] = _pdd_sign(payload, client_secret)
    return _post_form(PDD_API_URL, payload)


def search_jd_goods(keyword: str, page: int, page_size: int) -> dict[str, Any]:
    app_key = os.getenv("JD_APP_KEY")
    app_secret = os.getenv("JD_APP_SECRET")

    if not app_key or not app_secret:
        raise RuntimeError(
            "缺少京东联盟凭证。请设置 JD_APP_KEY 和 JD_APP_SECRET。"
            "你发来的那一串值更像单个密钥，京东联盟通常至少需要一对 app_key/app_secret。"
        )

    param_json = {
        "goodsReqDTO": {
            "keyword": keyword,
            "pageIndex": page,
            "pageSize": page_size,
        }
    }
    payload: dict[str, Any] = {
        "method": "jd.union.open.goods.search",
        "app_key": app_key,
        "sign_method": "md5",
        "format": "json",
        "v": "1.0",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "param_json": json.dumps(param_json, ensure_ascii=False, separators=(",", ":")),
    }
    payload["sign"] = _jd_sign(payload, app_secret)
    return _post_form(JD_API_URL, payload)


def _extract_pdd_items(result: dict[str, Any]) -> list[dict[str, Any]]:
    response = result.get("goods_search_response", {})
    return response.get("goods_list", [])


def _extract_jd_items(result: dict[str, Any]) -> list[dict[str, Any]]:
    for key, value in result.items():
        if "goods" not in key.lower():
            continue
        if isinstance(value, dict):
            raw_result = value.get("result")
            if isinstance(raw_result, str):
                try:
                    parsed = json.loads(raw_result)
                except json.JSONDecodeError:
                    continue
                for items_key in ("data", "goodsList", "result"):
                    items = parsed.get(items_key)
                    if isinstance(items, list):
                        return items
                    if isinstance(items, dict):
                        for nested_key in ("data", "goodsList", "items"):
                            nested_items = items.get(nested_key)
                            if isinstance(nested_items, list):
                                return nested_items
            if isinstance(value.get("data"), list):
                return value["data"]
    return []


def _print_preview(platform: str, items: list[dict[str, Any]]) -> None:
    if not items:
        print("没有解析出商品列表，下面会打印原始响应，便于你继续对字段做适配。")
        return

    print(f"成功解析到 {len(items)} 条 {platform} 商品结果，预览前 5 条：")
    for index, item in enumerate(items[:5], start=1):
        title = (
            item.get("goods_name")
            or item.get("skuName")
            or item.get("wareName")
            or item.get("name")
            or item.get("title")
            or "未知标题"
        )
        price = (
            item.get("min_group_price")
            or item.get("lowestPrice")
            or item.get("price")
            or item.get("unitPrice")
        )
        coupon = item.get("coupon_discount") or item.get("couponInfo") or item.get("coupon")
        print(f"{index}. {title}")
        print(f"   price={price} coupon={coupon}")


def main() -> int:
    parser = argparse.ArgumentParser(description="测试拼多多/京东商品搜索接口。")
    parser.add_argument("--platform", choices=["pdd", "jd"], required=True)
    parser.add_argument("--keyword", required=True, help="搜索关键词，例如：轻薄本")
    parser.add_argument("--page", type=int, default=1)
    parser.add_argument("--page-size", type=int, default=10)
    parser.add_argument(
        "--raw",
        action="store_true",
        help="打印完整原始响应，便于你调字段。",
    )
    args = parser.parse_args()

    try:
        if args.platform == "pdd":
            result = search_pdd_goods(args.keyword, args.page, args.page_size)
            items = _extract_pdd_items(result)
        else:
            result = search_jd_goods(args.keyword, args.page, args.page_size)
            items = _extract_jd_items(result)
    except Exception as exc:  # noqa: BLE001
        print(f"调用失败: {exc}")
        return 1

    _print_preview(args.platform, items)
    if args.raw or not items:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

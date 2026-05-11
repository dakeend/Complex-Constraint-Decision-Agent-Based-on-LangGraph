"""
拼多多多多进宝 - 生成 PID 授权备案链接

使用 pdd.ddk.rp.prom.url.generate 接口，channel_type=10 表示生成绑定备案链接。
备案成功后，该 PID 即可用于 pdd.ddk.goods.search 等接口。

使用步骤：
1. 设置环境变量：PDD_CLIENT_ID、PDD_CLIENT_SECRET、PDD_ACCESS_TOKEN
2. 可选：PDD_PID（默认 44151340_314628407）
3. 运行脚本，会打印备案链接
4. 用手机微信或浏览器打开该链接，完成授权
5. 备案成功后即可正常调用商品搜索接口
"""
import json
import os
import sys
import time
from typing import Any
from urllib import parse, request
from urllib.error import HTTPError, URLError


PDD_API_URL = "https://gw-api.pinduoduo.com/api/router"
DEFAULT_PDD_PID = "44151340_314628407"


def _md5_upper(text: str) -> str:
    import hashlib
    return hashlib.md5(text.encode("utf-8")).hexdigest().upper()


def _pdd_sign(params: dict[str, Any], secret: str) -> str:
    pieces: list[str] = []
    for key, value in sorted(params.items(), key=lambda item: item[0]):
        if key == "sign" or value is None or value == "":
            continue
        pieces.append(f"{key}{value}")
    return _md5_upper(f"{secret}{''.join(pieces)}{secret}")


def _post_form(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    encoded = parse.urlencode(payload).encode("utf-8")
    req = request.Request(url, data=encoded, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    with request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def generate_filing_url(
    pid: str,
    custom_parameters: str | None = None,
) -> dict[str, Any]:
    """
    调用 pdd.ddk.rp.prom.url.generate 生成备案链接。
    channel_type=10 表示生成绑定备案链接。
    """
    client_id = os.getenv("PDD_CLIENT_ID")
    client_secret = os.getenv("PDD_CLIENT_SECRET")
    access_token = os.getenv("PDD_ACCESS_TOKEN")

    if not client_id or not client_secret:
        raise RuntimeError("请设置环境变量 PDD_CLIENT_ID 和 PDD_CLIENT_SECRET")
    if not access_token:
        raise RuntimeError("请先运行 get_pdd_token.py 获取 access_token，并设置 PDD_ACCESS_TOKEN")

    # p_id_list 为 JSON 数组字符串，如 '["44151340_314628407"]'
    p_id_list = json.dumps([pid], ensure_ascii=False)

    payload: dict[str, Any] = {
        "type": "pdd.ddk.rp.prom.url.generate",
        "client_id": client_id,
        "access_token": access_token,
        "timestamp": int(time.time()),
        "data_type": "JSON",
        "p_id_list": p_id_list,
        "channel_type": "10",  # 10 = 绑定备案
    }
    if custom_parameters:
        payload["custom_parameters"] = custom_parameters

    payload["sign"] = _pdd_sign(payload, client_secret)
    return _post_form(PDD_API_URL, payload)


def main() -> int:
    pid = os.getenv("PDD_PID", DEFAULT_PDD_PID)
    # 如不需要 custom_parameters 跟单，可只备案 pid，不传此项
    custom_params = os.getenv("PDD_CUSTOM_PARAMS")  # 可选，如 '{"uid":"user123"}'

    print(f"PID: {pid}")
    if custom_params:
        print(f"custom_parameters: {custom_params}")
    print()

    try:
        result = generate_filing_url(pid, custom_params)
    except (HTTPError, URLError) as e:
        print(f"请求失败: {e}")
        return 1

    if "error_response" in result:
        err = result["error_response"]
        print(f"接口返回错误: {err.get('error_msg', '')} - {err.get('sub_msg', '')}")
        return 1

    # 解析返回的推广链接
    resp_key = "rp_promotion_url_generate_response"
    if resp_key not in result:
        print("无法解析返回结构，原始响应：")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1

    resp = result[resp_key]
    url_list = resp.get("url_list", [])
    if not url_list:
        print("未返回备案链接，请检查接口返回。原始响应：")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1

    filing_url = url_list[0].get("url") or url_list[0].get("mobile_url")
    if not filing_url:
        filing_url = url_list[0].get("we_app_info", {}).get("page_path")
        if filing_url:
            filing_url = f"（小程序路径）{filing_url}"
        else:
            filing_url = str(url_list[0])

    print("===== 备案链接已生成 =====")
    print()
    print(filing_url)
    print()
    print("请用手机微信或浏览器打开上述链接，完成授权备案。")
    print("备案成功后，即可使用该 PID 调用商品搜索等接口。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
拼多多开放平台 OAuth 2.0：获取 code 并换取 access_token

使用步骤：
1. 在拼多多开放平台应用里配置回调地址，例如：http://127.0.0.1:8080/callback
2. 设置环境变量：PDD_CLIENT_ID、PDD_CLIENT_SECRET、PDD_REDIRECT_URI
3. 运行脚本，按提示在浏览器中授权
4. 授权完成后会自动获取 access_token 并打印
"""
import json
import os
import sys
import urllib.parse
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


# 拼多多 OAuth 相关地址
# 多多进宝：https://jinbao.pinduoduo.com/open.html
# 商家后台：https://mai.pinduoduo.com/open.html
AUTH_URL_JINBAO = "https://jinbao.pinduoduo.com/open.html"
AUTH_URL_MAI = "https://mai.pinduoduo.com/open.html"
TOKEN_URL = "https://open-api.pinduoduo.com/oauth/token"

# 本地回调服务端口
CALLBACK_PORT = 8080
CALLBACK_PATH = "/callback"


class CallbackHandler(BaseHTTPRequestHandler):
    """接收拼多多回调，从中取出 code"""

    code = None

    def log_message(self, format, *args):
        """减少默认日志输出"""
        pass

    def do_GET(self):
        if not self.path.startswith(CALLBACK_PATH):
            self.send_error(404)
            return

        query = urllib.parse.urlparse(self.path).query
        params = urllib.parse.parse_qs(query)
        code_list = params.get("code", [])
        error_list = params.get("error", [])

        if code_list:
            CallbackHandler.code = code_list[0]
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            body = (
                "<html><body><h2>授权成功</h2>"
                "<p>已收到 code，正在换取 access_token，请关闭此窗口并回到终端查看结果。</p>"
                "</body></html>"
            ).encode("utf-8")
            self.wfile.write(body)
        elif error_list:
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            msg = error_list[0] if error_list else "未知错误"
            body = f"<html><body><h2>授权失败</h2><p>{msg}</p></body></html>".encode("utf-8")
            self.wfile.write(body)
        else:
            self.send_error(400, "缺少 code 参数")


def get_auth_url(
    client_id: str,
    redirect_uri: str,
    state: str = "pdd_oauth",
    use_jinbao: bool = True,
) -> str:
    """生成授权 URL。use_jinbao=True 为多多进宝，False 为商家后台"""
    base = AUTH_URL_JINBAO if use_jinbao else AUTH_URL_MAI
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "state": state,
    }
    return f"{base}?{urllib.parse.urlencode(params)}"


def exchange_code_for_token(
    client_id: str,
    client_secret: str,
    code: str,
    redirect_uri: str,
) -> dict:
    """用 code 换取 access_token"""
    payload = {
        "client_id": client_id,
        "client_secret": client_secret,
        "code": code,
        "grant_type": "authorization_code",
        "redirect_uri": redirect_uri,
    }
    data = json.dumps(payload).encode("utf-8")
    req = Request(TOKEN_URL, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    with urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    client_id = os.getenv("PDD_CLIENT_ID")
    client_secret = os.getenv("PDD_CLIENT_SECRET")
    redirect_uri = os.getenv(
        "PDD_REDIRECT_URI",
        f"http://127.0.0.1:{CALLBACK_PORT}{CALLBACK_PATH}",
    )

    if not client_id or not client_secret:
        print("请先设置环境变量：PDD_CLIENT_ID、PDD_CLIENT_SECRET")
        print("可选：PDD_REDIRECT_URI（默认 http://127.0.0.1:8080/callback）")
        return 1

    # 确保 redirect_uri 与开放平台应用里配置的回调地址完全一致
    print(f"回调地址 redirect_uri = {redirect_uri}")
    print("请确认该地址已在拼多多开放平台应用的回调配置中填写。\n")

    # 默认多多进宝；设置 PDD_AUTH_TYPE=mai 使用商家授权
    use_jinbao = os.getenv("PDD_AUTH_TYPE", "jinbao").lower() != "mai"

    # 方式 1：自动打开浏览器 + 本地服务器接收回调
    use_server = "--no-server" not in sys.argv
    if use_server:
        print("即将打开浏览器，请在页面中完成授权...")
        auth_url = get_auth_url(client_id, redirect_uri, use_jinbao=use_jinbao)
        print(f"授权地址：{auth_url}\n")
        webbrowser.open(auth_url)

        server = HTTPServer(("127.0.0.1", CALLBACK_PORT), CallbackHandler)
        print(f"本地回调服务已启动：http://127.0.0.1:{CALLBACK_PORT}{CALLBACK_PATH}")
        print("等待授权回调（授权完成后会自动继续）...\n")
        server.handle_request()
        code = CallbackHandler.code
    else:
        # 方式 2：手动输入 code
        auth_url = get_auth_url(client_id, redirect_uri, use_jinbao=use_jinbao)
        print("请复制以下链接到浏览器打开并完成授权：")
        print(auth_url)
        print("\n授权后，浏览器会跳转到回调地址，请从地址栏复制 code 参数的值。")
        code = input("请输入 code：").strip()

    if not code:
        print("未获取到 code，请重试。")
        return 1

    print("正在用 code 换取 access_token...")
    try:
        result = exchange_code_for_token(client_id, client_secret, code, redirect_uri)
    except HTTPError as e:
        body = e.read().decode("utf-8") if e.fp else ""
        print(f"请求失败 HTTP {e.code}: {body}")
        return 1
    except URLError as e:
        print(f"请求失败: {e.reason}")
        return 1

    if "error" in result or "error_response" in result:
        err = result.get("error", result.get("error_response", result))
        print(f"接口返回错误: {err}")
        return 1

    access_token = result.get("access_token")
    expires_in = result.get("expires_in")
    refresh_token = result.get("refresh_token")

    print("\n===== 换取成功 =====")
    print(f"access_token:  {access_token}")
    print(f"expires_in:    {expires_in} 秒")
    print(f"refresh_token: {refresh_token}")
    print("\n建议将 access_token 设为环境变量：")
    print(f'  $env:PDD_ACCESS_TOKEN="{access_token}"')
    return 0


if __name__ == "__main__":
    sys.exit(main())

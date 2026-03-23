"""FastAPI 入口"""
import os
import uuid
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# 加载 .env，必须在导入 LangChain/LangGraph 之前设置 LangSmith 环境变量
load_dotenv(Path(__file__).resolve().parents[2] / ".env")
os.environ.setdefault("LANGSMITH_TRACING", "true")
os.environ.setdefault("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
os.environ.setdefault("LANGSMITH_PROJECT", "商品智能推荐")

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .schemas import AgentResponse, CandidateProduct, ChatRequest, SessionInfo, Top5Recommendation, UserQuery
from .workflow import build_graph

app = FastAPI(title="商品选购辅助智能体")
graph = build_graph()

# 静态文件
STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# 会话存储（内存，重启后清空）
_sessions: dict[str, dict] = {}


def _format_response(r: AgentResponse) -> str:
    """将 AgentResponse 转为可读文本"""
    parts = []
    if r.clarifying_questions:
        parts.append("**需要补充的信息：**\n")
        for q in r.clarifying_questions:
            parts.append(f"- {q}")
        parts.append("")
    # 优先展示深度检索 Top5 全部结果
    if r.top5_recommendations:
        parts.append("## 深度检索推荐 Top5\n")
        for i, rec in enumerate(r.top5_recommendations, 1):
            parts.append(f"### {i}. {rec.name}\n")
            if rec.reason:
                parts.append(f"**推荐原因：** {rec.reason}\n")
            if rec.purchase_url:
                parts.append(f"[多多进宝购买链接]({rec.purchase_url})\n")
            parts.append("")
    if r.final_recommendation:
        p = r.final_recommendation
        parts.append("## 首选推荐\n")
        parts.append(f"**{p.name}**\n")
        parts.append(f"**价格：** ¥{p.price}\n")
        if p.brand:
            parts.append(f"**品牌：** {p.brand}\n")
        if p.cpu_model or p.gpu_model:
            specs = " / ".join(filter(None, [p.cpu_model, p.gpu_model]))
            parts.append(f"**配置：** {specs}\n")
        if p.memory_gb or p.storage_gb:
            parts.append(f"**内存/存储：** {p.memory_gb}GB / {p.storage_gb}GB\n")
        if p.purchase_url:
            parts.append(f"\n[点击购买]({p.purchase_url})\n")
        if r.recommendation_reason:
            parts.append("\n**推荐理由：**\n")
            for x in r.recommendation_reason:
                parts.append(f"- {x}")
    elif r.candidates:
        parts.append("**候选商品：**\n")
        for i, p in enumerate(r.candidates[:5], 1):
            parts.append(f"{i}. {p.name} - ¥{p.price}")
    elif not r.top5_recommendations:
        parts.append("暂未找到符合条件的商品，请补充更多需求后重试。")
    return "\n".join(parts)


def _to_candidate(item: dict) -> CandidateProduct:
    return CandidateProduct(
        product_id=str(item.get("product_id", item.get("goods_sign", ""))),
        name=item.get("name", item.get("product_name", "")),
        brand=item.get("brand", ""),
        price=item.get("price", 0),
        cpu_model=item.get("cpu_model", ""),
        gpu_model=item.get("gpu_model", ""),
        memory_gb=item.get("memory_gb", 0),
        storage_gb=item.get("storage_gb", 0),
        weight_kg=item.get("weight_kg", 0),
        battery_hours=item.get("battery_hours", 0),
        total_score=item.get("total_score", 0),
        matched_constraints=item.get("matched_constraints", []),
        violated_constraints=item.get("violated_constraints", []),
        evidence_count=item.get("evidence_count", 0),
        risks=item.get("risks", item.get("violated_constraints", [])),
        purchase_url=item.get("purchase_url"),
        is_fallback=item.get("is_fallback", False),
    )


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/sessions")
def list_sessions():
    items = sorted(
        _sessions.values(),
        key=lambda x: x["created_at"],
        reverse=True,
    )
    return {
        "sessions": [
            SessionInfo(id=s["id"], title=s["title"], created_at=s["created_at"])
            for s in items
        ]
    }


@app.post("/api/sessions")
def create_session():
    sid = str(uuid.uuid4())
    _sessions[sid] = {
        "id": sid,
        "title": "新对话",
        "created_at": datetime.now().isoformat(),
        "messages": [],
    }
    return {"session": _sessions[sid]}


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str):
    if session_id not in _sessions:
        return {"messages": []}
    s = _sessions[session_id]
    return {"messages": s["messages"]}


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str):
    _sessions.pop(session_id, None)
    return {"ok": True}


@app.post("/api/chat")
def chat(payload: ChatRequest):
    sid = payload.session_id
    if not sid or sid not in _sessions:
        sid = str(uuid.uuid4())
        _sessions[sid] = {
            "id": sid,
            "title": payload.message[:30] + ("…" if len(payload.message) > 30 else ""),
            "created_at": datetime.now().isoformat(),
            "messages": [],
        }
    sess = _sessions[sid]
    # 追加当前用户消息
    sess["messages"].append({"role": "user", "content": payload.message})
    if len(sess["messages"]) == 1:
        sess["title"] = payload.message[:30] + ("…" if len(payload.message) > 30 else "")

    # 构造对话历史，包含本轮之前的若干条消息
    history = sess["messages"][:-1]
    # 仅保留最近若干条，避免上下文过长
    history = history[-10:]

    # 为了让「补充回答」也能生效，将最近两轮用户输入拼在一起作为 query_text
    user_utts = [m["content"] for m in history if m.get("role") == "user"]
    user_utts.append(payload.message)
    merged_query = "。".join(user_utts[-2:]) if len(user_utts) >= 2 else payload.message

    state_in = {
        "query_text": merged_query,
        "user_id": None,
        "history": history,
    }
    result = graph.invoke(state_in)
    candidates = [_to_candidate(c) for c in result.get("candidates", [])]
    final_rec = result.get("final_recommendation")
    final_candidate = _to_candidate(final_rec) if final_rec else None
    top5_raw = result.get("top5_recommendations", [])
    top5 = [Top5Recommendation(name=t.get("name", ""), reason=t.get("reason", ""), purchase_url=t.get("purchase_url")) for t in top5_raw]
    resp = AgentResponse(
        task_summary=f"商品选购推荐：{payload.message}",
        extracted_constraints=result.get("extracted_constraints", {}),
        missing_info=result.get("missing_info", []),
        clarifying_questions=result.get("clarifying_questions", []),
        candidates=candidates,
        final_recommendation=final_candidate,
        recommendation_reason=result.get("recommendation_reason", []),
        risk_explanations=result.get("risk_explanations", []),
        purchase_links=result.get("purchase_links", []),
        top5_recommendations=top5,
    )
    formatted = _format_response(resp)
    sess["messages"].append({"role": "assistant", "content": formatted})
    return {
        "session_id": sid,
        "response": resp.model_dump(),
        "formatted_response": formatted,
    }


@app.post("/recommend", response_model=AgentResponse)
def recommend(payload: UserQuery):
    state_in = {
        "query_text": payload.query_text,
        "user_id": payload.user_id,
    }
    result = graph.invoke(state_in)

    candidates = [_to_candidate(c) for c in result.get("candidates", [])]
    final_rec = result.get("final_recommendation")
    final_candidate = _to_candidate(final_rec) if final_rec else None

    top5_raw = result.get("top5_recommendations", [])
    top5 = [Top5Recommendation(name=t.get("name", ""), reason=t.get("reason", ""), purchase_url=t.get("purchase_url")) for t in top5_raw]
    return AgentResponse(
        task_summary=f"商品选购推荐：{payload.query_text}",
        extracted_constraints=result.get("extracted_constraints", {}),
        missing_info=result.get("missing_info", []),
        clarifying_questions=result.get("clarifying_questions", []),
        candidates=candidates,
        final_recommendation=final_candidate,
        recommendation_reason=result.get("recommendation_reason", []),
        risk_explanations=result.get("risk_explanations", []),
        purchase_links=result.get("purchase_links", []),
        top5_recommendations=top5,
    )

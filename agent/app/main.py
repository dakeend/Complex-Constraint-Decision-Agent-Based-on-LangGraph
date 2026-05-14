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
from .services.session_store import save_session, load_session, list_sessions as db_list_sessions, delete_session as db_delete_session
from .workflow import build_graph

app = FastAPI(title="商品选购辅助智能体")
graph = build_graph()

# 静态文件
STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


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
        for i, rec in enumerate(r.top5_recommendations, 1):
            parts.append(f"{i}. {rec.name}\n")
            if rec.reason:
                parts.append(f"推荐原因：{rec.reason}\n")
            if rec.purchase_url:
                parts.append(f"[多多进宝购买链接]({rec.purchase_url})\n")
            parts.append("")
    if r.final_recommendation:
        p = r.final_recommendation
        parts.append("## 首选推荐\n")
        parts.append(f"**{p.name}**\n")
        if p.price and p.price > 0:
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
def list_sessions_api():
    items = db_list_sessions()
    return {
        "sessions": [
            SessionInfo(id=s["id"], title=s["title"], created_at=s["created_at"])
            for s in items
        ]
    }


@app.post("/api/sessions")
def create_session():
    sid = str(uuid.uuid4())
    now = datetime.now().isoformat()
    save_session(sid, "新对话", now, [], {})
    return {"session": {"id": sid, "title": "新对话", "created_at": now}}


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str):
    s = load_session(session_id)
    if not s:
        return {"messages": []}
    return {"messages": s.get("messages", [])}


@app.delete("/api/sessions/{session_id}")
def delete_session_api(session_id: str):
    db_delete_session(session_id)
    return {"ok": True}


@app.post("/api/chat")
def chat(payload: ChatRequest):
    sid = payload.session_id
    now = datetime.now().isoformat()

    # 从 SQLite 加载会话
    sess = load_session(sid) if sid else None
    if not sess:
        sid = str(uuid.uuid4())
        title = payload.message[:30] + ("…" if len(payload.message) > 30 else "")
        sess = {"id": sid, "title": title, "created_at": now, "messages": [], "last_state": {}}
    else:
        if len(sess.get("messages", [])) == 0:
            sess["title"] = payload.message[:30] + ("…" if len(payload.message) > 30 else "")

    messages = sess.get("messages", [])

    # 追加当前用户消息
    messages.append({"role": "user", "content": payload.message})

    # 构造对话历史
    history = messages[:-1][-10:]

    # 获取上一次推荐的 state，供追问节点使用
    prev_state = sess.get("last_state", {})

    state_in = {
        **prev_state,
        "query_text": payload.message,
        "user_id": None,
        "history": history,
    }
    try:
        result = graph.invoke(state_in)
    except Exception as exc:
        err_msg = f"推荐流程执行失败，请稍后重试。错误信息：{exc}"
        if "rate_limit" in str(exc).lower() or "429" in str(exc):
            err_msg = "API 调用频率已达上限，请等待几分钟后再试。"
        messages.append({"role": "assistant", "content": err_msg})
        save_session(
            sid,
            sess.get("title", "新对话"),
            sess.get("created_at", now),
            messages,
            {},
        )
        return {
            "session_id": sid,
            "formatted_response": err_msg,
        }

    # 判断是否为追问轮次
    followup_answer = result.get("followup_answer", "")
    if followup_answer:
        formatted = followup_answer
    else:
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
            followup_answer=None,
        )
        formatted = _format_response(resp)

    messages.append({"role": "assistant", "content": formatted})

    # 持久化会话和推荐状态
    save_session(
        sid,
        sess.get("title", "新对话"),
        sess.get("created_at", now),
        messages,
        result,
    )

    return {
        "session_id": sid,
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

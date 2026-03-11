from fastapi import FastAPI

from .schemas import AgentResponse, CandidateProduct, UserQuery
from .workflow import build_graph


app = FastAPI(title="Notebook Recommender Agent")
graph = build_graph()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/recommend", response_model=AgentResponse)
def recommend(payload: UserQuery):
    result = graph.invoke({"query_text": payload.query_text})
    candidates = [CandidateProduct(**item) for item in result.get("candidates", [])]
    final_recommendation = (
        CandidateProduct(**result["final_recommendation"])
        if result.get("final_recommendation")
        else None
    )

    return AgentResponse(
        task_summary=f"为请求生成笔记本推荐：{payload.query_text}",
        extracted_constraints=result.get("extracted_constraints", {}),
        missing_info=result.get("missing_info", []),
        clarifying_questions=result.get("clarifying_questions", []),
        candidates=candidates,
        final_recommendation=final_recommendation,
        recommendation_reason=result.get("recommendation_reason", []),
    )

import json
from pathlib import Path

from .state import AgentState


DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "products.json"


def _load_products() -> list[dict]:
    with DATA_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_query(state: AgentState) -> AgentState:
    query = state["query_text"]
    constraints: dict[str, object] = {
        "budget_max": None,
        "usage_scenarios": [],
        "brand_avoid": [],
        "need_portability": "轻" in query or "便携" in query,
        "need_dedicated_gpu": "独显" in query or "游戏" in query,
        "memory_min_gb": 16 if "16" in query else None,
    }

    if "编程" in query or "代码" in query:
        constraints["usage_scenarios"].append("编程")
    if "游戏" in query:
        constraints["usage_scenarios"].append("游戏")
    if "联想" in query and ("不要" in query or "排除" in query):
        constraints["brand_avoid"].append("联想")

    for token in query.replace("元", " ").replace("以内", " ").split():
        if token.isdigit():
            value = int(token)
            if 3000 <= value <= 30000:
                constraints["budget_max"] = value
                break

    return {"extracted_constraints": constraints}


def check_missing_info(state: AgentState) -> AgentState:
    constraints = state["extracted_constraints"]
    missing_info: list[str] = []
    questions: list[str] = []

    if not constraints.get("budget_max"):
        missing_info.append("budget_max")
        questions.append("你的预算上限大概是多少？")

    if not constraints.get("usage_scenarios"):
        missing_info.append("usage_scenarios")
        questions.append("你主要是拿来做编程、办公还是游戏？")

    return {"missing_info": missing_info, "clarifying_questions": questions}


def retrieve_products(state: AgentState) -> AgentState:
    constraints = state["extracted_constraints"]
    products = _load_products()
    candidates: list[dict] = []

    for product in products:
        if constraints.get("budget_max") and product["price"] > constraints["budget_max"]:
            continue
        if product["brand"] in constraints.get("brand_avoid", []):
            continue
        if constraints.get("memory_min_gb") and product["memory_gb"] < constraints["memory_min_gb"]:
            continue
        candidates.append(product)

    return {"candidates": candidates[:5]}


def score_candidates(state: AgentState) -> AgentState:
    constraints = state["extracted_constraints"]
    scored: list[dict] = []

    for item in state.get("candidates", []):
        score = 0.0
        matched: list[str] = []
        violated: list[str] = []

        if constraints.get("budget_max") and item["price"] <= constraints["budget_max"]:
            score += 30
            matched.append("预算内")
        else:
            violated.append("超预算")

        if "编程" in constraints.get("usage_scenarios", []):
            score += 20
            matched.append("适合编程")

        if constraints.get("need_portability"):
            if item["weight_kg"] <= 1.8:
                score += 20
                matched.append("便携性较好")
            else:
                violated.append("偏重")

        if constraints.get("need_dedicated_gpu"):
            if item["gpu_model"] != "核显":
                score += 20
                matched.append("具备独显能力")
            else:
                violated.append("显卡能力不足")

        if item["battery_hours"] >= 8:
            score += 10
            matched.append("续航尚可")

        item = {
            **item,
            "total_score": score,
            "matched_constraints": matched,
            "violated_constraints": violated,
        }
        scored.append(item)

    scored.sort(key=lambda x: x["total_score"], reverse=True)
    return {
        "candidates": scored[:3],
        "final_recommendation": scored[0] if scored else None,
    }


def generate_answer(state: AgentState) -> AgentState:
    recommendation = state.get("final_recommendation")
    reasons: list[str] = []

    if recommendation:
        reasons = recommendation.get("matched_constraints", [])[:]
        reasons.append("这是当前 mock 数据下综合评分最高的候选。")

    return {"recommendation_reason": reasons}

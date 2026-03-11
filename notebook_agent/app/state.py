from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    query_text: str
    extracted_constraints: dict[str, Any]
    missing_info: list[str]
    clarifying_questions: list[str]
    candidates: list[dict[str, Any]]
    final_recommendation: dict[str, Any] | None
    recommendation_reason: list[str]

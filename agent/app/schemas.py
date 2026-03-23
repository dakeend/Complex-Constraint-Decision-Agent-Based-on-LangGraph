"""Pydantic 模型"""
from typing import Any

from pydantic import BaseModel, Field


class CommonFields(BaseModel):
    budget_min: int | None = None
    budget_max: int | None = None
    usage_scenarios: list[str] = Field(default_factory=list)
    brand_preference: list[str] = Field(default_factory=list)
    brand_avoid: list[str] = Field(default_factory=list)
    category: str | None = None
    keyword: str | None = None


class LaptopFields(BaseModel):
    need_portability: bool | None = None
    need_dedicated_gpu: bool | None = None
    memory_min_gb: int | None = None
    storage_min_gb: int | None = None


class UserQuery(BaseModel):
    query_text: str
    user_id: str | None = None
    common_fields: CommonFields = Field(default_factory=CommonFields)
    category_fields: LaptopFields = Field(default_factory=LaptopFields)


class CandidateProduct(BaseModel):
    product_id: str
    name: str
    brand: str = ""
    price: int | float = 0
    cpu_model: str = ""
    gpu_model: str = ""
    memory_gb: int = 0
    storage_gb: int = 0
    weight_kg: float = 0
    battery_hours: float = 0
    total_score: float = 0.0
    matched_constraints: list[str] = Field(default_factory=list)
    violated_constraints: list[str] = Field(default_factory=list)
    evidence_count: int = 0
    risks: list[str] = Field(default_factory=list)
    purchase_url: str | None = None
    is_fallback: bool = False


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str


class SessionInfo(BaseModel):
    id: str
    title: str
    created_at: str


class Top5Recommendation(BaseModel):
    """深度检索 Top5 单项：商品名、推荐原因、多多进宝链接"""
    name: str = ""
    reason: str = ""
    purchase_url: str | None = None


class AgentResponse(BaseModel):
    task_summary: str
    extracted_constraints: dict[str, Any]
    missing_info: list[str] = Field(default_factory=list)
    clarifying_questions: list[str] = Field(default_factory=list)
    candidates: list[CandidateProduct] = Field(default_factory=list)
    final_recommendation: CandidateProduct | None = None
    recommendation_reason: list[str] = Field(default_factory=list)
    risk_explanations: list[str] = Field(default_factory=list)
    purchase_links: list[dict[str, str]] = Field(default_factory=list)
    # 深度检索 Top5 全部结果（含推荐原因、多多进宝链接）
    top5_recommendations: list[Top5Recommendation] = Field(default_factory=list)

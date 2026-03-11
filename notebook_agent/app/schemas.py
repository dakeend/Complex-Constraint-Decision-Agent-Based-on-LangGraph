from typing import Any

from pydantic import BaseModel, Field


class CommonFields(BaseModel):
    budget_min: int | None = None
    budget_max: int | None = None
    usage_scenarios: list[str] = Field(default_factory=list)
    brand_preference: list[str] = Field(default_factory=list)
    brand_avoid: list[str] = Field(default_factory=list)


class LaptopFields(BaseModel):
    need_portability: bool | None = None
    need_dedicated_gpu: bool | None = None
    memory_min_gb: int | None = None
    storage_min_gb: int | None = None


class UserQuery(BaseModel):
    query_text: str
    common_fields: CommonFields = Field(default_factory=CommonFields)
    category_fields: LaptopFields = Field(default_factory=LaptopFields)


class CandidateProduct(BaseModel):
    product_id: str
    name: str
    brand: str
    price: int
    cpu_model: str
    gpu_model: str
    memory_gb: int
    storage_gb: int
    weight_kg: float
    battery_hours: float
    total_score: float = 0.0
    matched_constraints: list[str] = Field(default_factory=list)
    violated_constraints: list[str] = Field(default_factory=list)


class AgentResponse(BaseModel):
    task_summary: str
    extracted_constraints: dict[str, Any]
    missing_info: list[str] = Field(default_factory=list)
    clarifying_questions: list[str] = Field(default_factory=list)
    candidates: list[CandidateProduct] = Field(default_factory=list)
    final_recommendation: CandidateProduct | None = None
    recommendation_reason: list[str] = Field(default_factory=list)

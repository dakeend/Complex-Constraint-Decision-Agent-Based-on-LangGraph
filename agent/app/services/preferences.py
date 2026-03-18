"""用户偏好召回与 Query 重写"""
import json
from pathlib import Path

PREFERENCES_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "user_preferences.json"


def _load_preferences() -> dict:
    if PREFERENCES_PATH.exists():
        try:
            with PREFERENCES_PATH.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def recall_preferences(user_id: str | None) -> dict:
    """召回用户历史偏好"""
    prefs = _load_preferences()
    if not user_id:
        return {}
    return prefs.get(user_id, {})


def rewrite_query(original_query: str, preferences: dict) -> str:
    """根据偏好重写 Query，生成更适合检索的结构化查询"""
    parts = [original_query.strip()]
    if preferences.get("budget_max"):
        parts.append(f"预算{preferences['budget_max']}以内")
    if preferences.get("price_range"):
        parts.append(preferences["price_range"])
    if preferences.get("brand_preference"):
        for b in preferences["brand_preference"][:3]:
            parts.append(b)
    if preferences.get("brand_avoid"):
        for b in preferences["brand_avoid"][:2]:
            parts.append(f"排除{b}")
    if preferences.get("usage_scenarios"):
        parts.append(" ".join(preferences["usage_scenarios"][:2]))
    return " ".join(parts)

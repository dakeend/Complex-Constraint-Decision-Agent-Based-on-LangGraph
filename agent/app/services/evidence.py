"""证据抽取与候选商品聚合"""
import re


def extract_evidence(content_results: list[dict], review_results: list[dict]) -> list[dict]:
    """
    从多平台搜索结果中抽取商品、优缺点、适用场景、风险点
    返回: [{"product_name": str, "pros": list, "cons": list, "scenarios": list, "risks": list, "sources": list}]
    """
    by_product: dict[str, dict] = {}

    for r in content_results:
        for name in r.get("products_mentioned", []):
            norm = _normalize_name(name)
            if norm not in by_product:
                by_product[norm] = {
                    "product_name": norm,
                    "original_names": [name],
                    "pros": [],
                    "cons": [],
                    "scenarios": [],
                    "risks": [],
                    "sources": [],
                }
            else:
                if name not in by_product[norm]["original_names"]:
                    by_product[norm]["original_names"].append(name)
            by_product[norm]["sources"].append(r.get("platform", "unknown"))

    for r in review_results:
        title = r.get("goods_name") or r.get("name") or r.get("title", "")
        if title:
            norm = _normalize_name(title)
            if norm not in by_product:
                by_product[norm] = {
                    "product_name": norm,
                    "original_names": [title],
                    "pros": [],
                    "cons": [],
                    "scenarios": [],
                    "risks": [],
                    "sources": ["pdd_review"],
                    "goods_id": r.get("goods_id"),
                    "goods_sign": r.get("goods_sign"),
                    "price": r.get("price"),
                }

    evidence_list = list(by_product.values())
    for e in evidence_list:
        e["evidence_count"] = len(e["sources"])
    return evidence_list


def _normalize_name(name: str) -> str:
    """商品名标准化：去版本号、规格、品牌括号等"""
    if not name:
        return ""
    s = name.strip()
    s = re.sub(r"\[[^\]]*\]", "", s)
    s = re.sub(r"\s*20\d{2}\s*", " ", s)
    s = re.sub(r"\s*Pro\s*\d*\s*", " Pro ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:60] if len(s) > 60 else s


def deduplicate_candidates(candidates: list[dict]) -> list[dict]:
    """同款对齐与去重"""
    seen: set[str] = set()
    result = []
    for c in candidates:
        norm = _normalize_name(c.get("product_name", c.get("name", "")))
        if not norm or norm in seen:
            continue
        seen.add(norm)
        result.append({**c, "normalized_name": norm})
    return result

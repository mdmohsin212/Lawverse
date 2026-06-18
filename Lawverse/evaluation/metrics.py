from __future__ import annotations
import math
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Sequence
from langchain_core.documents import Document

_WORD_RE = re.compile(r"[\w\u0980-\u09FF]+", re.UNICODE)

def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def tokenize(text: str) -> List[str]:
    return _WORD_RE.findall(normalize_text(text))


def contains_phrase(text: str, phrase: str) -> bool:
    return normalize_text(phrase) in normalize_text(text)


def keyword_coverage(text: str, keywords: Sequence[str]) -> float:
    if not keywords:
        return 0.0
    hits = sum(1 for kw in keywords if contains_phrase(text, kw) or all(t in tokenize(text) for t in tokenize(kw)))
    return hits / len(keywords)


def source_match(source_text: str, expected_source: str) -> bool:
    if not expected_source or expected_source == "none":
        return False
    source_text = normalize_text(source_text).replace("_", "-")
    expected = normalize_text(expected_source).replace("_", "-")
    parts = [p for p in re.split(r"[-_\s.]+", expected) if len(p) > 2]
    return any(part in source_text for part in parts[:4])


def doc_text(doc: Document) -> str:
    metadata = doc.metadata or {}
    meta_text = " ".join(str(v) for v in metadata.values())
    return f"{meta_text}\n{doc.page_content or ''}"


def doc_relevance_score(doc: Document, case: Dict[str, Any]) -> float:
    text = doc_text(doc)
    keyword_score = keyword_coverage(text, case.get("expected_keywords", []))
    section_score = keyword_coverage(text, case.get("expected_sections", []))
    source_score = 1.0 if source_match(str((doc.metadata or {}).get("source", "")), case.get("source_file", "")) else 0.0
    return max(keyword_score, section_score * 0.85, source_score * 0.65)


def binary_relevance(doc: Document, case: Dict[str, Any], threshold: float = 0.25) -> int:
    return 1 if doc_relevance_score(doc, case) >= threshold else 0


def precision_at_k(relevance: Sequence[int], k: int) -> float:
    if k <= 0:
        return 0.0
    top = list(relevance[:k])
    if not top:
        return 0.0
    return sum(top) / k


def hit_rate_at_k(relevance: Sequence[int], k: int) -> float:
    return 1.0 if any(relevance[:k]) else 0.0


def recall_at_k(relevance: Sequence[int], total_relevant: int, k: int) -> float:
    if total_relevant <= 0:
        return 0.0
    return min(sum(relevance[:k]) / total_relevant, 1.0)


def mrr_at_k(relevance: Sequence[int], k: int) -> float:
    for idx, rel in enumerate(relevance[:k], start=1):
        if rel:
            return 1.0 / idx
    return 0.0


def dcg_at_k(relevance: Sequence[float], k: int) -> float:
    score = 0.0
    for idx, rel in enumerate(relevance[:k], start=1):
        score += float(rel) / math.log2(idx + 1)
    return score


def ndcg_at_k(relevance: Sequence[float], k: int) -> float:
    actual = dcg_at_k(relevance, k)
    ideal = dcg_at_k(sorted(relevance, reverse=True), k)
    if ideal == 0:
        return 0.0
    return actual / ideal


def average_precision_at_k(relevance: Sequence[int], k: int) -> float:
    rel_seen = 0
    total = 0.0
    for idx, rel in enumerate(relevance[:k], start=1):
        if rel:
            rel_seen += 1
            total += rel_seen / idx
    if rel_seen == 0:
        return 0.0
    return total / rel_seen


def answer_keyword_score(answer: str, case: Dict[str, Any]) -> float:
    must_include = case.get("answer_must_include") or case.get("expected_keywords", [])
    return keyword_coverage(answer, must_include)


def forbidden_content_score(answer: str, forbidden: Sequence[str]) -> float:
    if not forbidden:
        return 1.0
    violations = sum(1 for phrase in forbidden if contains_phrase(answer, phrase))
    return 1.0 - (violations / len(forbidden))


def has_sources_section(answer: str) -> bool:
    answer = normalize_text(answer)
    return "sources" in answer or "source" in answer


def has_disclaimer(answer: str) -> bool:
    answer = normalize_text(answer)
    return "not a substitute" in answer or "licensed lawyer" in answer or "legal disclaimer" in answer


def aggregate_numeric(rows: List[Dict[str, Any]], keys: Iterable[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key in keys:
        values = [float(row[key]) for row in rows if key in row and isinstance(row[key], (int, float))]
        out[key] = round(sum(values) / len(values), 4) if values else 0.0
    return out


def domain_breakdown(rows: List[Dict[str, Any]], metric_keys: Iterable[str]) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row.get("domain", "unknown"), []).append(row)
    return {domain: aggregate_numeric(items, metric_keys) for domain, items in grouped.items()}


def confusion_counts(rows: List[Dict[str, Any]], expected_key: str, predicted_key: str) -> Dict[str, int]:
    counter = Counter()
    for row in rows:
        counter[f"{row.get(expected_key)} -> {row.get(predicted_key)}"] += 1
    return dict(counter)
from __future__ import annotations
import argparse
import csv
import time
from pathlib import Path
from typing import Any, Dict, List
from Lawverse.evaluation.metrics import (
    aggregate_numeric,
    binary_relevance,
    doc_relevance_score,
    domain_breakdown,
    hit_rate_at_k,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    keyword_coverage,
)
from Lawverse.evaluation.testset import load_eval_dataset, legal_cases, save_json
from Lawverse.logger import logging
from Lawverse.pipeline.rag_pipeline import rag_components
from Lawverse.agents.tools import retrieve_with_hybrid_tool
OUTPUT_DIR = Path("artifacts/evaluation")


def _set_retriever_k(retriever, max_k: int) -> None:
    try:
        retriever.final_top_k = max_k
    except Exception:
        pass
    try:
        retriever.initial_top_k = max(max_k * 4, 25)
    except Exception:
        pass


def evaluate_retrieval(k_values: List[int] | None = None, max_cases: int | None = None) -> Dict[str, Any]:
    k_values = k_values or [1, 3, 5, 10]
    max_k = max(k_values)
    cases = legal_cases(load_eval_dataset())
    if max_cases:
        cases = cases[:max_cases]

    components = rag_components()
    retriever = components["retriever"]
    _set_retriever_k(retriever, max_k)

    rows: List[Dict[str, Any]] = []
    start = time.perf_counter()

    for case in cases:
        query_start = time.perf_counter()
        docs = retrieve_with_hybrid_tool(retriever, case["question"], top_k=max_k)
        latency_ms = round((time.perf_counter() - query_start) * 1000, 2)

        relevance_scores = [doc_relevance_score(doc, case) for doc in docs]
        relevance_binary = [binary_relevance(doc, case) for doc in docs]
        context_text = "\n".join(doc.page_content or "" for doc in docs)
        keyword_recall = keyword_coverage(context_text, case.get("expected_keywords", []))

        row = {
            "id": case["id"],
            "domain": case["domain"],
            "question": case["question"],
            "num_docs": len(docs),
            "latency_ms": latency_ms,
            "keyword_recall": round(keyword_recall, 4),
            "top_source": str((docs[0].metadata or {}).get("source", "")) if docs else "",
            "top_score": float((docs[0].metadata or {}).get("score", 0.0)) if docs else 0.0,
        }

        for k in k_values:
            row[f"hit@{k}"] = hit_rate_at_k(relevance_binary, k)
            row[f"precision@{k}"] = round(precision_at_k(relevance_binary, k), 4)
            row[f"mrr@{k}"] = round(mrr_at_k(relevance_binary, k), 4)
            row[f"ndcg@{k}"] = round(ndcg_at_k(relevance_scores, k), 4)

        rows.append(row)
        logging.info(f"Retrieval eval case {case['id']} completed: hit@{max_k}={row[f'hit@{max_k}']}")

    metric_keys = ["keyword_recall", "latency_ms"]
    for k in k_values:
        metric_keys.extend([f"hit@{k}", f"precision@{k}", f"mrr@{k}", f"ndcg@{k}"])

    summary = aggregate_numeric(rows, metric_keys)
    report = {
        "evaluation_type": "retrieval",
        "dataset": "Lawverse legal QA dataset from Digital Security Act, Labour Act, and Companies Act PDFs",
        "num_cases": len(rows),
        "k_values": k_values,
        "metrics": summary,
        "domain_breakdown": domain_breakdown(rows, metric_keys),
        "notes": [
            "Exact gold chunk IDs are unavailable, so retrieval relevance is estimated using expected source, section, and keyword coverage.",
            "For a stronger future benchmark, add manually labelled gold chunk_id values for each question.",
        ],
        "generated_at_unix": time.time(),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_json(OUTPUT_DIR / "retrieval_metrics.json", report)

    csv_path = OUTPUT_DIR / "retrieval_cases.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["id"])
        writer.writeheader()
        writer.writerows(rows)
        
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Lawverse retrieval evaluation.")
    parser.add_argument("--max-cases", type=int, default=None, help="Run only first N legal cases.")
    parser.add_argument("--k", type=str, default="1,3,5,10", help="Comma-separated K values.")
    args = parser.parse_args()
    k_values = [int(x.strip()) for x in args.k.split(",") if x.strip()]
    report = evaluate_retrieval(k_values=k_values, max_cases=args.max_cases)
    print(f"Saved retrieval metrics to {OUTPUT_DIR / 'retrieval_metrics.json'}")
    print(report["metrics"])


if __name__ == "__main__":
    main()
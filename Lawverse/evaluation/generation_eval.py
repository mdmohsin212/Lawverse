from __future__ import annotations
import argparse
import csv
import time
from pathlib import Path
from typing import Any, Dict, List
from Lawverse.agents.graph import create_agentic_chain
from Lawverse.evaluation.metrics import (
    aggregate_numeric,
    answer_keyword_score,
    forbidden_content_score,
    has_disclaimer,
    has_sources_section,
    keyword_coverage,
    domain_breakdown,
)
from Lawverse.evaluation.testset import legal_cases, load_eval_dataset, save_json
from Lawverse.logger import logging
from Lawverse.pipeline.llm_loader import llm
from Lawverse.pipeline.rag_pipeline import rag_components
OUTPUT_DIR = Path("artifacts/evaluation")


def evaluate_generation(max_cases: int | None = None, dry_run: bool = False) -> Dict[str, Any]:
    cases = legal_cases(load_eval_dataset())
    if max_cases:
        cases = cases[:max_cases]

    chain = None
    if not dry_run:
        components = rag_components()
        chain = create_agentic_chain(components, llm)

    rows: List[Dict[str, Any]] = []
    for case in cases:
        start = time.perf_counter()
        status = "live"
        state: Dict[str, Any] = {}
        answer = ""
        error = ""

        try:
            if dry_run:
                status = "dry_run_reference_answer"
                answer = case["reference_answer"]
            else:
                state = chain.invoke_state({"input": case["question"], "chat_history": []})
                answer = state.get("final_answer") or state.get("draft_answer") or ""
        except Exception as e:
            status = "failed"
            error = str(e)
            logging.error(f"Generation evaluation failed for {case['id']}: {e}")

        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        reference_overlap = keyword_coverage(answer, case.get("expected_keywords", []))
        row = {
            "id": case["id"],
            "domain": case["domain"],
            "question": case["question"],
            "status": status,
            "latency_ms": latency_ms,
            "answer_keyword_score": round(answer_keyword_score(answer, case), 4),
            "expected_keyword_coverage": round(reference_overlap, 4),
            "forbidden_content_score": round(forbidden_content_score(answer, case.get("answer_must_not_include", [])), 4),
            "has_sources": 1.0 if has_sources_section(answer) else 0.0,
            "has_disclaimer": 1.0 if has_disclaimer(answer) else 0.0,
            "evidence_score": float(state.get("evidence_score", 0.0) or 0.0),
            "has_enough_evidence": 1.0 if state.get("has_enough_evidence") else 0.0,
            "citation_check_passed": 1.0 if state.get("citation_check_passed") else 0.0,
            "answer_preview": answer[:500].replace("\n", " "),
            "error": error,
        }
        rows.append(row)

    metric_keys = [
        "latency_ms",
        "answer_keyword_score",
        "expected_keyword_coverage",
        "forbidden_content_score",
        "has_sources",
        "has_disclaimer",
        "evidence_score",
        "has_enough_evidence",
        "citation_check_passed",
    ]
    report = {
        "evaluation_type": "rag_generation",
        "num_cases": len(rows),
        "dry_run": dry_run,
        "metrics": aggregate_numeric(rows, metric_keys),
        "domain_breakdown": domain_breakdown(rows, metric_keys),
        "notes": [
            "This evaluator checks citation/disclaimer/source presence plus keyword-grounding against the curated legal dataset.",
            "Use --dry-run only to verify the evaluator without calling an LLM. Real project metrics should be generated without --dry-run.",
            "Ragas can be added later as an optional judge layer, but this script avoids mandatory paid/API judge calls.",
        ],
        "generated_at_unix": time.time(),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_json(OUTPUT_DIR / "rag_generation_metrics.json", report)

    csv_path = OUTPUT_DIR / "rag_generation_cases.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["id"])
        writer.writeheader()
        writer.writerows(rows)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Lawverse RAG generation evaluation.")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Do not call retriever/LLM; evaluate reference answers only.")
    args = parser.parse_args()
    report = evaluate_generation(max_cases=args.max_cases, dry_run=args.dry_run)
    print(f"Saved generation metrics to {OUTPUT_DIR / 'rag_generation_metrics.json'}")
    print(report["metrics"])


if __name__ == "__main__":
    main()
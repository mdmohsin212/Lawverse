from __future__ import annotations
import csv
import time
from pathlib import Path
from typing import Any, Dict, List
from langchain_core.documents import Document
from Lawverse.agents.nodes import (
    answer_generator_node,
    citation_verifier_node,
    evidence_grader_node,
    intent_classifier_node,
    retrieval_planner_node,
)
from Lawverse.agents.state import AgentState
from Lawverse.evaluation.metrics import aggregate_numeric, confusion_counts
from Lawverse.evaluation.testset import load_eval_dataset, save_json
OUTPUT_DIR = Path("artifacts/evaluation")


class FakeLLM:
    def invoke(self, prompt: str):
        class Response:
            def __init__(self, content: str):
                self.content = content

        lower = prompt.lower()
        if "checking whether retrieved legal context is sufficient" in lower:
            if "section" in lower or "act" in lower or "worker" in lower or "company" in lower:
                return Response("SUFFICIENT: Retrieved context contains legal terms related to the question.")
            return Response("INSUFFICIENT: No relevant legal evidence was found.")
        return Response("### Answer\nFake answer for deterministic evaluation.\n\n### Sources\n- Fake source")


def fake_docs_for_case(case: Dict[str, Any]) -> List[Document]:
    if case.get("source_file") == "none":
        return []
    text = " ".join(case.get("expected_keywords", []))
    return [
        Document(
            page_content=f"Relevant legal context: {text}",
            metadata={
                "source": case.get("source_file"),
                "page_label": case.get("source_page_hint", "unknown"),
                "chunk_id": f"eval_{case.get('id')}",
                "score": 1.0,
            },
        )
    ]


def evaluate_agent_behavior() -> Dict[str, Any]:
    cases = load_eval_dataset()
    llm = FakeLLM()
    rows: List[Dict[str, Any]] = []

    for case in cases:
        state: AgentState = {"input": case["question"], "chat_history": []}
        state = intent_classifier_node(state, llm=None)
        state = retrieval_planner_node(state, llm=None)

        docs = fake_docs_for_case(case)
        state["retrieved_docs"] = docs
        state = evidence_grader_node(state, llm=llm)
        state = answer_generator_node(state, llm=llm)
        state = citation_verifier_node(state, llm=llm)

        expected_intent = case.get("expected_intent")
        expected_plan = case.get("expected_retrieval_plan")
        predicted_intent = state.get("intent")
        predicted_plan = state.get("retrieval_plan")

        row = {
            "id": case["id"],
            "domain": case["domain"],
            "question": case["question"],
            "expected_intent": expected_intent,
            "predicted_intent": predicted_intent,
            "intent_correct": 1.0 if predicted_intent == expected_intent else 0.0,
            "expected_plan": expected_plan,
            "predicted_plan": predicted_plan,
            "retrieval_plan_correct": 1.0 if predicted_plan == expected_plan else 0.0,
            "has_enough_evidence": 1.0 if state.get("has_enough_evidence") else 0.0,
            "citation_check_passed": 1.0 if state.get("citation_check_passed") else 0.0,
            "has_final_answer": 1.0 if bool(state.get("final_answer")) else 0.0,
            "final_answer_preview": (state.get("final_answer") or "")[:300].replace("\n", " "),
        }
        rows.append(row)

    metric_keys = [
        "intent_correct",
        "retrieval_plan_correct",
        "has_enough_evidence",
        "citation_check_passed",
        "has_final_answer",
    ]
    report = {
        "evaluation_type": "agent_behavior",
        "num_cases": len(rows),
        "metrics": aggregate_numeric(rows, metric_keys),
        "intent_confusion": confusion_counts(rows, "expected_intent", "predicted_intent"),
        "plan_confusion": confusion_counts(rows, "expected_plan", "predicted_plan"),
        "notes": [
            "This deterministic evaluation checks agent routing, planner decisions, evidence grading, and citation verifier behavior without calling paid LLM APIs.",
            "The retriever itself is evaluated separately in retrieval_eval.py.",
        ],
        "generated_at_unix": time.time(),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_json(OUTPUT_DIR / "agent_metrics.json", report)

    csv_path = OUTPUT_DIR / "agent_cases.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["id"])
        writer.writeheader()
        writer.writerows(rows)

    return report


def main() -> None:
    report = evaluate_agent_behavior()
    print(f"Saved agent metrics to {OUTPUT_DIR / 'agent_metrics.json'}")
    print(report["metrics"])


if __name__ == "__main__":
    main()
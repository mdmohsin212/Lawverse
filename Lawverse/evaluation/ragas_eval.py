from __future__ import annotations
from Lawverse.evaluation.generation_eval import evaluate_generation

def run_ragas_evaluation(eval_data=None, llm=None):
    report = evaluate_generation(dry_run=False)
    return report.get("metrics", {})

if __name__ == "__main__":
    evaluate_generation(dry_run=False)
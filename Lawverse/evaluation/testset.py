from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List
DATASET_PATH = Path(__file__).resolve().parent / "datasets" / "eval_dataset.jsonl"
REQUIRED_FIELDS = {
    "id",
    "domain",
    "doc_title",
    "source_file",
    "question",
    "reference_answer",
    "expected_keywords",
    "expected_sections",
    "expected_intent",
    "expected_retrieval_plan",
    "answer_must_include",
    "answer_must_not_include",
}

def load_eval_dataset(path: str | Path = DATASET_PATH) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {path}")

    cases: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {e}") from e
            validate_case(item, line_no=line_no)
            cases.append(item)
    return cases


def validate_case(case: Dict[str, Any], line_no: int | None = None) -> None:
    missing = REQUIRED_FIELDS - set(case)
    if missing:
        prefix = f"line {line_no}: " if line_no else ""
        raise ValueError(f"{prefix}missing required fields: {sorted(missing)}")

    for key in ["expected_keywords", "expected_sections", "answer_must_include", "answer_must_not_include"]:
        if not isinstance(case.get(key), list):
            prefix = f"line {line_no}: " if line_no else ""
            raise ValueError(f"{prefix}{key} must be a list")


def save_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def legal_cases(cases: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [case for case in cases if case.get("source_file") != "none"]


def behavior_cases(cases: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [case for case in cases if case.get("source_file") == "none"]


if __name__ == "__main__":
    loaded = load_eval_dataset()
    print(f"Loaded {len(loaded)} Lawverse evaluation cases from {DATASET_PATH}")
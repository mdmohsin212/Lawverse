from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict

OUTPUT_DIR = Path("artifacts/evaluation")
REPORT_FILES = {
    "retrieval": OUTPUT_DIR / "retrieval_metrics.json",
    "rag_generation": OUTPUT_DIR / "rag_generation_metrics.json",
    "agent": OUTPUT_DIR / "agent_metrics.json",
    "safety": OUTPUT_DIR / "safety_metrics.json",
}

def load_report(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "path": str(path)}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_summary() -> Dict[str, Any]:
    return {name: load_report(path) for name, path in REPORT_FILES.items()}


def main() -> None:
    summary = build_summary()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "evaluation_summary.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved evaluation summary to {output_path}")


if __name__ == "__main__":
    main()
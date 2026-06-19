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
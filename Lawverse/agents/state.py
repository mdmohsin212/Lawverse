from __future__ import annotations
from typing import Any, Dict, List, Optional, TypedDict
from langchain_core.documents import Document


class AgentState(TypedDict, total=False):
    input: str
    chat_history: List[Any]
    
    intent: str
    intent_reason: str
    standalone_query: str
    retrieval_plan: str
    
    retrieved_docs: List[Document]
    evidence_score: float
    has_enough_evidence: bool
    evidence_reason: str
    
    draft_answer: str
    final_answer: str
    sources: List[Dict[str, Any]]
    citation_check_passed: bool
    citation_issues: List[str]

    error: Optional[str]
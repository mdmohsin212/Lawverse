from __future__ import annotations
from typing import Any, List
from langchain_core.documents import Document
from Lawverse.agents.state import AgentState
from Lawverse.agents.prompts import QUERY_REWRITE_PROMPT, EVIDENCE_GRADER_PROMPT, ANSWER_GENERATION_PROMPT
from Lawverse.agents.tools import (
    build_source,
    format_docs_for_prompt,
    lexical_evidence_score,
    retrieve_with_hybrid_tool
)
from Lawverse.guardrails.answer_policy import (
    CLOSING_RESPONSE,
    GREETING_RESPONSE,
    INSUFFICIENT_EVIDENCE_RESPONSE,
    NON_LEGAL_RESPONSE,
    classify_simple_intent   
)
from Lawverse.guardrails.legal_disclaimer import append_legal_disclaimer
from Lawverse.logger import logging


def _content_from_llm_response(response: Any) -> str:
    if response is None:
        return ""
    if hasattr(response, "content"):
        return str(response.content)
    return str(response)


def _history_to_text(chat_history: List[Any], max_items: int = 6) -> str:
    if not chat_history:
        return "No previous chat history."
    items = []
    for msg in chat_history[-max_items:]:
        role = msg.__class__.__name__.replace("Message", "")
        content = getattr(msg, "content", str(msg))
        items.append(f"{role}: {content}")
    return "\n".join(items)


def intent_classifier_node(state: AgentState, llm=None) -> AgentState:
    user_input = state.get("input", "")
    intent, reason = classify_simple_intent(user_input)
    state["intent"] = intent
    state["intent_reason"] = reason
    logging.info(f"Agent intent classified as {intent}: {reason}")
    return state



def query_rewriter_node(state: AgentState, llm=None) -> AgentState:
    question = state.get("input", "")
    if state.get("intent") != "legal_question":
        state["standalone_query"] = question
        return state

    try:
        prompt = QUERY_REWRITE_PROMPT.format(
            chat_history=_history_to_text(state.get("chat_history", [])),
            question=question,
        )
        rewritten = _content_from_llm_response(llm.invoke(prompt)).strip() if llm else question
        state["standalone_query"] = rewritten or question
    except Exception as e:
        logging.warning(f"Query rewrite failed; falling back to original query. Error: {e}")
        state["standalone_query"] = question

    return state


def retrieval_planner_node(state: AgentState, llm=None) -> AgentState:
    if state.get("intent") != "legal_question":
        state["retrieval_plan"] = "no_retrieval"
    else:
        state["retrieval_plan"] = "hybrid_dense_sparse_rerank"
    return state


def hybrid_retriever_node(state: AgentState, retriever=None) -> AgentState:
    if state.get("retrieval_plan") == "no_retrieval":
        state["retrieved_docs"] = []
        return state

    query = state.get("standalone_query") or state.get("input", "")
    try:
        docs = retrieve_with_hybrid_tool(retriever, query, top_k=5)
        state["retrieved_docs"] = docs
        state["sources"] = build_source(docs)
    except Exception as e:
        logging.error(f"Agent retrieval failed: {e}")
        state["retrieved_docs"] = []
        state["sources"] = []
        state["error"] = str(e)
    return state



def evidence_grader_node(state: AgentState, llm=None) -> AgentState:
    docs: List[Document] = state.get("retrieved_docs", []) or []
    question = state.get("standalone_query") or state.get("input", "")

    if state.get("intent") != "legal_question":
        state["has_enough_evidence"] = False
        state["evidence_score"] = 0.0
        state["evidence_reason"] = "No legal retrieval required."
        return state

    score = lexical_evidence_score(question, docs)
    state["evidence_score"] = score

    if not docs:
        state["has_enough_evidence"] = False
        state["evidence_reason"] = "No retrieved documents."
        return state

    try:
        context = format_docs_for_prompt(docs, max_chars=5000)
        prompt = EVIDENCE_GRADER_PROMPT.format(question=question, context=context)
        grade = _content_from_llm_response(llm.invoke(prompt)).strip() if llm else ""
        lower = grade.lower()
        if lower.startswith("sufficient"):
            state["has_enough_evidence"] = True
            state["evidence_reason"] = grade
            return state
        if lower.startswith("insufficient"):
            state["has_enough_evidence"] = score >= 0.45
            state["evidence_reason"] = grade
            return state
    except Exception as e:
        logging.warning(f"LLM evidence grading failed; using lexical score. Error: {e}")

    state["has_enough_evidence"] = score >= 0.25
    state["evidence_reason"] = f"Lexical evidence score={score}."
    return state


def answer_generator_node(state: AgentState, llm=None) -> AgentState:
    intent = state.get("intent")

    if intent == "greeting":
        state["draft_answer"] = GREETING_RESPONSE
        return state
    if intent == "closing":
        state["draft_answer"] = CLOSING_RESPONSE
        return state
    if intent in {"non_legal", "empty"}:
        state["draft_answer"] = NON_LEGAL_RESPONSE
        return state
    if not state.get("has_enough_evidence"):
        state["draft_answer"] = INSUFFICIENT_EVIDENCE_RESPONSE
        return state

    docs = state.get("retrieved_docs", []) or []
    context = format_docs_for_prompt(docs)
    question = state.get("input", "")

    try:
        prompt = ANSWER_GENERATION_PROMPT.format(question=question, context=context)
        answer = _content_from_llm_response(llm.invoke(prompt)).strip() if llm else ""
        state["draft_answer"] = answer or INSUFFICIENT_EVIDENCE_RESPONSE
    except Exception as e:
        logging.error(f"Answer generation failed: {e}")
        state["draft_answer"] = INSUFFICIENT_EVIDENCE_RESPONSE
        state["error"] = str(e)

    return state


def citation_verifier_node(state: AgentState, llm=None) -> AgentState:
    answer = state.get("draft_answer", "") or ""
    docs = state.get("retrieved_docs", []) or []
    issues = []

    if state.get("intent") == "legal_question" and state.get("has_enough_evidence"):
        if "### Sources" not in answer:
            issues.append("Answer did not include a Sources section; sources were appended automatically.")
            sources = build_source(docs)
            source_lines = []
            for src in sources:
                source_lines.append(
                    f"- Source {src['rank']}: {src['source']} | Page: {src['page']} | "
                    f"Chunk: {src['chunk_id']} | Score: {src['score']}"
                )
            answer = f"{answer}\n\n### Sources\n" + "\n".join(source_lines)

    state["citation_issues"] = issues
    state["citation_check_passed"] = len(issues) == 0
    state["final_answer"] = append_legal_disclaimer(answer)
    return state
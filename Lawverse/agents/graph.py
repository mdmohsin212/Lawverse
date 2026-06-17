from __future__ import annotations
from typing import Any, Dict, Iterable, Optional
from langgraph.graph import END, StateGraph

from Lawverse.agents.state import AgentState
from Lawverse.agents.nodes import (
    answer_generator_node,
    citation_verifier_node,
    evidence_grader_node,
    hybrid_retriever_node,
    intent_classifier_node,
    query_rewriter_node,
    retrieval_planner_node,
)
from Lawverse.logger import logging


class AgenticLawverseChain:
    def __init__(self, retriever, llm, use_langgraph: bool = True):
        self.retriever = retriever
        self.llm = llm
        self.use_langgraph = use_langgraph
        self._compiled_graph = None
        if use_langgraph:
            self._compiled_graph = self._try_build_langgraph()

    def _try_build_langgraph(self):
        try:
            graph = StateGraph(AgentState)
            graph.add_node("intent_classifier", lambda s: intent_classifier_node(s, self.llm))
            graph.add_node("query_rewriter", lambda s: query_rewriter_node(s, self.llm))
            graph.add_node("retrieval_planner", lambda s: retrieval_planner_node(s, self.llm))
            graph.add_node("hybrid_retriever", lambda s: hybrid_retriever_node(s, self.retriever))
            graph.add_node("evidence_grader", lambda s: evidence_grader_node(s, self.llm))
            graph.add_node("answer_generator", lambda s: answer_generator_node(s, self.llm))
            graph.add_node("citation_verifier", lambda s: citation_verifier_node(s, self.llm))

            graph.set_entry_point("intent_classifier")
            graph.add_edge("intent_classifier", "query_rewriter")
            graph.add_edge("query_rewriter", "retrieval_planner")
            graph.add_edge("retrieval_planner", "hybrid_retriever")
            graph.add_edge("hybrid_retriever", "evidence_grader")
            graph.add_edge("evidence_grader", "answer_generator")
            graph.add_edge("answer_generator", "citation_verifier")
            graph.add_edge("citation_verifier", END)

            compiled = graph.compile()
            logging.info("LangGraph agent workflow compiled successfully.")
            return compiled
        
        except Exception as e:
            logging.warning(f"LangGraph is unavailable or failed to compile; using fallback sequential graph. Error: {e}")
            return None

    def _run_fallback_graph(self, state: AgentState) -> AgentState:
        state = intent_classifier_node(state, self.llm)
        state = query_rewriter_node(state, self.llm)
        state = retrieval_planner_node(state, self.llm)
        state = hybrid_retriever_node(state, self.retriever)
        state = evidence_grader_node(state, self.llm)
        state = answer_generator_node(state, self.llm)
        state = citation_verifier_node(state, self.llm)
        return state

    def invoke(self, inputs: Dict[str, Any], config: Optional[dict] = None) -> str:
        state: AgentState = {
            "input": inputs.get("input", ""),
            "chat_history": inputs.get("chat_history", []),
        }

        if self._compiled_graph is not None:
            output_state = self._compiled_graph.invoke(state, config=config)
        else:
            output_state = self._run_fallback_graph(state)

        return output_state.get("final_answer") or output_state.get("draft_answer") or ""

    def stream(self, inputs: Dict[str, Any], config: Optional[dict] = None) -> Iterable[str]:
        answer = self.invoke(inputs, config=config)
        chunk_size = 80
        for i in range(0, len(answer), chunk_size):
            yield answer[i:i + chunk_size]


def create_agentic_chain(components, llm, use_langgraph: bool = True) -> AgenticLawverseChain:
    retriever = components["retriever"]
    return AgenticLawverseChain(retriever=retriever, llm=llm, use_langgraph=use_langgraph)
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List
from langchain_core.documents import Document


def retrieve_with_hybrid_tool(retriever, query: str, top_k: int = 5) -> List[Document]:
    if retriever is None:
        return []
    try:
        docs = retriever.invoke(query)
    except Exception:
        try:
            docs = retriever.get_relevant_documents(query)
        except Exception:
            docs = retriever._get_relevant_documents(query)

    return list(docs or [])[:top_k]


def clean_source_name(source: str) -> str:
    if not source:
        return "Unknown document"

    source = str(source)
    name = Path(source).name
    return name or source


def document_to_source(doc: Document, rank: int | None = None) -> Dict[str, Any]:
    metadata = dict(doc.metadata or {})
    raw_source = metadata.get("source", "unknown")

    return {
        "rank": rank or metadata.get("rank"),
        "source": clean_source_name(raw_source),
        "raw_source": raw_source,
        "page": metadata.get("page_label", metadata.get("page", "unknown")),
        "chunk_id": metadata.get("chunk_id", "unknown"),
        "chunk_uid": metadata.get("chunk_uid", "unknown"),
        "score": metadata.get("score", metadata.get("rrf_score", "unknown")),
        "retriever": metadata.get("retriever", "hybrid"),
        "preview": (doc.page_content or "")[:350].replace("\n", " "),
    }


def format_docs_for_prompt(docs: List[Document], max_chars: int = 7000) -> str:
    blocks = []
    total = 0

    for idx, doc in enumerate(docs or [], 1):
        source = document_to_source(doc, rank=idx)
        text = doc.page_content or ""
        block = (
            f"[{idx}] Document: {source['source']} | Page: {source['page']} | "
            f"Chunk: {source['chunk_id']} | Score: {source['score']}\n"
            f"{text}"
        )

        if total + len(block) > max_chars:
            break
        blocks.append(block)
        total += len(block)
    return "\n\n---\n\n".join(blocks)


def build_sources(docs: List[Document]) -> List[Dict[str, Any]]:
    return [document_to_source(doc, rank=i) for i, doc in enumerate(docs or [], 1)]


def lexical_evidence_score(question: str, docs: List[Document]) -> float:
    if not question or not docs:
        return 0.0

    stop = {
        "the", "a", "an", "and", "or", "to", "of", "in", "for", "is", "are", "am", "i",
        "what", "how", "why", "when", "can", "could", "should", "me", "my", "about",
        "tell", "explain",
    }
    q_words = {w.strip(".,?!;:()[]{}'\"").lower() for w in question.split()}
    q_words = {w for w in q_words if len(w) > 2 and w not in stop}

    if not q_words:
        return 0.0

    context = " ".join((doc.page_content or "") for doc in docs).lower()
    hits = sum(1 for w in q_words if w in context)
    return round(hits / max(len(q_words), 1), 4)
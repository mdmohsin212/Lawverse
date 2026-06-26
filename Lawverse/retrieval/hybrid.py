from __future__ import annotations
from collections import defaultdict
from typing import Dict
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field, PrivateAttr
from sentence_transformers import CrossEncoder
from Lawverse.retrieval.sparse import bm25_retrieve
from Lawverse.logger import logging
from Lawverse.exception import ExceptionHandle
import sys


def _doc_key(doc: Document) -> str:
    metadata = doc.metadata or {}
    chunk_id = metadata.get("chunk_id")
    if chunk_id is not None:
        return str(chunk_id)
    return str(metadata.get("source") or doc.page_content[:120])


def _clone_with_metadata(doc: Document, extra_metadata: dict) -> Document:
    metadata = dict(doc.metadata or {})
    metadata.update(extra_metadata)
    return Document(page_content=doc.page_content, metadata=metadata)


def hybrid_retrieve(
    query,
    faiss_db,
    bm25,
    chunks,
    cross_encoder=None,
    initial_top_k=25,
    final_top_k=5,
    rrf_k=60,
):
    try:
        dense_results = faiss_db.similarity_search(query, k=initial_top_k)
        sparse_results = bm25_retrieve(bm25, query, chunks, top_k=initial_top_k)

        doc_map: Dict[str, Document] = {}
        rrf_scores = defaultdict(float)
        retrieval_meta = defaultdict(dict)

        for rank, doc in enumerate(dense_results, 1):
            key = _doc_key(doc)
            doc_map[key] = doc
            rrf_scores[key] += 1.0 / (rrf_k + rank)
            retrieval_meta[key].update({"dense_rank": rank, "retriever": "hybrid"})

        for rank, (doc, score) in enumerate(sparse_results, 1):
            key = _doc_key(doc)
            doc_map[key] = doc
            rrf_scores[key] += 1.0 / (rrf_k + rank)
            retrieval_meta[key].update({"sparse_rank": rank, "bm25_score": float(score), "retriever": "hybrid"})

        fused_keys = [key for key, _ in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)]
        if not fused_keys:
            logging.info("No documents found after hybrid fusion.")
            return []

        candidate_keys = fused_keys[:initial_top_k]
        candidate_docs = [doc_map[key] for key in candidate_keys]

        if cross_encoder is not None and candidate_docs:
            sentence_pairs = [[query, doc.page_content] for doc in candidate_docs]
            cross_encoder_scores = cross_encoder.predict(sentence_pairs, show_progress_bar=False)
            ranked = sorted(
                zip(candidate_keys, candidate_docs, cross_encoder_scores),
                key=lambda x: x[2],
                reverse=True,
            )
        else:
            ranked = [(key, doc_map[key], rrf_scores[key]) for key in candidate_keys]

        final_docs = []
        for rank, (key, doc, score) in enumerate(ranked[:final_top_k], 1):
            metadata = {
                "rank": rank,
                "score": float(score),
                "rrf_score": float(rrf_scores[key]),
                **retrieval_meta[key],
            }
            final_docs.append(_clone_with_metadata(doc, metadata))

        logging.info("Hybrid retrieval and re-ranking successfully completed.")
        return final_docs

    except Exception as e:
        logging.error(f"Hybrid retrieval failed: {e}")
        raise ExceptionHandle(e, sys)


class HybridRetriever(BaseRetriever):
    faiss_db: object = Field(...)
    bm25: object = Field(...)
    chunks: list = Field(...)
    initial_top_k: int = Field(default=25)
    final_top_k: int = Field(default=4)
    _cross_encoder: CrossEncoder = PrivateAttr(default=None)

    def init_cross_encoder(self):
        try:
            self._cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            logging.info("Cross-encoder reranker loaded successfully.")
        except Exception as e:
            self._cross_encoder = None
            logging.warning(f"Cross-encoder could not be loaded. Falling back to RRF ranking. Error: {e}")

    def _get_relevant_documents(self, query, *, run_manager=None):
        return hybrid_retrieve(
            query,
            self.faiss_db,
            self.bm25,
            self.chunks,
            self._cross_encoder,
            initial_top_k=self.initial_top_k,
            final_top_k=self.final_top_k,
        )

    async def _aget_relevant_documents(self, query, *, run_manager=None):
        return self._get_relevant_documents(query)
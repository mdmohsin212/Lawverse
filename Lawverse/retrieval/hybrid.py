import numpy as np
from langchain.schema import Document
from langchain.schema.retriever import BaseRetriever
from Lawverse.retrieval.sparse import bm25_retrieve
from Lawverse.logger import logging
from Lawverse.exception import ExceptionHandle
import sys

def hybrid_retrieve(query, faiss_db, bm25, chunks, top_k=3, alpha=0.5):
    try:
        dense_results = faiss_db.similarity_search(query, k=top_k)
        sparse_results = bm25_retrieve(bm25, chunks, query, top_k=top_k)

        dense_pairs = []
        for item in dense_results:
            if isinstance(item, tuple):
                doc = item[0]
                score = item[1] if len(item) > 1 else 0
                dense_pairs.append((doc, score))
            else:
                dense_pairs.append((item, 0))

        sparse_pairs = []
        for item in sparse_results:
            if isinstance(item, tuple):
                text = item[0]
                score = item[1] if len(item) > 1 else 0
                sparse_pairs.append((text, score))
            else:
                sparse_pairs.append((item, 0))

        dense_scores = np.array([s for _, s in dense_pairs])
        dense_scores = (dense_scores - dense_scores.min()) / (dense_scores.max() - dense_scores.min() + 1e-8)
        sparse_scores = np.array([s for _, s in sparse_pairs])
        sparse_scores = (sparse_scores - sparse_scores.min()) / (sparse_scores.max() - sparse_scores.min() + 1e-8)

        combined = {}
        for (doc, _), ds in zip(dense_pairs, dense_scores):
            combined[doc.page_content] = alpha * ds
        for (text, _), ss in zip(sparse_pairs, sparse_scores):
            combined[text] = combined.get(text, 0) + (1 - alpha) * ss
            
        ranked_docs = sorted(combined.items(), key=lambda x: x[1], reverse=True)

        logging.info(f"Hybrid retrieval successfully completed.")
        return [Document(page_content=doc) for doc, _ in ranked_docs[:top_k]]

    except Exception as e:
        logging.error(f"Hybrid retrieval failed")
        raise ExceptionHandle(e, sys)


class HybridRetriever(BaseRetriever):
    def __init__(self, faiss_db, bm25, chunks, top_k=4, alpha=0.5):
        self.faiss_db = faiss_db
        self.bm25 = bm25
        self.chunks = chunks
        self.top_k = top_k
        self.alpha = alpha

    def _get_relevant_documents(self, query, *, run_manager=None):
        return hybrid_retrieve(query, self.faiss_db, self.bm25, self.chunks, top_k=self.top_k, alpha=self.alpha)
    
    async def _aget_relevant_documents(self, query, *, run_manager=None):
        return self._get_relevant_documents(query)
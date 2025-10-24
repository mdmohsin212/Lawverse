import numpy as np
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field, PrivateAttr
from sentence_transformers import CrossEncoder
from collections import defaultdict
from Lawverse.retrieval.sparse import bm25_retrieve
from Lawverse.logger import logging
from Lawverse.exception import ExceptionHandle
import sys

def hybrid_retrieve(query, faiss_db, bm25, chunks, cross_encoder, initial_top_k=25, final_top_k=5, rrf_k=60):
    try:
        dense_results = faiss_db.similarity_search(query, k=initial_top_k)
        sparse_results = bm25_retrieve(bm25, chunks, query, top_k=initial_top_k)
        
        dense_ranked_list = [doc.page_content for doc in dense_results]
        sparse_ranked_list = [text for text, _ in sparse_results]
        
        rrf_scores = defaultdict(float)
        
        for rank, doc_contant in enumerate(dense_ranked_list, 1):
            rrf_scores[doc_contant] += 1.0 / (rrf_k + rank)
            
        for rank, doc_contant in enumerate(sparse_ranked_list, 1):
            rrf_scores[doc_contant] += 1.0 / (rrf_k + rank)
        
        fused_results = sorted(rrf_scores.items(), key=lambda x : x[1], reverse=True)
        
        top_fused_docs_contant = [doc_contant for doc_contant, _ in fused_results[:initial_top_k]]
        
        if not top_fused_docs_contant:
            logging.info("No documents found after fusion.")
            return []

        sentence_pairs = [[query, doc_contant] for doc_contant in top_fused_docs_contant]
        
        cross_encoder_scores = cross_encoder.predict(sentence_pairs, show_progress_bar=False)
        
        reranked_results = sorted(
            zip(top_fused_docs_contant, cross_encoder_scores), 
            key=lambda x: x[1],
            reverse=True
        )
        
        final_docs = []
        for doc_content, score in reranked_results[:final_top_k]:
            final_docs.append(
                Document(
                    page_content=doc_content,
                    metadata={"score": float(score)}
                )
            )

        logging.info(f"Hybrid retrieval and re-ranking successfully completed.")
        return final_docs

    except Exception as e:
        logging.error(f"Hybrid retrieval failed")
        raise ExceptionHandle(e, sys)


class HybridRetriever(BaseRetriever):
    faiss_db: object = Field(...)
    bm25: object = Field(...)
    chunks: list = Field(...)
    inital_top_k: int = Field(default=25)
    final_top_k: int = Field(default=4)
    _cross_encoder : CrossEncoder = PrivateAttr(default=None)
    
    def init_cross_encoder(self):
        self._cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    def _get_relevant_documents(self, query, *, run_manager=None):
        return hybrid_retrieve(
            query, 
            self.faiss_db,
            self.bm25, 
            self.chunks, 
            self._cross_encoder,
            initial_top_k=self.inital_top_k,
            final_top_k=self.final_top_k)
    
    async def _aget_relevant_documents(self, query, *, run_manager=None):
        return self._get_relevant_documents(query)
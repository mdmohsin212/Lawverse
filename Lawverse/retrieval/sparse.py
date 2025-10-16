import numpy as np
from rank_bm25 import BM25Okapi
from Lawverse.logger import logging
from Lawverse.exception import ExceptionHandle
import sys

def build_sparse_index(chunks):
    try:
        tokenized_corpus = [chunk.split() for chunk in chunks]
        bm25 = BM25Okapi(tokenized_corpus)
        
        logging.info(f"BM25 sparse index successfully built with {len(chunks)} chunks.")
        return bm25

    except Exception as e:
        logging.error(f"Failed to build BM25 sparse index. Error: {e}")
        raise ExceptionHandle(e, sys)

def bm25_retrieve(bm25, chunks, query, top_k=3):
    try:
        tokenized_query = query.split()
        scores = bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[-top_k:][::-1]

        logging.info(f"Successfully retrieved top {len(top_indices)} relevant chunks using BM25.")
        return [(chunks[i], scores[i]) for i in top_indices]

    except Exception as e:
        logging.error(f"BM25 retrieval failed")
        raise ExceptionHandle(e, sys)
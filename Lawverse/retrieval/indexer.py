from Lawverse.retrieval.dense import build_dense_index
from Lawverse.retrieval.sparse import build_sparse_index
from Lawverse.logger import logging
from Lawverse.exception import ExceptionHandle
import sys

def build_index(chunks):
    try:
        logging.info("Building dense FAISS index...")
        dense_db = build_dense_index(chunks)
        
        logging.info("Building sparse BM25 index...")
        sparse_db = build_sparse_index(chunks)
        
        logging.info(f"Successfully built dense and sparse indexes.")
        return dense_db, sparse_db
    
    except Exception as e:
        logging.error(f"Failed to build indexes")
        raise ExceptionHandle(e, sys)
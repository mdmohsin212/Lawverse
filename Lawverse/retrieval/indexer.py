from Lawverse.retrieval.dense import build_dense_index
from Lawverse.retrieval.sparse import build_sparse_index
from Lawverse.logger import logging
from Lawverse.exception import ExceptionHandle
import sys
import os

def build_index(chunks, FAISS_PATH):
    try:
        logging.info("Building sparse BM25 index...")
        sparse_db = build_sparse_index(chunks)
        
        logging.info(f"Successfully built dense and sparse indexes.")
        
        if os.path.exists(FAISS_PATH):
            logging.info(f"Dense FAISS index already exists at: {FAISS_PATH}")
            return FAISS_PATH, sparse_db
        
        logging.info("Building New dense FAISS index..."    )
        dense_db = build_dense_index(chunks)
        
        dense_db.save_local(FAISS_PATH)
        logging.info(f"FAISS index saved at {FAISS_PATH}")

        return FAISS_PATH, sparse_db
    
    except Exception as e:
        logging.error(f"Failed to build indexes")
        raise ExceptionHandle(e, sys)
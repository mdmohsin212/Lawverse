from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from Lawverse.logger import logging
from Lawverse.exception import ExceptionHandle
import sys

def build_dense_index(chunks):
    try:
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        
        logging.info("Building FAISS vector store from text chunks...")
        db = FAISS.from_texts(chunks, embeddings)
        
        logging.info(f"FAISS dense index successfully built with {len(chunks)} chunks.")
        return db
    
    except Exception as e:
        logging.error(f"Failed to build FAISS dense index. Error: {e}")
        raise ExceptionHandle(e, sys)

def faiss_retriver(db, query, top_k=2):
    try:
        results = db.similarity_search(query, k=top_k)

        logging.info(f"Successfully retrieved {len(results)} relevant documents using FAISS.")
        return results
    
    except Exception as e:
        logging.error(f"FAISS retrieval failed")
        raise ExceptionHandle(e, sys)
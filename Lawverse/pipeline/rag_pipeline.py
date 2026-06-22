from Lawverse.datapipeline.ingest import fetch_file
from Lawverse.datapipeline.dataset_loader import load_pdf_text
from Lawverse.datapipeline.preprocess import chunk_documents, create_bilingual_chunks
from Lawverse.retrieval.indexer import build_index
from Lawverse.retrieval.hybrid import HybridRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from Lawverse.logger import logging
from Lawverse.exception import ExceptionHandle
from Lawverse.utils.config import FAISS_PATH
import sys
import pickle

def rag_components():
    try:
        pdf_path = fetch_file()
        documents = load_pdf_text(pdf_path)

        chunks = chunk_documents(documents, 1200, 300)
        bilingual_chunks_path = create_bilingual_chunks(chunks)

        logging.info(f"Loading processed chunks from {bilingual_chunks_path}")
        with open(bilingual_chunks_path, "rb") as f:
            bilingual_chunks = pickle.load(f)

        logging.info(f"Successfully loaded {len(bilingual_chunks)} bilingual chunks.")

        dense_db_path, bm25 = build_index(bilingual_chunks, FAISS_PATH)
        logging.info("Dense + BM25 indexes built successfully from aligned bilingual chunks.")

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        dense_db = FAISS.load_local(
            dense_db_path,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        logging.info(
            f"FAISS index loaded successfully from '{dense_db_path}' "
            f"with {dense_db.index.ntotal} vectors."
        )

        retriever = HybridRetriever(
            faiss_db=dense_db,
            bm25=bm25,
            chunks=bilingual_chunks,
            initial_top_k=10,
            final_top_k=5,
        )
        retriever.init_cross_encoder()

        logging.info("Hybrid retriever initialized successfully with aligned dense + sparse chunks.")
        logging.info("RAG retrieval components loaded successfully for agentic workflow.")

        return {
            "retriever": retriever,
            "num_chunks": len(bilingual_chunks),
            "faiss_path": str(dense_db_path),
            "chunk_source": str(bilingual_chunks_path),
        }

    except Exception as e:
        logging.error(f"RAG component loading failed: {e}")
        raise ExceptionHandle(e, sys)
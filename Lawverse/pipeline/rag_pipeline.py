from Lawverse.datapipeline.ingest import fetch_file
from Lawverse.datapipeline.dataset_loader import load_pdf_text
from Lawverse.datapipeline.preprocess import chunk_text, translate_chunks
from Lawverse.retrieval.indexer import build_index
from Lawverse.retrieval.hybrid import HybridRetriever
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from Lawverse.pipeline.llm_loader import llm
from Lawverse.memory.langchain_memory import ChatMemory
from Lawverse.logger import logging
from Lawverse.exception import ExceptionHandle
from Lawverse.utils.config import FAISS_PATH

from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
import sys

def rag_components():
    try:
        pdf_path = fetch_file()
        text = load_pdf_text(pdf_path)
        
        chunks = chunk_text(text, 1200, 250)
        
        chunks_translated = translate_chunks(chunks)
        logging.info("Chunks translated successfully.")
        
        bilingual_chunks = [
                f"EN: {en}\nBN: {bn}"
                for en, bn in zip(chunks, chunks_translated)
            ]
        logging.info(f"Bilingual chunks created: {len(bilingual_chunks)}")
        
        dense_db_path, bm25 = build_index(bilingual_chunks, FAISS_PATH)
        logging.info("Dense + BM25 indexes built successfully.")
        
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        dense_db = FAISS.load_local(dense_db_path, embeddings, allow_dangerous_deserialization=True)
        logging.info(f"FAISS index loaded successfully from '{dense_db_path}' with {dense_db.index.ntotal} vectors.")
        
        retriever = HybridRetriever(faiss_db=dense_db, bm25=bm25, chunks=bilingual_chunks, top_k=2, alpha=0.5)
        logging.info("Hybrid retriever initialized successfully with dense and sparse indexes.")
                
        qa_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            You are a helpful **legal assistant** that explains the **Companies Act, 1994 (Bangladesh)** in clear, natural, and accurate English.

            Use the provided context to find the **most relevant legal definition or explanation**.
            If possible, mention the **exact Section number or heading** where the definition appears.

            When explaining:
            - Use plain English so that an ordinary Bangladeshi reader (not a lawyer) can understand easily.
            - When the question involves definitions, show how it differs from related terms (e.g., Articles vs Memorandum).
            - Keep the answer concise — ideally 3–6 short paragraphs.

            Context:
            {context}

            Question:
            {question}

            Now give a factual and simplified explanation.
            """
            )
        
        logging.info("RAG components loaded successfully.")
        return {
            "retriever": retriever,
            "qa_prompt": qa_prompt
        }

    except Exception as e:
        logging.error(f"RAG pipeline failed: {e}")
        raise ExceptionHandle(e, sys)
    

def create_chat_chian(components, chat_id=None):
    try:
        memory_manager = ChatMemory(chat_id=chat_id)
        memory = memory_manager.memory
        
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=components["retriever"],
            combine_docs_chain_kwargs={"prompt": components["qa_prompt"]},
            memory=memory,
            return_source_documents=True,
            output_key="answer"
        )
        
        logging.info("RAG chain initialized successfully.")
        return chain, memory_manager
    
    except Exception as e:
        logging.error(f"Chat chain creation failed")
        raise ExceptionHandle(e, sys)
from Lawverse.datapipeline.ingest import fetch_file
from Lawverse.datapipeline.dataset_loader import load_pdf_text
from Lawverse.datapipeline.preprocess import chunk_text, translate_chunks
from Lawverse.retrieval.indexer import build_index
from Lawverse.retrieval.hybrid import HybridRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from Lawverse.pipeline.llm_loader import llm
from Lawverse.memory.langchain_memory import ChatMemory
from Lawverse.logger import logging
from Lawverse.exception import ExceptionHandle
from Lawverse.utils.config import FAISS_PATH

from langchain_classic.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
import sys

def rag_components():
    try:
        pdf_path = fetch_file()
        text = load_pdf_text(pdf_path)
        
        chunks = chunk_text(text, 1200, 300)
        
        chunks_translated_path = translate_chunks(chunks)
        logging.info("Chunks translated successfully.")
        
        with open(chunks_translated_path, "r", encoding="utf-8") as f:
            chunks_translated = [line.strip() for line in f if line.strip()]    
        
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
            You are a helpful **legal assistant** specializing in **Bangladeshi law**, capable of communicating in both **English and Bengali**.
            
            You must answer questions based *only* on the provided context, which contains information from three different laws:
            1.  The Companies Act, 1994
            2.  The Bangladesh Labour Act, 2006
            3.  The Digital Security Act, 2018 (or "Digital Security Law")

            **Instructions:**
            1.  **Language Match:** You MUST first detect the language of the 'Question'. Your entire answer MUST be in the **same language** as the question (e.g., if the question is in Bengali, the answer must be in Bengali).
            2.  **Identify the Law:** Read the question and find the part of the context that is most relevant.
            3.  **Cite Accurately:** You MUST cite the **full name of the Act** (e.g., "According to the Bangladesh Labour Act, 2006..." or "বাংলাদেশ শ্রম আইন, ২০০৬ অনুযায়ী...") AND the **exact Section number**.
            4.  **Explain Simply:** Use plain, simple language (English or Bengali) so an ordinary Bangladeshi reader (not a lawyer) can understand.
            5.  **Be Concise:** Keep the answer focused, ideally 3-6 short paragraphs.
            6.  **Handle Ambiguity:** If a question is vague (e.g., "What is a 'director'?"), check if multiple laws define it. Answer using the *most relevant law* first.

            Context:
            {context}

            Question:
            {question}

            Now, detect the question's language, identify the correct law, and provide a factual, simplified explanation in that same language, based *only* on the text above.
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
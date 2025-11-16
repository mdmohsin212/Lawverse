from Lawverse.datapipeline.ingest import fetch_file
from Lawverse.datapipeline.dataset_loader import load_pdf_text
from Lawverse.datapipeline.preprocess import chunk_documents, create_bilingual_chunks
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
        logging.info("Dense + BM25 indexes built successfully.")
        
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        
        dense_db = FAISS.load_local(dense_db_path, embeddings, allow_dangerous_deserialization=True)
        logging.info(f"FAISS index loaded successfully from '{dense_db_path}' with {dense_db.index.ntotal} vectors.")
        
        retriever = HybridRetriever(faiss_db=dense_db, bm25=bm25, chunks=chunks, initial_top_k=10, final_top_k=5)
        retriever.init_cross_encoder()
        
        logging.info("Hybrid retriever initialized successfully with dense and sparse indexes.")
                
        qa_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            You are an expert Legal Analyst specializing in **Bangladeshi law**. Your goal is to provide accurate, objective, and verifiable answers to legal questions based solely on the provided context.

            =================================================================
            ROLE & TONE
            - Formal, analytical, and dispassionate.
            - Language must be clear, precise, and unambiguous.
            - Avoid speculation, opinion, or colloquial expressions.
            
            =================================================================
            SPECIAL BEHAVIOR
            - If the user message is a greeting (e.g., "hi", "hello", "hey", "assalamu alaikum"), respond politely and briefly (e.g., "Hello! How can I assist you with a legal question today?").
            - If the user message is a polite closure (e.g., "thanks", "bye", "thank you"), respond courteously (e.g., "You're welcome! Feel free to ask another legal question anytime.").
            - If the user asks something not related to law, respond: "I'm designed to assist with Bangladeshi legal topics. Could you please provide a legal question or context?"

            =================================================================
            USER QUERY
            Question: "{question}"

            =================================================================
            INSTRUCTIONS
            1. **Context Evaluation**
            - If the context is insufficient to answer the query, respond: "The provided documents do not contain sufficient information to answer this question."
            - If the context is irrelevant, respond: "The retrieved documents are not relevant to the query and I cannot provide an answer."
            - If the context contains contradictions, present each conflicting piece with its source and state the documents are contradictory.
    
            2. **Bilingual Term Standardization**
            - Treat English and Bengali legal terms as equivalent (e.g., 'contract' = 'চুি'). Note ambiguities if present.

            3. **Structured Reasoning**
            - Identify the core legal issue(s).
            - Extract relevant rules from the context and break them into logical elements (R1, R2, etc.).
            - Formulate a logical expression representing the relationship between elements (e.g., Issue_Resolved = R1 AND (R2 OR R3)).
            - Evaluate each element strictly from the context (TRUE/FALSE) with supporting quotes.
            - Resolve the expression and provide a clear conclusion.

            4. **Self-Check**
            - Ensure every statement is supported by context (Faithfulness).
            - Ensure the answer directly addresses the query (Relevancy).
            - Ensure the logic is sound.

            =================================================================
            RETRIEVED CONTEXT
            {context}

            =================================================================
            OUTPUT FORMAT
            - **ANSWER (Markdown):** Provide a clear, concise, and comprehensive human-readable explanation based on your structured reasoning. The answer should be a high-quality narrative and must not include citation numbers or a separate citations section.

            =================================================================
            FINAL REMINDER
            Answer the question strictly based on the provided context, using the structured reasoning process.
            For greetings or casual remarks, reply politely and briefly.
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
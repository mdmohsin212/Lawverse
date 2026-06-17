from Lawverse.datapipeline.ingest import fetch_file
from Lawverse.datapipeline.dataset_loader import load_pdf_text
from Lawverse.datapipeline.preprocess import chunk_documents, create_bilingual_chunks
from Lawverse.retrieval.indexer import build_index
from Lawverse.retrieval.hybrid import HybridRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from Lawverse.pipeline.llm_loader import llm
from Lawverse.memory.langchain_memory import ChatMemory
from Lawverse.logger import logging
from Lawverse.exception import ExceptionHandle
from Lawverse.utils.config import FAISS_PATH
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
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

        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        dense_db = FAISS.load_local(dense_db_path, embeddings, allow_dangerous_deserialization=True)
        logging.info(f"FAISS index loaded successfully from '{dense_db_path}' with {dense_db.index.ntotal} vectors.")

        retriever = HybridRetriever(
            faiss_db=dense_db,
            bm25=bm25,
            chunks=bilingual_chunks,
            initial_top_k=10,
            final_top_k=5,
        )
        retriever.init_cross_encoder()

        logging.info("Hybrid retriever initialized successfully with aligned dense + sparse chunks.")

        qa_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            You are Lawverse, an educational legal document intelligence assistant focused on Bangladeshi legal documents.

            IMPORTANT BOUNDARY
            - You provide legal information from the retrieved documents, not professional legal advice.
            - Answer strictly from the retrieved context.
            - If the context is insufficient, say: "The provided documents do not contain sufficient information to answer this question."
            - Do not invent laws, sections, citations, page numbers, or document names.

            SPECIAL BEHAVIOR
            - If the user message is a greeting, reply politely and briefly.
            - If the user message is a polite closure, reply courteously.
            - If the user asks something unrelated to law, say: "I'm designed to assist with Bangladeshi legal topics. Could you please provide a legal question or context?"

            USER QUESTION
            {question}

            RETRIEVED CONTEXT
            {context}

            OUTPUT FORMAT
            ### Answer
            Provide a clear, concise explanation based only on the retrieved context.

            ### Sources
            List the sources you used. For each source, include document/source name, page if available, chunk id if available, and why it supports the answer.

            ### Legal Disclaimer
            Lawverse is an educational legal information assistant. It is not a substitute for a licensed lawyer.
            """
    )

        document_prompt = PromptTemplate.from_template(
            """[Source: {source} | Page: {page_label} | Chunk: {chunk_id} | Score: {score}]\n{page_content}"""
        )

        logging.info("RAG components loaded successfully.")
        return {
            "retriever": retriever,
            "qa_prompt": qa_prompt,
            "document_prompt": document_prompt,
        }

    except Exception as e:
        logging.error(f"RAG pipeline failed: {e}")
        raise ExceptionHandle(e, sys)


def create_chat_chain(components, chat_id=None):
    try:
        memory_manager = ChatMemory(chat_id=chat_id)
        retriever = components["retriever"]

        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question, formulate a standalone "
            "question that can be understood without the chat history. Do not answer the "
            "question. If no rewrite is needed, return it as is."
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )
        template_str = components["qa_prompt"].template
        if "{question}" in template_str:
            template_str = template_str.replace("{question}", "{input}")

        qa_prompt = ChatPromptTemplate.from_template(template_str)
        question_answer_chain = create_stuff_documents_chain(
            llm,
            qa_prompt,
            document_prompt=components.get("document_prompt"),
            document_variable_name="context",
        )

        rag_chain = create_retrieval_chain(
            history_aware_retriever,
            question_answer_chain,
        )
        final_chain = rag_chain | (lambda x: x["answer"])

        logging.info("LCEL RAG chain initialized successfully.")
        return final_chain, memory_manager

    except Exception as e:
        logging.error(f"Chat chain creation failed: {e}")
        raise ExceptionHandle(e, sys)


def create_chat_chian(components, chat_id=None):
    return create_chat_chain(components, chat_id=chat_id)
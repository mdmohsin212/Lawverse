from Lawverse.logger import logging
from Lawverse.exception import ExceptionHandle
from langchain_community.document_loaders import PyMuPDFLoader
import sys

def load_pdf_text(pdf_paths):
    documents = []
    try:
        for path in pdf_paths:
            loader = PyMuPDFLoader(str(path))
            documents.extend(loader.load())
            logging.info(f"Successfully loaded {len(documents)} pages from PDF: {path}")
        return documents
    
    except Exception as e:
        logging.error(f"Failed to extract text from PDFs: {pdf_paths}")
        raise ExceptionHandle(e, sys)
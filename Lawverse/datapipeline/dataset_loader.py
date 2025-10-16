from Lawverse.logger import logging
from Lawverse.exception import ExceptionHandle
import PyPDF2
import sys

def load_pdf_text(pdf_path : str):
    try:
        reader = PyPDF2.PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        logging.info(f"Successfully extracted text from PDF: {pdf_path}")
        return text
    
    except Exception as e:
        logging.error(f"Failed to extract text from PDF: {pdf_path}. Error: {e}")
        raise ExceptionHandle(e, sys)
from Lawverse.logger import logging
from Lawverse.exception import ExceptionHandle
import PyPDF2
import sys

def load_pdf_text(pdf_paths):
    all_text = ""
    try:
        for path in pdf_paths:
            reader = PyPDF2.PdfReader(path)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    all_text += text
            logging.info(f"Successfully extracted text from PDF: {path}")
        return all_text
    
    except Exception as e:
        logging.error(f"Failed to extract text from PDFs: {pdf_paths}")
        raise ExceptionHandle(e, sys)
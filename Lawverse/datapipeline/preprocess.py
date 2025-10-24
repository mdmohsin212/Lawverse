from concurrent.futures import ThreadPoolExecutor, as_completed
from langdetect import detect
from langchain_core.documents import Document
from deep_translator import GoogleTranslator
from langchain_text_splitters import RecursiveCharacterTextSplitter
from Lawverse.logger import logging
from Lawverse.exception import ExceptionHandle
from Lawverse.utils.config import PROCESSED_DIR
import sys
import re
from typing import List
import pickle
from pathlib import Path

def clean_text(text : str):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\(.*?\)', '', text)
    return text.strip()

def chunk_documents(documents: List, chunk_size=1200, overlap=300) -> List:
    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
        chunks = splitter.split_documents(documents)
        
        for chunk in chunks:
            chunk.page_content = clean_text(chunk.page_content)
            
        logging.info(f"Successfully split text into {len(chunks)} chunks.")
        return chunks

    except Exception as e:
        logging.error(f"Failed to chunk text. Error: {e}")
        raise ExceptionHandle(e, sys)

def translate_text(chunk_content : str, src, tgt) -> str:
    try:
        return GoogleTranslator(source=src, target=tgt).translate(chunk_content)
    
    except Exception as e:
        logging.error(f"Translation failed")
        raise ExceptionHandle(e, sys)

def create_bilingual_chunks(chunks, max_workers=8) -> Path:
    save_path = PROCESSED_DIR / "translated_chunks.pkl"
    try:
        if save_path.exists():
            logging.info(f"Translated file already exists at: {save_path}.")
            return save_path
        
        bilingual_docs = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_chunk = {
                executor.submit(translate_text, chunk.page_content, "en", "bn") : chunk for chunk in chunks
            }
            for future in as_completed(future_to_chunk):
                original_chunk = future_to_chunk[future]
                try:
                    translated_content = future.result()
                    original_content = original_chunk.page_content
                    lang = detect(original_content)
                    en_text = original_content if lang == 'en' else translated_content
                    bn_text = translated_content if lang == 'en' else original_content
                    bilingual_content = f"EN: {en_text}\nBN: {bn_text}"
                    bilingual_docs.append(
                        Document(
                            page_content=bilingual_content,
                            metadata=original_chunk.metadata
                        )
                    )
                except Exception as e:
                    logging.error(f"Error processing a translated chunk: {e}")
                    raise ExceptionHandle(e, sys)
                
        logging.info(f"Bilingual chunks completed for {len(bilingual_docs)} chunks.")
        
        with open(save_path, "wb") as f:
            pickle.dump(bilingual_docs, f)
                
        logging.info(f"Bilingual chunks saved at: {save_path}")    
        return save_path

    except Exception as e:
        logging.error(f"Translation process failed. Error: {e}")
        raise ExceptionHandle(e, sys)
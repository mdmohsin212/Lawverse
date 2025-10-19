from concurrent.futures import ThreadPoolExecutor, as_completed
from langdetect import detect
from deep_translator import GoogleTranslator
from langchain_text_splitters import RecursiveCharacterTextSplitter
from Lawverse.logger import logging
from Lawverse.exception import ExceptionHandle
from Lawverse.utils.config import PROCESSED_DIR
import sys
import re

def clean_text(text : str):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\(.*?\)', '', text)
    return text.strip()

def chunk_text(text: str, chunk_size=1200, overlap=300):
    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
        chunks = splitter.split_text(text)
        logging.info(f"Successfully split text into {len(chunks)} chunks.")
        return [clean_text(c) for c in chunks]

    except Exception as e:
        logging.error(f"Failed to chunk text. Error: {e}")
        raise ExceptionHandle(e, sys)

def translate(chunk, source, target):
    try:
        return GoogleTranslator(source=source, target=target).translate(chunk)
    
    except Exception as e:
        logging.error(f"Translation failed")
        raise ExceptionHandle(e, sys)

def translate_chunks(chunks, max_workers=8):
    try:
        translated = []
        save_path = PROCESSED_DIR / "translated_chunks.txt"
        
        if save_path.exists():
            logging.info(f"Translated file already exists at: {save_path}.")
            return save_path
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for c in chunks:
                lang = detect(c)
                if lang == "bn":
                    src, tgt = "bn", "en"
                else:
                    src, tgt = "en", "bn"
                futures[executor.submit(translate, c, src, tgt)] = c
                
            for f in as_completed(futures):
                translated.append(f.result())
        
        logging.info(f"Translation completed for {len(translated)} chunks.")
        
        with open(save_path, "w", encoding="utf-8") as f:
            for t in translated:
                f.write(t + "\n")
                
        logging.info(f"Translated chunks saved at: {save_path}")    
        return save_path

    except Exception as e:
        logging.error(f"Translation process failed. Error: {e}")
        raise ExceptionHandle(e, sys)
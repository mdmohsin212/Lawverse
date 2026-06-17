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

def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "")
    return text.strip()


def _enrich_metadata(metadata: dict, chunk_id: int) -> dict:
    metadata = dict(metadata or {})
    metadata.setdefault("chunk_id", chunk_id)
    metadata.setdefault("source", metadata.get("file_path") or metadata.get("source") or "unknown")

    if "page" in metadata:
        try:
            metadata.setdefault("page_label", int(metadata["page"]) + 1)
        except Exception:
            metadata.setdefault("page_label", metadata["page"])
    else:
        metadata.setdefault("page_label", "unknown")
    return metadata

def chunk_documents(documents: List, chunk_size=1200, overlap=300) -> List:
    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
        chunks = splitter.split_documents(documents)

        cleaned_chunks = []
        for idx, chunk in enumerate(chunks):
            chunk.page_content = clean_text(chunk.page_content)
            chunk.metadata = _enrich_metadata(chunk.metadata, idx)
            cleaned_chunks.append(chunk)

        logging.info(f"Successfully split text into {len(cleaned_chunks)} metadata-aligned chunks.")
        return cleaned_chunks

    except Exception as e:
        logging.error(f"Failed to chunk text. Error: {e}")
        raise ExceptionHandle(e, sys)


def translate_text(chunk_content: str, src, tgt) -> str:
    try:
        return GoogleTranslator(source=src, target=tgt).translate(chunk_content)
    except Exception as e:
        logging.warning(f"Translation failed; falling back to original text. Error: {e}")
        return chunk_content


def _build_bilingual_document(index: int, chunk: Document) -> Document:
    original_content = chunk.page_content
    try:
        lang = detect(original_content) if original_content else "unknown"
    except Exception:
        lang = "unknown"

    translated_content = translate_text(original_content, "en", "bn")
    en_text = original_content if lang == "en" else original_content
    bn_text = translated_content if lang == "en" else translated_content

    bilingual_content = f"EN: {en_text}\nBN: {bn_text}"
    metadata = _enrich_metadata(chunk.metadata, index)
    metadata["bilingual"] = True
    return Document(page_content=bilingual_content, metadata=metadata)


def create_bilingual_chunks(chunks, max_workers=8) -> Path:
    save_path = PROCESSED_DIR / "translated_chunks.pkl"
    try:
        if save_path.exists():
            logging.info(f"Translated file already exists at: {save_path}.")
            return save_path

        bilingual_docs = [None] * len(chunks)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(_build_bilingual_document, idx, chunk): idx
                for idx, chunk in enumerate(chunks)
            }
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    bilingual_docs[idx] = future.result()
                except Exception as e:
                    logging.error(f"Error processing translated chunk {idx}: {e}")
                    bilingual_docs[idx] = Document(
                        page_content=chunks[idx].page_content,
                        metadata=_enrich_metadata(chunks[idx].metadata, idx),
                    )

        bilingual_docs = [doc for doc in bilingual_docs if doc is not None]
        logging.info(f"Bilingual chunks completed for {len(bilingual_docs)} chunks in stable order.")

        with open(save_path, "wb") as f:
            pickle.dump(bilingual_docs, f)

        logging.info(f"Bilingual chunks saved at: {save_path}")
        return save_path

    except Exception as e:
        logging.error(f"Translation process failed. Error: {e}")
        raise ExceptionHandle(e, sys)
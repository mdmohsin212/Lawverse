from __future__ import annotations
import re
import sys
from typing import List, Tuple, Union
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from Lawverse.logger import logging
from Lawverse.exception import ExceptionHandle


STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when", "while",
    "is", "are", "am", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "for", "from", "by", "with", "as", "at", "into",
    "this", "that", "these", "those", "it", "its", "he", "she", "they", "them",
    "his", "her", "their", "we", "you", "your", "i", "me", "my",
    "shall", "may", "under", "subsection",
}


def bm25_tokenizer(text: Union[str, List[str], tuple]) -> List[str]:
    if text is None:
        return []

    if isinstance(text, (list, tuple)):
        return [
            str(token).lower().strip()
            for token in text
            if str(token).strip() and str(token).lower().strip() not in STOP_WORDS
        ]

    text = str(text).lower()
    english_tokens = re.findall(r"[a-zA-Z0-9]+", text)
    bangla_tokens = re.findall(r"[\u0980-\u09FF]+", text)
    tokens = english_tokens + bangla_tokens

    return [
        token
        for token in tokens
        if len(token) > 1 and token not in STOP_WORDS
    ]


def build_sparse_index(chunks: List[Document], k1: float = 1.5, b: float = 0.8) -> BM25Okapi:
    try:
        logging.info("Building sparse BM25 index...")
        tokenized_corpus = [
            bm25_tokenizer(chunk.page_content)
            for chunk in chunks
        ]

        bm25 = BM25Okapi(tokenized_corpus, k1=k1, b=b)
        logging.info(f"BM25 sparse index successfully built with {len(chunks)} chunks.")
        return bm25

    except Exception as e:
        logging.error(f"Failed to build BM25 sparse index. Error: {e}")
        raise ExceptionHandle(e, sys)


def bm25_retrieve(
    bm25: BM25Okapi,
    query: Union[str, List[str], tuple],
    chunks: List[Document],
    top_k: int = 10,
) -> List[Tuple[Document, float]]:
    try:
        query_tokens = bm25_tokenizer(query)
        scores = bm25.get_scores(query_tokens)

        ranked = sorted(
            enumerate(scores),
            key=lambda item: item[1],
            reverse=True,
        )[:top_k]

        results = []
        for idx, score in ranked:
            doc = chunks[idx]
            doc.metadata = dict(doc.metadata or {})
            doc.metadata["bm25_score"] = float(score)
            results.append((doc, float(score)))

        return results

    except Exception as e:
        logging.error(f"BM25 retrieval failed. Error: {e}")
        raise ExceptionHandle(e, sys)
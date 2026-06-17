import numpy as np
from rank_bm25 import BM25Okapi
from Lawverse.logger import logging
from Lawverse.exception import ExceptionHandle
import sys
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

def _safe_nltk_download(package: str):
    try:
        nltk.data.find(package)
    except LookupError:
        nltk.download(package.split('/')[-1], quiet=True)

_safe_nltk_download('tokenizers/punkt')
_safe_nltk_download('corpora/stopwords')
_safe_nltk_download('corpora/wordnet')

try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    stop_words = set()

lemmatizer = WordNetLemmatizer()

def bm25_tokenizer(text):
    text = (text or "").lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    try:
        tokens = word_tokenize(text)
    except LookupError:
        tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word and word not in stop_words]
    return tokens

def build_sparse_index(chunks, k1=1.5, b=0.75):
    try:
        tokenized_corpus = [bm25_tokenizer(chunk.page_content) for chunk in chunks]
        bm25 = BM25Okapi(tokenized_corpus, k1=k1, b=b)

        logging.info(f"BM25 sparse index successfully built with {len(chunks)} chunks.")
        return bm25

    except Exception as e:
        logging.error(f"Failed to build BM25 sparse index. Error: {e}")
        raise ExceptionHandle(e, sys)

def bm25_retrieve(bm25, chunks, query, top_k=10):
    try:
        tokenized_query = bm25_tokenizer(query)
        scores = bm25.get_scores(tokenized_query)

        num_docs = len(chunks)
        actual_top_k = min(top_k, num_docs)
        top_indices = np.argsort(scores)[-actual_top_k:][::-1]

        logging.info(f"Successfully retrieved top {len(top_indices)} relevant chunks using BM25.")
        return [(chunks[i], float(scores[i])) for i in top_indices]

    except Exception as e:
        logging.error("BM25 retrieval failed")
        raise ExceptionHandle(e, sys)
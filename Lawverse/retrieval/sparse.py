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

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
    
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def bm25_tokenizer(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return tokens

def build_sparse_index(chunks, k1=1.5, b=0.75):
    try:
        tokenized_corpus = [chunk.page_content.split() for chunk in chunks]
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
        actual_tok_k = min(top_k, num_docs)
        
        top_indices = np.argsort(scores)[-actual_tok_k:][::-1]

        logging.info(f"Successfully retrieved top {len(top_indices)} relevant chunks using BM25.")
        return [(chunks[i].page_content, scores[i]) for i in top_indices]

    except Exception as e:
        logging.error(f"BM25 retrieval failed")
        raise ExceptionHandle(e, sys)
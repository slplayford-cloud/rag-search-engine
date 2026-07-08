#!/usr/bin/env python3

from nltk import defaultdict

from .inverted_index import InvertedIndex, load_index
from .search_utils import BM25_B, BM25_K1, DEFAULT_SEARCH_LIMIT
from .text_processing import single_token, tokenize_text


def build_command():
    iv_index = InvertedIndex()
    iv_index.build()
    iv_index.save()


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    iv_index = load_index()

    results = []
    query_tokens = tokenize_text(query)
    for tk in query_tokens:
        results.extend(iv_index.get_documents(tk))
        if len(results) >= limit:
            break

    return results


def bm25_search(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[tuple]:
    iv_index = load_index()

    # doc_id -> total BM25 score
    scores = defaultdict(float)
    query_tokens = tokenize_text(query)
    
    for tk in query_tokens:
        for doc in iv_index.get_documents(tk):
            scores[doc["id"]] += iv_index.get_bm25(doc["id"], tk)

    return sorted(scores.items(), key=lambda p: p[1], reverse=True)[:limit]


def term_frequency_command(doc_id: int, term: str) -> int:
    iv_index = load_index()
    tk = single_token(term)
    return iv_index.get_tf(doc_id, tk)


def inverse_frequency_command(term: str) -> float:
    iv_index = load_index()
    tk = single_token(term)
    return iv_index.get_idf(tk)


def tf_idf_command(doc_id: int, term: str) -> float:
    iv_index = load_index()
    tk = single_token(term)
    return iv_index.get_tfidf(doc_id, tk)


def bm25_idf_command(term: str) -> float:
    iv_index = load_index()
    tk = single_token(term)
    return iv_index.get_bm25_idf(tk)


def bm25_tf_command(doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
    iv_index = load_index()
    tk = single_token(term)
    return iv_index.get_bm25_tf(doc_id, tk, k1, b)



#!/usr/bin/env python3

import math
import os
import pickle
from collections import Counter, defaultdict

from .search_utils import BM25_B, BM25_K1, Movie, load_movies, CACHE_DIR
from .text_processing import tokenize_text


class InvertedIndex:
    def __init__(self):
        self.index: dict[str, set[int]] = defaultdict(set)  # this translates token to doc_id list
        self.docmap: dict[int, Movie] = {}  # doc_id to movie object
        self.term_frequencies: dict[int, Counter] = {}  # doc_id to word Counter
        self.doc_lengths: dict[int, int] = {}

        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.tf_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")

    """
    CLASS HELPER FUNCTIONS
    """

    def build(self):
        movies = load_movies()
        for m in movies:
            self.docmap[m["id"]] = m
            self._add_document(m["id"], f"{m['title']} {m['description']}")

    def _add_document(self, doc_id: int, text: str):
        tokens = tokenize_text(text)
        self._record_term_frequencies(doc_id, tokens)
        self._record_doc_lengths(doc_id, len(tokens))

        for token in tokens:
            self.index[token].add(doc_id)

    def _record_term_frequencies(self, doc_id: int, tokens: list[str]):
        self.term_frequencies[doc_id] = Counter(tokens)

    def _record_doc_lengths(self, doc_id: int, token_count: int):
        self.doc_lengths[doc_id] = token_count

    def _get_avg_doc_length(self) -> float:
        try:
            return sum(self.doc_lengths.values()) / len(self.doc_lengths)
        except:
            return 0.0

    """
    LOADING AND SAVING FUNCTIONS FOR CACHING
    """

    def save(self):
        os.makedirs(CACHE_DIR, exist_ok=True)

        with open(self.index_path, "wb") as idx_file:
            pickle.dump(self.index, idx_file)
        with open(self.docmap_path, "wb") as docmap_file:
            pickle.dump(self.docmap, docmap_file)
        with open(self.tf_path, "wb") as freq_file:
            pickle.dump(self.term_frequencies, freq_file)
        with open(self.doc_lengths_path, "wb") as doc_len_file:
            pickle.dump(self.doc_lengths, doc_len_file)

    def load(self):
        with open(self.index_path, "rb") as idx_file:
            self.index = pickle.load(idx_file)
        with open(self.docmap_path, "rb") as docmap_file:
            self.docmap = pickle.load(docmap_file)
        with open(self.tf_path, "rb") as freq_file:
            self.term_frequencies = pickle.load(freq_file)
        with open(self.doc_lengths_path, "rb") as doc_len_file:
            self.doc_lengths = pickle.load(doc_len_file)

    """
    INDEX SCORING METHODS
    """

    def get_documents(self, term: str):
        doc_ids = self.index[term]

        documents = []
        for id in doc_ids:
            documents.append(self.docmap[id])

        return documents

    def get_tf(self, doc_id: int, token: str):
        try:
            return self.term_frequencies[doc_id][token]
        except:
            return 0

    def get_idf(self, term: str):
        return math.log((len(self.docmap) + 1) / (len(self.index[term]) + 1))

    def get_tfidf(self, doc_id: int, token: str):
        return self.get_tf(doc_id, token) * self.get_idf(token)

    def get_bm25_idf(self, term: str) -> float:
        # log((N - df + 0.5) / (df + 0.5) + 1) N == total documents df == doc frequency
        N = len(self.docmap)
        df = len(self.get_documents(term))
        return math.log((N - df + 0.5) / (df + 0.5) + 1)

    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B):
        tf = self.get_tf(doc_id, term)
        # Length normalization factor
        length_norm = 1 - b + b * (self.doc_lengths[doc_id] / self._get_avg_doc_length())

        # Apply to term frequency
        tf_component = (tf * (k1 + 1)) / (tf + k1 * length_norm)

        return tf_component

    def get_bm25(self, doc_id, term) -> float:
        return self.get_bm25_tf(doc_id, term) * self.get_bm25_idf(term)

class IndexNotBuiltError(Exception):
    """Raised when the cached index can't be loaded from disk."""

def load_index() -> InvertedIndex:
    iv_index = InvertedIndex()
    try:
        iv_index.load()
    except (FileNotFoundError, EOFError, pickle.UnpicklingError) as e:
        raise IndexNotBuiltError("Index not found on disk: run 'build' first.") from e

    return iv_index
    

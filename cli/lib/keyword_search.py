#!/usr/bin/env python3

import math
import os
import pickle
import string
from collections import Counter, defaultdict

from nltk.stem import PorterStemmer

from .search_utils import (DEFAULT_SEARCH_LIMIT, STOPWORDS_PATH, Movie,
                           load_movies)


class InvertedIndex:
    def __init__(self):
        self.index: dict[str, set[int]] = defaultdict(
            set
        )  # this translates token to doc_id list
        self.docmap: dict[int, Movie] = {}  # doc_id to movie object

        self.term_frequencies: dict[int, Counter] = {}  # doc_id to word Counter

    def _add_document(self, doc_id: int, text: str):
        tokens = tokenize_text(text)
        self._update_frequency(doc_id, tokens)

        for token in tokens:
            self.index[token].add(doc_id)

    def _update_frequency(self, doc_id: int, tokens: list[str]):
        self.term_frequencies[doc_id] = Counter(tokens)

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
        
    def build(self):
        movies = load_movies()
        for m in movies:
            self.docmap[m["id"]] = m
            self._add_document(m["id"], f"{m['title']} {m['description']}")

    def save(self):
        os.makedirs("cache", exist_ok=True)

        with open("cache/index.pkl", "wb") as idx_file:
            pickle.dump(self.index, idx_file)
        with open("cache/docmap.pkl", "wb") as docmap_file:
            pickle.dump(self.docmap, docmap_file)
        with open("cache/term_frequencies.pkl", "wb") as freq_file:
            pickle.dump(self.term_frequencies, freq_file)

    def load(self):
        with open("cache/index.pkl", "rb") as idx_file:
            self.index = pickle.load(idx_file)
        with open("cache/docmap.pkl", "rb") as docmap_file:
            self.docmap = pickle.load(docmap_file)
        with open("cache/term_frequencies.pkl", "rb") as freq_file:
            self.term_frequencies = pickle.load(freq_file)


def build_command():
    iv_index = InvertedIndex()
    iv_index.build()
    iv_index.save()


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    # movies = load_movies()
    iv_index = InvertedIndex()
    try:
        iv_index.load()
    except:
        print("Could not laod movies index from disk")
        exit()

    results = []
    query_tokens = tokenize_text(query)
    for tk in query_tokens:
        results.extend(iv_index.get_documents(tk))
        if len(results) >= limit:
            break

    return results


def term_frequency(doc_id: int, term: str) -> int:
    iv_index = InvertedIndex()
    try:
        iv_index.load()
    except:
        print("Could not laod movies index from disk")
        exit()

    tk = single_token(term)
    return iv_index.get_tf(doc_id, tk)


def inverse_frequency(term: str) -> float:
    iv_index = InvertedIndex()
    try:
        iv_index.load()
    except:
        print("Could not laod movies index from disk")
        exit()

    tk = single_token(term)
    return iv_index.get_idf(tk)

def get_tf_idf(doc_id: int, term: str) -> float:
    iv_index = InvertedIndex()
    try:
        iv_index.load()
    except:
        print("Could not laod movies index from disk")
        exit()

    tk = single_token(term)
    return iv_index.get_tfidf(doc_id, tk)


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))

    return text


def load_stop_words() -> set:
    result = set()

    with open(STOPWORDS_PATH, "r") as f:
        text = f.read()
        text = text.splitlines()

        result = set(map(preprocess_text, text))

    return result


STOPWORDS = load_stop_words()


def tokenize_text(text: str) -> list[str]:
    text = preprocess_text(text)

    words = text.split()
    words = list(filter(lambda s: s not in STOPWORDS, words))

    stemmer = PorterStemmer()
    words = list(map(stemmer.stem, words))

    return words


def single_token(term: str):
    tokens = tokenize_text(term)
    if len(tokens) != 1:
        raise RuntimeError("Not a single token")

    return tokens[0]


def has_matching_token(query: list[str], title: list[str]) -> bool:
    for query_token in query:
        for title_token in title:
            if query_token in title_token:
                return True

    return False

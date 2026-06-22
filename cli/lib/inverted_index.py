#!/usr/bin/env python3

import os
import pickle
from collections import defaultdict

from .keyword_search import tokenize_text
from .search_utils import load_movies


class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(list)
        self.docmap = {}

    def _add_document(self, doc_id: int, text: str):
        tokens = tokenize_text(text)

        for token in tokens:
            self.index[token].append(doc_id)

    def get_documents(self, term: str):
        doc_ids = self.index[term]

        documents = []
        for id in doc_ids:
            documents.append(self.docmap[id])

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

def build_command():
    iv_index = InvertedIndex()
    iv_index.build()
    iv_index.save()

    print(f"First document ID for 'merida': {iv_index.index['merida'][0]}")

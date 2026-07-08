#!/usr/bin/env python3

import string

from nltk.stem import PorterStemmer

from .search_utils import STOPWORDS_PATH


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

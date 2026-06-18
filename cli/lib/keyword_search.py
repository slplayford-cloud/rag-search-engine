#!/usr/bin/env python3

import string

from .search_utils import DEFAULT_SEARCH_LIMIT, STOPWORDS_PATH, load_movies


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:

    movies = load_movies()

    results = []
    for movie in movies:
        query_tokens = tokenize_text(query)
        title_tokens = tokenize_text(movie["title"])

        if has_matching_token(query_tokens, title_tokens):
            results.append(movie)
            if len(results) >= limit:
                break

    return results


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

    return words


def has_matching_token(query: list[str], title: list[str]) -> bool:
    for query_token in query:
        for title_token in title:
            if query_token in title_token:
                return True

    return False

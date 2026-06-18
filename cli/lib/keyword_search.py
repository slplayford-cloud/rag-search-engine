#!/usr/bin/env python3

import string

from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies

DEFAULT_TOKEN_LIMIT = 1

def search_command(
        query: str, limit: int = DEFAULT_SEARCH_LIMIT 
) -> list[dict]:

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

def tokenize_text(text: str) -> list[str]:
    text = preprocess_text(text)

    words = text.split()
    words = list(filter(lambda s: s != "", words))

    return words

def has_matching_token(query: list[str], title: list[str]) -> bool:
    for query_token in query:
        for title_token in title:
            if query_token in title_token:
                return True

    return False

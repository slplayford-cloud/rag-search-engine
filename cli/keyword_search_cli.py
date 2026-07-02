#!/usr/bin/env python3

import argparse
import json

from lib.keyword_search import search_command, build_command, term_frequency, inverse_frequency

# Load the file from the param. into a dict
def load_json(file_path: str):
    with open(file_path, "r") as fp:
        data = json.load(fp)

    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build the inverted index and save it to disk")

    tf_parser = subparsers.add_parser("tf", help="Get the frequency of a token in a given doc_id")
    tf_parser.add_argument("doc_id", type=int, help="doc_id of the search document")
    tf_parser.add_argument("term", type=str, help="The token count to look up")

    idf_parser = subparsers.add_parser("idf", help="Get the inverse frequency for a token")
    idf_parser.add_argument("term", type=str, help="Token to look up")
    

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            matches = search_command(args.query)
            for i, movie in enumerate(matches[:5]):
                print(f"{i+1}. {movie['title']}")
        case "build":
            build_command()
        case "tf":
            print(f"Looking for \'{args.term}\' in document ({args.doc_id})")
            fq = term_frequency(args.doc_id, args.term)
            print(f"Found \'{args.term}\' in document ({args.doc_id}) {fq} times")
        case "idf":
            print(f"Looking for inverse frequency of \'{args.term}\'")
            idf = inverse_frequency(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case _:
            parser.print_help()



if __name__ == "__main__":
    main()

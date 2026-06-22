#!/usr/bin/env python3

import argparse
import json

from lib.keyword_search import search_command
from lib.inverted_index import build_command

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

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            matches = search_command(args.query)
            for i, movie in enumerate(matches[:5]):
                print(f"{i+1}. {movie['title']}")
        case "build":
            build_command()
        case _:
            parser.print_help()



if __name__ == "__main__":
    main()

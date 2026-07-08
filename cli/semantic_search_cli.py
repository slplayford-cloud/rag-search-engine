import argparse
from lib.semantic_search import verify_model

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    match args.command:
        case "verify":
           verify_model() 
        case _:
            parser.print_help()


def build_parser():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verify the embedding model used")

    return parser

if __name__ == "__main__":
    main()

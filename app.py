import argparse

from rag.llm import answer
from rag.retrieve import search


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--search-only", action="store_true")
    args = parser.parse_args()

    source, passage = search(args.question)
    print("Manba:", source)
    print("Topilgan parcha:", passage)

    if not args.search_only:
        print("Qwen javobi:", answer(args.question, passage))


if __name__ == "__main__":
    main()

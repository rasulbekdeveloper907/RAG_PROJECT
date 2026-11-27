import argparse

from qdrant_client import QdrantClient
from course_rag.index import COLLECTION, STORE
from rag.embeddings import embed_question
from rag.llm import answer


def search(question, k):
    pointer = STORE / "current"
    if not pointer.exists():
        raise SystemExit("Avval python -m course_rag.index buyrug‘ini bajaring.")

    db_path = STORE / pointer.read_text(encoding="utf-8").strip() / "db"
    query = embed_question(question).tolist()
    client = QdrantClient(path=str(db_path))
    try:
        hits = client.query_points(
            collection_name=COLLECTION,
            query=query,
            limit=k,
            with_payload=True,
        ).points
        return [hit.payload for hit in hits]
    finally:
        client.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--search-only", action="store_true")
    parser.add_argument("--k", type=int, choices=range(1, 6), default=2)
    args = parser.parse_args()

    parts = []
    for i, row in enumerate(search(args.question, args.k), start=1):
        part = f"[{i}] {row['source']} | {row['chunk']}-parcha\n{row['text']}"
        parts.append(part)
        print(part, end="\n\n")

    if not parts:
        raise SystemExit("Qdrant’da parcha topilmadi.")
    if not args.search_only:
        print("Qwen javobi:", answer(args.question, "\n\n".join(parts)))


if __name__ == "__main__":
    main()

import re

import numpy as np

from rag.embeddings import ROOT, embed_passages


DOCS = ROOT / "data" / "docs"
INDEX = ROOT / "data" / "index" / "rag.npz"


def main():
    texts = []
    sources = []

    for path in sorted(DOCS.rglob("*.txt")):
        content = path.read_text(encoding="utf-8-sig").strip()
        for number, paragraph in enumerate(re.split(r"\n\s*\n", content), start=1):
            if paragraph.strip():
                texts.append(paragraph.strip())
                sources.append(f"{path.relative_to(DOCS)} | {number}-parcha")

    if not texts:
        raise SystemExit("data/docs ichiga matn yozilgan .txt fayl qo‘shing.")

    vectors = embed_passages(texts)
    INDEX.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(INDEX, vectors=vectors, texts=texts, sources=sources)
    print(f"Indeks tayyor. Parchalar soni: {len(texts)}")


if __name__ == "__main__":
    main()

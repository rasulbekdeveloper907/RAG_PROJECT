import json
import tempfile
from pathlib import Path

from qdrant_client import QdrantClient, models
from rag.embeddings import ROOT, embed_passages, get_model


DOCS = Path(__file__).resolve().parent / "docs"
STORE = ROOT / "data" / "index" / "course"
COLLECTION = "course_chunks"


def load_documents():
    """TXT va MD fayllarni manba nomi bilan o‘qish."""
    for path in sorted(DOCS.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".txt", ".md"}:
            continue
        raw = path.read_text(encoding="utf-8-sig")
        text = "\n".join(line.rstrip() for line in raw.splitlines()).strip()
        if text:
            yield path.relative_to(DOCS).as_posix(), text


def split_text(text, tokenizer, size=160, overlap=24):
    """Uzun matnni token bo‘yicha bo‘lish, asl matnni saqlash."""
    offsets = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_offsets_mapping=True,
        verbose=False,
    )["offset_mapping"]

    for start in range(0, len(offsets), size - overlap):
        end = min(start + size, len(offsets))
        yield text[offsets[start][0]:offsets[end - 1][1]]
        if end == len(offsets):
            break


def main():
    documents = list(load_documents())
    if not documents:
        raise SystemExit("course_rag/docs ichiga UTF-8 .txt yoki .md fayl qo‘shing.")

    tokenizer = get_model().tokenizer
    chunks = []
    for source, text in documents:
        for number, part in enumerate(split_text(text, tokenizer), start=1):
            chunks.append({"source": source, "chunk": number, "text": part})

    if not chunks:
        raise SystemExit("Indekslash uchun matn topilmadi.")

    vectors = embed_passages([chunk["text"] for chunk in chunks])
    dimension = int(vectors.shape[1])

    # Har bir indekslash alohida papkada tugallanadi.
    # Qidiruv yangi indeks tayyor bo‘lgandan keyin unga o‘tadi.
    STORE.mkdir(parents=True, exist_ok=True)
    build = Path(tempfile.mkdtemp(prefix="build-", dir=STORE))
    client = QdrantClient(path=str(build / "db"))
    try:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=models.VectorParams(
                size=dimension, distance=models.Distance.COSINE
            ),
        )
        points = [
            models.PointStruct(id=i, vector=vector.tolist(), payload=chunk)
            for i, (vector, chunk) in enumerate(zip(vectors, chunks, strict=True))
        ]
        client.upsert(collection_name=COLLECTION, points=points, wait=True)
        count = client.count(collection_name=COLLECTION, exact=True).count
        if count != len(chunks):
            raise RuntimeError("Qdrant’dagi vektorlar soni parchalar soniga mos kelmadi.")
    finally:
        client.close()

    with (build / "chunks.jsonl").open("w", encoding="utf-8") as output:
        for chunk in chunks:
            print(json.dumps(chunk, ensure_ascii=False), file=output)

    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=STORE, delete=False
    ) as pointer:
        pointer.write(build.name)
    Path(pointer.name).replace(STORE / "current")

    print(f"Hujjatlar: {len(documents)} | Parchalar: {len(chunks)}")
    print(f"Qdrant vektorlari: {count} | Vektor o‘lchami: {dimension}")
    print("Indekslangan matnlar:", build / "chunks.jsonl")


if __name__ == "__main__":
    main()

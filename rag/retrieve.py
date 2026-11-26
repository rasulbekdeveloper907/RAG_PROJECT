import numpy as np

from rag.embeddings import embed_question
from rag.ingest import INDEX


def search(question):
    if not INDEX.exists():
        raise SystemExit("Avval python -m rag.ingest buyrug‘ini bajaring.")

    query = embed_question(question)
    with np.load(INDEX, allow_pickle=False) as index:
        scores = index["vectors"] @ query
        best = int(scores.argmax())
        return str(index["sources"][best]), str(index["texts"][best])

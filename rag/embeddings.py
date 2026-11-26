from functools import lru_cache
from pathlib import Path

from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def get_model():
    return SentenceTransformer(
        str(ROOT / "models" / "e5-base"),
        device="cpu",
        local_files_only=True,
    )


def embed_passages(texts):
    passages = ["passage: " + text for text in texts]
    return get_model().encode(passages, normalize_embeddings=True)


def embed_question(question):
    return get_model().encode("query: " + question, normalize_embeddings=True)

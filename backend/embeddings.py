"""
embeddings.py — Local text embeddings using sentence-transformers.

Runs entirely on your machine — no internet required after the first
model download (model is cached by HuggingFace in ~/.cache/huggingface).
"""

from functools import lru_cache
from sentence_transformers import SentenceTransformer

# A lightweight but high-quality model for semantic similarity.
# 384-dim vectors, ~80 MB download.
MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    """
    Load (and cache) the embedding model once.
    Explicitly runs on CPU for maximum compatibility.
    """
    print(f"[EMBED] Loading model '{MODEL_NAME}'…")

    model = SentenceTransformer(
        MODEL_NAME,
        device="cpu"   # Ensures it runs on any laptop without GPU issues
    )

    print(
        f"[EMBED] Model loaded — vector dim: {model.get_sentence_embedding_dimension()}"
    )

    return model


def get_embedding(text: str) -> list[float]:
    """
    Return a normalised embedding vector for the given text.
    Used for single queries (e.g. user questions).
    """
    model = _load_model()

    vector = model.encode(
        text,
        normalize_embeddings=True
    )

    return vector.tolist()


def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """
    Batch-encode multiple texts (much faster than one-by-one).
    Used when indexing chunks from PDFs.
    """
    if not texts:
        return []

    model = _load_model()

    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=False
    )

    return [v.tolist() for v in vectors]
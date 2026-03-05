"""
vectorstore.py — Persistent local vector database using ChromaDB.

All data is stored in ./chroma_db on disk.
No internet connection required.
"""

import chromadb
from chromadb.config import Settings
from pathlib import Path

DB_PATH = str(Path(__file__).parent / "chroma_db")
COLLECTION_NAME = "notes_chunks"


class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=DB_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"[STORE] ChromaDB initialised — {self.collection.count()} existing chunks")

    # ── Write ─────────────────────────────────────────────────────────────────

    def add(
        self,
        doc_id: str,
        text: str,
        embedding: list[float],
        metadata: dict,
    ) -> None:
        """Upsert a single chunk."""
        self.collection.upsert(
            ids=[doc_id],
            documents=[text],
            embeddings=[embedding],
            metadatas=[metadata],
        )

    def add_batch(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> None:
        """Upsert a batch of chunks (faster)."""
        self.collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    # ── Read ──────────────────────────────────────────────────────────────────

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        """
        Return the top_k most similar chunks.
        Each result is {"text": str, "metadata": dict, "score": float}.
        """

        if self.collection.count() == 0:
            return []

        top_k = min(top_k, self.collection.count())

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        # Safety check in case Chroma returns empty results
        if not results or not results["documents"] or not results["documents"][0]:
            return []

        output = []

        for text, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # Convert cosine distance → similarity score
            similarity = 1.0 - dist

            output.append({
                "text": text,
                "metadata": meta,
                "score": similarity
            })

        # Sort results by similarity (highest first)
        output.sort(key=lambda x: x["score"], reverse=True)

        if not output:
            return []

        # Dynamic similarity filtering
        top_score = output[0]["score"]

        filtered = [
            r for r in output
            if r["score"] >= top_score * 0.6
        ]

        return filtered

    def count(self) -> int:
        return self.collection.count()

    def clear(self) -> None:
        """Delete all chunks (keeps the collection schema)."""
        self.client.delete_collection(COLLECTION_NAME)

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        print("[STORE] All chunks deleted.")
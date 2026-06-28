"""
Vector store wrapper around ChromaDB.

Persists document chunks + embeddings to disk so the knowledge base survives
Streamlit reloads. Embeddings are generated via Gemini (text-embedding-004) and
passed in explicitly so ChromaDB doesn't need its own embedding function.
"""
from __future__ import annotations

import logging
import os
from typing import List, Optional

import chromadb
from chromadb.config import Settings

from core.document_processor import Document
from core.gemini_client import GeminiClient

logger = logging.getLogger(__name__)

COLLECTION_NAME = "multimodal_kb"


class VectorStore:
    """Persistent ChromaDB-backed vector store."""

    def __init__(self, persist_dir: str = "data/chroma_db"):
        os.makedirs(persist_dir, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    # ---------- Writes ----------

    def add_documents(self, docs: List[Document], gemini: GeminiClient) -> int:
        """Embed and upsert documents. Returns the number of chunks added."""
        if not docs:
            return 0

        # Deduplicate against existing chunk_ids to avoid re-embedding on re-upload.
        existing = set()
        try:
            existing_ids = self._collection.get(ids=[d.chunk_id for d in docs]).get("ids", [])
            existing = set(existing_ids)
        except Exception:
            pass

        new_docs = [d for d in docs if d.chunk_id not in existing]
        if not new_docs:
            logger.info("All %d chunks already present; skipping.", len(docs))
            return 0

        texts = [d.text for d in new_docs]
        embeddings = gemini.embed_batch(texts)

        self._collection.upsert(
            ids=[d.chunk_id for d in new_docs],
            documents=texts,
            embeddings=embeddings,
            metadatas=[
                {
                    "source": d.metadata.get("source", "unknown"),
                    "modality": d.metadata.get("modality", "unknown"),
                    "chunk_idx": d.metadata.get("chunk_idx", 0),
                }
                for d in new_docs
            ],
        )
        logger.info("Added %d new chunks to the vector store.", len(new_docs))
        return len(new_docs)

    # ---------- Reads ----------

    def query(
        self,
        query_text: str,
        gemini: GeminiClient,
        top_k: int = 5,
    ) -> List[dict]:
        """Return top-k chunks with their text, source, modality, and similarity score."""
        query_embedding = gemini.embed(query_text)
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        chunks: List[dict] = []
        if not result or not result.get("documents"):
            return chunks

        docs = result["documents"][0]
        metas = result["metadatas"][0]
        dists = result["distances"][0]

        for doc, meta, dist in zip(docs, metas, dists):
            # ChromaDB returns cosine distance; similarity = 1 - distance.
            similarity = max(0.0, 1.0 - float(dist))
            chunks.append(
                {
                    "text": doc,
                    "source": meta.get("source", "unknown"),
                    "modality": meta.get("modality", "unknown"),
                    "similarity": similarity,
                }
            )
        return chunks

    # ---------- Maintenance ----------

    def clear(self) -> None:
        """Drop the collection and recreate it empty."""
        try:
            self._client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def stats(self) -> dict:
        """Return basic stats about the store."""
        try:
            count = self._collection.count()
        except Exception:
            count = 0
        return {"total_chunks": count}

    def list_sources(self) -> List[str]:
        """Return the distinct list of source filenames in the store."""
        try:
            all_data = self._collection.get(include=["metadatas"])
            sources = {m.get("source", "unknown") for m in all_data.get("metadatas", [])}
            return sorted(sources)
        except Exception:
            return []

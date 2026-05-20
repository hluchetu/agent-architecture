from __future__ import annotations

import httpx
import chromadb
from chromadb import Collection

from agent_architecture.config import Settings


class SemanticStore:
    """Stores and retrieves facts using vector similarity search."""

    def __init__(
        self,
        settings: Settings | None = None,
        collection_name: str = "semantic_memory",
        persist_path: str = "chroma_db",
    ) -> None:
        self.settings = settings or Settings()
        self._client = chromadb.PersistentClient(path=persist_path)
        self._collection: Collection = self._client.get_or_create_collection(collection_name)

    async def _embed(self, text: str) -> list[float]:
        async with httpx.AsyncClient(timeout=self.settings.ollama_timeout_seconds) as client:
            response = await client.post(
                f"{self.settings.ollama_base_url}/api/embeddings",
                json={"model": self.settings.ollama_embedding_model, "prompt": text},
            )
        response.raise_for_status()
        return response.json()["embedding"]

    async def add(self, fact_id: str, fact: str, metadata: dict | None = None) -> None:
        embedding = await self._embed(fact)
        self._collection.add(
            ids=[fact_id],
            embeddings=[embedding],
            documents=[fact],
            metadatas=[metadata or {}],
        )

    async def search(self, query: str, n_results: int = 3) -> list[str]:
        embedding = await self._embed(query)
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
        )
        return results["documents"][0] if results["documents"] else []

    def delete(self, fact_id: str) -> None:
        self._collection.delete(ids=[fact_id])

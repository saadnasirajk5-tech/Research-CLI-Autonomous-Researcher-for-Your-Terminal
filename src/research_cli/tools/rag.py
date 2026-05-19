"""
RAG Tool - Local vector store.

Primary: ChromaDB for full embedding-based retrieval.
Fallback: Simple keyword-based search when ChromaDB is unavailable.
Small models only see the top-k most relevant snippets per task.
"""

import os
import uuid
from research_cli.config import Config

CHROMADB_AVAILABLE = False
try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    pass


class LocalRAG:
    """
    Local RAG index for storing and retrieving research snippets.

    Uses ChromaDB when available, falls back to keyword search.
    Each finding is stored with metadata for filtering.
    """

    def __init__(self, config: Config | None = None):
        """Initialize the RAG index."""
        self.config = config or Config()
        self.config.initialize()
        self._client = None
        self._collection = None
        self._fallback_store: list[dict] = []
        self._use_fallback = not CHROMADB_AVAILABLE

    def _get_client(self):
        """Lazy-load ChromaDB client."""
        if self._client is None and CHROMADB_AVAILABLE:
            self._client = chromadb.PersistentClient(
                path=str(self.config.storage.chroma_dir)
            )
        return self._client

    def _get_collection(self):
        """Get or create the findings collection."""
        if self._collection is None and self._get_client() is not None:
            self._collection = self._client.get_or_create_collection(
                name="research_findings",
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def add_finding(self, text: str, task_id: str = "", source_url: str = "", metadata: dict | None = None) -> str:
        """Add a research finding to the index."""
        doc_id = str(uuid.uuid4())
        meta = {"task_id": task_id, "source_url": source_url or ""}
        if metadata:
            meta.update(metadata)
        if not self._use_fallback and self._get_collection() is not None:
            collection = self._get_collection()
            collection.add(documents=[text], ids=[doc_id], metadatas=[meta])
        else:
            self._fallback_store.append({
                "id": doc_id,
                "text": text,
                "task_id": task_id,
                "source_url": source_url or "",
                "metadata": meta,
            })
        return doc_id

    def query(self, text: str, n_results: int = 3, task_id: str | None = None) -> list[dict]:
        """Retrieve the most relevant findings for a query."""
        if not self._use_fallback and self._get_collection() is not None:
            collection = self._get_collection()
            where = {"task_id": task_id} if task_id else None
            results = collection.query(
                query_texts=[text],
                n_results=n_results,
                where=where,
            )
            findings = []
            if results and results.get("documents"):
                for i, doc in enumerate(results["documents"][0]):
                    meta = results["metadatas"][0][i] if results.get("metadatas") else {}
                    dist = results["distances"][0][i] if results.get("distances") else 0.0
                    findings.append({
                        "text": doc,
                        "source_url": meta.get("source_url", ""),
                        "task_id": meta.get("task_id", ""),
                        "distance": dist,
                    })
            return findings
        else:
            return self._keyword_query(text, n_results, task_id)

    def _keyword_query(self, text: str, n_results: int, task_id: str | None) -> list[dict]:
        """Fallback: simple keyword matching."""
        words = set(text.lower().split())
        scored = []
        for item in self._fallback_store:
            if task_id and item.get("task_id") != task_id:
                continue
            item_words = set(item["text"].lower().split())
            overlap = len(words & item_words)
            scored.append((overlap, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for _, item in scored[:n_results]:
            results.append({
                "text": item["text"],
                "source_url": item.get("source_url", ""),
                "task_id": item.get("task_id", ""),
                "distance": 0.0,
            })
        return results

    def clear(self):
        """Delete all findings from the index."""
        if self._collection is not None and self._client is not None:
            try:
                self._client.delete_collection("research_findings")
            except Exception:
                pass
            self._collection = None
        self._fallback_store.clear()

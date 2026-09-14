"""Persistent ChromaDB vector store wrapper used as the RAG backend for the research agent."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.utils import embedding_functions

DEFAULT_PERSIST_DIR = str(Path(__file__).resolve().parent.parent / "chroma_db")
DEFAULT_COLLECTION_NAME = "games"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


class VectorStoreManager:
    """Wraps a persistent ChromaDB collection with add/query/reset helpers."""

    def __init__(
        self,
        persist_directory: str = DEFAULT_PERSIST_DIR,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        base_url: Optional[str] = None,
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            model_name=embedding_model,
            api_base=base_url or os.getenv("OPENAI_BASE_URL"),
        )
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self):
        return self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
        )

    def add_documents(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Embed and upsert documents, keyed by stable id (e.g. a slugified title)."""
        self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    def query(self, text: str, k: int = 5) -> Dict[str, Any]:
        """Run a semantic search and return the top-k matches with distances."""
        return self.collection.query(query_texts=[text], n_results=k)

    def reset(self) -> None:
        """Drop and recreate the collection, clearing all stored documents."""
        self.client.delete_collection(self.collection_name)
        self.collection = self._get_or_create_collection()

    def count(self) -> int:
        return self.collection.count()

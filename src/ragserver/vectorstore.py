from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import time

import chromadb
from openai import RateLimitError

from .azure_clients import embed_texts
from .config import Settings


@dataclass
class Document:
    doc_id: str
    text: str
    metadata: dict[str, Any]


class VectorStore:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = chromadb.PersistentClient(path=str(settings.vector_db_path))
        self.collection = self.client.get_or_create_collection("fastmcp-code")

    def add(self, documents: list[Document], batch_size: int = 32) -> None:
        if not documents:
            return

        for start in range(0, len(documents), batch_size):
            batch = documents[start : start + batch_size]
            while True:
                try:
                    embeddings = embed_texts([doc.text for doc in batch], self.settings)
                    break
                except RateLimitError:
                    time.sleep(10)

            self.collection.upsert(
                ids=[doc.doc_id for doc in batch],
                embeddings=embeddings,
                documents=[doc.text for doc in batch],
                metadatas=[doc.metadata for doc in batch],
            )

    def search(self, query: str, top_k: int = 5) -> list[Document]:
        embedding = embed_texts([query], self.settings)[0]
        result = self.collection.query(query_embeddings=[embedding], n_results=top_k)
        docs: list[Document] = []
        for idx, doc_id in enumerate(result["ids"][0]):
            docs.append(
                Document(
                    doc_id=doc_id,
                    text=result["documents"][0][idx],
                    metadata=result["metadatas"][0][idx],
                )
            )
        return docs

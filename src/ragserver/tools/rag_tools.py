from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
from fastmcp import FastMCP

from ..azure_clients import chat_with_context
from ..config import Settings
from ..vectorstore import VectorStore


def _fetch_github_file(path: str, ref: str, token: str | None) -> str:
    url = f"https://api.github.com/repos/jlowin/fastmcp/contents/{path}"
    headers: dict[str, str] = {"Accept": "application/vnd.github.v3.raw", "User-Agent": "fastmcp-rag"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    params = {"ref": ref}
    response = httpx.get(url, headers=headers, params=params, timeout=30)
    response.raise_for_status()
    return response.text


def register_rag_tools(mcp: FastMCP, store: VectorStore, settings: Settings) -> None:
    @mcp.tool
    def search_fastmcp(query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Semantic søk i fastmcp-koden (lokal indeks)."""
        docs = store.search(query, top_k=top_k)
        return [
            {
                "path": doc.metadata.get("path"),
                "chunk": doc.metadata.get("chunk"),
                "text": doc.text,
            }
            for doc in docs
        ]

    @mcp.tool
    def ask_fastmcp(question: str, top_k: int = 4) -> str:
        """Svar på spørsmål ved å hente kontekst fra fastmcp-indeksen og bruke Azure OpenAI."""
        docs = store.search(question, top_k=top_k)
        contexts = [f"{doc.metadata.get('path')}: {doc.text}" for doc in docs]
        return chat_with_context(question, contexts, settings)

    @mcp.tool
    def get_file_from_disk(path: str) -> str:
        """Returner filinnhold fra lokal fastmcp-klone."""
        file_path = settings.index_root / path
        if not file_path.exists():
            raise FileNotFoundError(f"Fant ikke filen {path} i lokal indeks")
        return file_path.read_text(encoding="utf-8")

    @mcp.tool
    def get_file_from_github(path: str, ref: str = "main") -> str:
        """Hent fil fra GitHub API (nyttig når lokal indeks er utdatert)."""
        return _fetch_github_file(path, ref, settings.github_token)

    @mcp.tool
    def list_local_files(limit: int = 50) -> list[str]:
        """List opp de første filene i den lokale klonen (for å oppdage sti)."""
        files: list[str] = []
        for idx, file_path in enumerate(Path(settings.index_root).rglob("*")):
            if file_path.is_file():
                files.append(file_path.relative_to(settings.index_root).as_posix())
            if idx + 1 >= limit:
                break
        return files

from __future__ import annotations

from fastmcp import FastMCP

from .config import get_settings
from .tools.rag_tools import register_rag_tools
from .vectorstore import VectorStore


def build_server() -> FastMCP:
    settings = get_settings()
    store = VectorStore(settings)
    mcp = FastMCP("FastMCP RAG Demo")

    @mcp.resource("config://status")
    def status() -> dict[str, str]:
        return {
            "repo": "fastmcp",
            "index_root": str(settings.index_root),
            "vector_db": str(settings.vector_db_path),
        }

    register_rag_tools(mcp, store, settings)
    return mcp


def main() -> None:
    server = build_server()
    server.run(transport="http", host="127.0.0.1", port=8000, path="/mcp")


if __name__ == "__main__":
    main()

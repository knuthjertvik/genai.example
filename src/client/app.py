from __future__ import annotations

import asyncio
import os
from typing import Any

import streamlit as st
from fastmcp import Client

SERVER_URL = os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8000/mcp")


def _extract_text(result: Any) -> str:
    parts = []
    for item in getattr(result, "content", []) or []:
        text = getattr(item, "text", None)
        if text:
            parts.append(text)
    if not parts and result is not None:
        parts.append(str(result))
    return "\n".join(parts)


def call_tool(tool: str, params: dict[str, Any]) -> Any:
    async def _call() -> Any:
        async with Client(SERVER_URL) as client:
            return await client.call_tool(tool, params)

    return asyncio.run(_call())


def main() -> None:
    st.set_page_config(page_title="FastMCP RAG klient", page_icon="🔌", layout="wide")
    st.title("FastMCP RAG klient")
    st.caption("Snakk med FastMCP-koden via MCP-serveren")

    question = st.text_area("Spørsmål til fastmcp-koden", placeholder="Hvordan registreres et verktøy i FastMCP?", height=120)
    col1, col2 = st.columns(2)
    with col1:
        top_k = st.slider("Antall kontekstsvar", min_value=1, max_value=8, value=4)
    with col2:
        if st.button("Spørr RAG", type="primary", disabled=not question):
            with st.spinner("Kaller MCP-server..."):
                result = call_tool("ask_fastmcp", {"question": question, "top_k": top_k})
                st.subheader("Svar")
                st.write(_extract_text(result))

    st.divider()
    st.subheader("Semantic søk i fastmcp")
    query = st.text_input("Søkestreng", placeholder="auth provider", value="")
    if st.button("Søk", disabled=not query):
        with st.spinner("Søker..."):
            hits = call_tool("search_fastmcp", {"query": query, "top_k": 5})
            if hasattr(hits, "content"):
                payload = getattr(hits, "content", hits)
            else:
                payload = hits
            st.json(payload)

    st.sidebar.header("Oppsett")
    st.sidebar.write(f"MCP endpoint: {SERVER_URL}")
    st.sidebar.write("Kjør serveren med `uv run rag-server`")


if __name__ == "__main__":
    main()

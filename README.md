# GenAI Workshop 🚀

En hands-on workshop som dekker de viktigste GenAI-trendene:

| År | Teknologi | Beskrivelse |
|----|-----------|-------------|
| 2024 | **RAG** | Retrieval Augmented Generation - snakk med egne dokumenter |
| 2025 | **MCP** | Model Context Protocol - koble AI til verktøy og API-er |

## Kom i gang

1. **Klon repoet** og åpne `workshop_ai_trends.ipynb` i VS Code
2. **Hent `.env`-fil** med API-nøkler (instruksjoner i notebooken)
3. **Opprett virtuelt miljø:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # macOS/Linux
   ```
4. **Installer pakker:**
   ```bash
   pip install openai chromadb tiktoken python-dotenv httpx fastmcp rich pymupdf langchain-text-splitters langchain-mcp-adapters langchain-openai langgraph
   ```
5. **Velg kernel** i VS Code: `.venv`
6. **Kjør cellene** i notebooken!

## Innhold

- 📄 `workshop_ai_trends.ipynb` - Selve workshoppen
- 📁 `data/` - Eksempel-PDF for RAG-delen
- 🔑 `.env.example` - Mal for miljøvariabler
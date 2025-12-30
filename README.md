# FastMCP RAG-demo

Et enkelt, men modulært eksempel som viser hvordan du kan bruke FastMCP som server og Streamlit som klient for å gjøre RAG over `fastmcp`-repoet. Prosjektet bruker `uv` for avhengigheter, Azure OpenAI for embeddings/LLM, lokal ChromaDB for vektorlagring, og GitHub API for å hente ferskt kildekodeinnhold.

## Arkitektur
- MCP-server (`ragserver`) med verktøy for søk og Q&A over indeksen, lese filer lokalt og via GitHub API.
- Indexer (`rag-index`) som kloner/oppdaterer `fastmcp`-repoet og bygger embeddings i ChromaDB.
- Streamlit-klient (`src/client/app.py`) som kaller MCP-serveren.
- Konfig via `.env` (se `.env.example`).

## Kom i gang
1. Installer uv om du ikke har det: `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. Sync avhengigheter: `uv sync`
3. Kopier `.env.example` til `.env` og fyll inn Azure OpenAI-verdier (endpoint, key, modeller) og ev. `GITHUB_TOKEN`.
4. Bygg indeks:
   ```bash
   uv run rag-index
   ```
5. Start MCP-serveren (HTTP-transport):
   ```bash
   uv run rag-server
   ```
6. Start Streamlit-klienten (i et nytt vindu):
   ```bash
   uv run streamlit run src/client/app.py
   ```

## MCP-verktøy
- `search_fastmcp(query, top_k)`: semantisk søk i lokal indeks.
- `ask_fastmcp(question, top_k)`: henter kontekst og svarer med Azure OpenAI.
- `get_file_from_disk(path)`: les lokal fil fra klonen.
- `get_file_from_github(path, ref)`: hent fil via GitHub API.
- `list_local_files(limit)`: enkel liste for å oppdage stier.

## Mappestruktur
- `src/ragserver/` – server, verktøy og indeksering
- `src/client/` – Streamlit-klient
- `data/chroma` – vedvarende ChromaDB (opprettes automatisk)
- `.cache/fastmcp` – lokal klone av `fastmcp`

## Tips
- Juster chunking i `src/ragserver/indexing/chunker.py` om du vil ha større/mindre biter.
- Endre transport/host i `ragserver/server.py` om du vil kjøre stdio/SSE.
- Legg til nye verktøy i `src/ragserver/tools/` og registrer dem i `build_server()`.

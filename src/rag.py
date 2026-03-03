"""
RAG (Retrieval Augmented Generation) system for planning case documents.
"""

import os
from pathlib import Path

from langchain_core.prompts import PromptTemplate
from langchain_classic.retrievers import EnsembleRetriever, MultiQueryRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_chroma import Chroma


# Maps subfolder names (doc_type metadata) to human-readable labels shown to
# both the LLM (in context) and the user (in the sources panel).
_DOC_TYPE_LABELS: dict[str, str] = {
    "plandok": "Plandokument",
    "merknader": "Merknad",
    "plandokument": "Plandokument",  # root-level fallback
}

# --- Norwegian prompts ---

_MULTI_QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""Du hjelper med å søke i plansaksdokumenter fra norske kommuner.
Lag 3 alternative formuleringer av spørsmålet nedenfor for å dekke ulik terminologi og vinklinger.
Skriv ett spørsmål per linje, ingen nummerering eller punkter.

Spørsmål: {question}""",
)

_SYSTEM_PROMPT = """Du er en erfaren norsk jurist og sivilingeniør med ekspertise innen \
plan- og bygningsrett og teknisk prosjektvurdering.

Din oppgave er å hjelpe innbyggere med å forstå og analysere plansaksdokumenter – \
med et kritisk og analytisk blikk.

Du skal:
- Gi faktabaserte svar basert KUN på informasjon i de oppgitte saksdokumentene
- Aktivt lete etter svakheter i metodebruk, kunnskapsgrunnlag og konsekvensutredninger
- Identifisere temaer som ikke er tilstrekkelig belyst eller mangler i saksunderlaget
- Peke på mulige prosessuelle feil i planprosessen (jf. plan- og bygningsloven)
- Presentere «djevelens advokat»-perspektiv: steelman de sterkeste motargumentene
- Bruke nøkternt, faktabasert språk som overbeviser kommunale saksbehandlere og politikere
- Alltid referere til hvilke dokumenter du henter informasjon fra
- Svare på norsk (bokmål)
- Svare kortfattet og presist – maks 3–4 avsnitt med mindre spørsmålet krever mer

Hvis informasjonen ikke finnes i dokumentene, si tydelig fra og unngå å spekulere.

Kontekst fra saksdokumentene:
{context}"""

_QA_TEMPLATE = """Svar på følgende spørsmål basert på saksdokumentene:

{question}

Gi et kortfattet og presist svar. Der det er relevant: pek på svakheter, mangler eller \
uklarheter i saksunderlaget knyttet til spørsmålet. Oppgi hvilke dokumenter du refererer til."""

_CONFLICT_TEMPLATE = """Gjennomfør en kritisk faglig analyse av saksdokumentene \
knyttet til temaet: **{topic}**

Strukturer analysen slik:

1. **Hva planforslaget sier** – faktabasert sammendrag av relevante deler av saksunderlaget

2. **Metodiske svakheter** – svakheter i kunnskapsgrunnlag, beregningsmetoder, \
forutsetninger eller datagrunnlag som er brukt

3. **Manglende utredning** – temaer, scenarier eller konsekvenser som burde vært belyst \
men ikke er tilstrekkelig dekket

4. **Prosessuelle forhold** – mulige feil eller mangler i planprosessen \
(kunngjøring, medvirkning, utredningsplikt, jf. plan- og bygningsloven)

5. **Steelman-motargumenter** – de sterkeste faglige og juridiske argumentene \
mot planforslaget på dette temaet, formulert slik at de kan brukes i et høringsinnspill

6. **Hvem som berøres** – konkrete parter og interesser som rammes

Bruk presist fagspråk. Referer til konkrete dokumenter og sidetall der det er mulig. \
Unngå å spekulere utover det som kan underbygges av saksunderlaget."""

_HEARING_TEMPLATE = """Du er ekspert på kommunal planprosess i Norge. \
Hjelp med å formulere et strukturert høringsinnspill av typen: **{hearing_type}**

Brukerens bekymring eller merknad:
{concern}

{reference_block}

Skriv et høringsinnspill med følgende struktur:
1. **Innledning** – hvem som sender innspillet og tilknytning til saken
2. **Saksforhold** – konkret beskrivelse av bekymringen med faglig begrunnelse
3. **Krav eller forslag** – konkrete endringer eller avklaringer som ønskes
4. **Avslutning** – oppfordring til å ta hensyn og kontaktinfo-felt

Hold en saklig og respektfull tone. Skriv i første person som om du er innbyggeren. \
La [NAVN], [ADRESSE] og lignende være som plassholdere slik at brukeren kan fylle dem inn."""


class RAGSystem:
    def __init__(self, case_key: str):
        from src.config import CASES
        self.case_key = case_key
        self.case_config = CASES[case_key]
        self._embeddings = self._init_embeddings()
        self._llm = self._init_llm()
        self._vectorstore = self._load_vectorstore()
        self._retriever = self._build_retriever()

    def _init_embeddings(self) -> AzureOpenAIEmbeddings:
        return AzureOpenAIEmbeddings(
            azure_deployment=os.environ["AZURE_OPENAI_EMBED_MODEL"],
            azure_endpoint=os.environ["AZURE_OPENAI_EMBEDDING_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_EMBEDDING_API_KEY"],
            api_version=os.environ["AZURE_OPENAI_EMBEDDING_API_VERSION"],
        )

    def _init_llm(self) -> AzureChatOpenAI:
        return AzureChatOpenAI(
            azure_deployment=os.environ["AZURE_OPENAI_CHAT_MODEL"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            max_tokens=1024,
        )

    def _load_vectorstore(self) -> Chroma:
        return Chroma(
            persist_directory=self.case_config["vector_store_dir"],
            embedding_function=self._embeddings,
            collection_name=self.case_config["collection_name"],
        )

    def _build_retriever(self):
        # Load all stored documents from ChromaDB to back the BM25 index.
        stored = self._vectorstore.get()
        all_docs = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(stored["documents"], stored["metadatas"])
        ]

        # BM25 — exact keyword matching, great for §-references and specific numbers.
        bm25 = BM25Retriever.from_documents(all_docs, k=10)

        # Vector — semantic similarity via embeddings.
        vector = self._vectorstore.as_retriever(search_kwargs={"k": 10})

        # Hybrid ensemble with Reciprocal Rank Fusion (equal weights).
        ensemble = EnsembleRetriever(
            retrievers=[bm25, vector],
            weights=[0.5, 0.5],
        )

        # Multi-query: generate 3 Norwegian query variants to improve recall.
        return MultiQueryRetriever.from_llm(
            retriever=ensemble,
            llm=self._llm,
            prompt=_MULTI_QUERY_PROMPT,
        )

    def _retrieve(self, query: str, k: int = 5) -> list:
        docs = self._retriever.invoke(query)
        # Deduplicate — MultiQueryRetriever often returns the same chunk multiple times.
        seen, result = set(), []
        for doc in docs:
            key = doc.page_content[:80]
            if key not in seen:
                seen.add(key)
                result.append(doc)
            if len(result) >= k:
                break
        return result

    def _format_context(self, docs: list) -> tuple[str, list[dict]]:
        """Returns (context_string, sources).

        Each source is a dict:
            {"label": str, "path": str, "page": int}
        so that callers can render both a display tag and a download button.
        """
        parts, sources, seen_labels = [], [], set()
        for doc in docs:
            source = doc.metadata.get("source", "")
            page = doc.metadata.get("page", "")
            doc_type = doc.metadata.get("doc_type", "")
            type_label = _DOC_TYPE_LABELS.get(doc_type, doc_type.capitalize() if doc_type else "")

            filename = Path(source).name if source else "Ukjent dokument"
            page_str = f", side {page + 1}" if page != "" else ""
            label = f"[{type_label}] {filename}{page_str}" if type_label else f"{filename}{page_str}"

            parts.append(f"[Fra: {label}]\n{doc.page_content}")
            if label not in seen_labels:
                seen_labels.add(label)
                sources.append({
                    "label": label,
                    "path": source,
                    "page": page if page != "" else 0,
                })
        return "\n\n---\n\n".join(parts), sources

    def query(self, question: str, chat_history: list | None = None) -> dict:
        docs = self._retrieve(question)
        context, sources = self._format_context(docs)

        history_text = ""
        if chat_history:
            recent = chat_history[-6:]  # last 3 turns
            lines = []
            for msg in recent:
                role = "Bruker" if msg["role"] == "user" else "Assistent"
                lines.append(f"{role}: {msg['content'][:300]}")
            history_text = "\nTidligere i samtalen:\n" + "\n".join(lines) + "\n\n"

        messages = [
            ("system", _SYSTEM_PROMPT.format(context=context)),
            ("human", history_text + _QA_TEMPLATE.format(question=question)),
        ]
        response = self._llm.invoke(messages)
        return {"answer": response.content, "sources": sources}

    def stream_query(self, question: str, chat_history: list | None = None) -> tuple:
        """Returns (generator_of_text_chunks, sources) for streaming responses."""
        docs = self._retrieve(question)
        context, sources = self._format_context(docs)

        history_text = ""
        if chat_history:
            recent = chat_history[-6:]
            lines = []
            for msg in recent:
                role = "Bruker" if msg["role"] == "user" else "Assistent"
                lines.append(f"{role}: {msg['content'][:300]}")
            history_text = "\nTidligere i samtalen:\n" + "\n".join(lines) + "\n\n"

        messages = [
            ("system", _SYSTEM_PROMPT.format(context=context)),
            ("human", history_text + _QA_TEMPLATE.format(question=question)),
        ]

        def _gen():
            for chunk in self._llm.stream(messages):
                if chunk.content:
                    yield chunk.content

        return _gen(), sources

    def analyze_conflicts(self, topic: str) -> dict:
        docs = self._retrieve(topic, k=8)
        context, sources = self._format_context(docs)
        messages = [
            ("system", _SYSTEM_PROMPT.format(context=context)),
            ("human", _CONFLICT_TEMPLATE.format(topic=topic)),
        ]
        response = self._llm.invoke(messages)
        return {"answer": response.content, "sources": sources}

    def generate_hearing_response(
        self, concern: str, hearing_type: str, include_references: bool
    ) -> dict:
        docs = self._retrieve(concern, k=6)
        context, sources = self._format_context(docs)

        if include_references:
            reference_block = (
                "Bruk gjerne relevante referanser fra saksdokumentene nedenfor:\n\n"
                + context
            )
        else:
            reference_block = ""
            sources = []

        messages = [
            (
                "system",
                "Du er ekspert på kommunal planprosess i Norge. Svar på norsk (bokmål).",
            ),
            (
                "human",
                _HEARING_TEMPLATE.format(
                    hearing_type=hearing_type,
                    concern=concern,
                    reference_block=reference_block,
                ),
            ),
        ]
        response = self._llm.invoke(messages)
        return {"answer": response.content, "sources": sources}


def vector_store_exists(case_key: str) -> bool:
    from src.config import CASES
    path = Path(CASES[case_key]["vector_store_dir"]) / "chroma.sqlite3"
    return path.exists()

"""
RAG (Retrieval Augmented Generation) system for planning case documents.
"""

import os
from pathlib import Path

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_chroma import Chroma


# --- Norwegian prompts ---

_SYSTEM_PROMPT = """Du er en nøytral og kritisk assistent som hjelper innbyggere med å forstå \
og analysere plansaksdokumenter i Oslo kommune.

Du skal:
- Gi faktabaserte svar basert KUN på informasjon i de oppgitte saksdokumentene
- Peke på potensielle problemstillinger og konflikter objektivt
- Hjelpe brukere å formulere presise spørsmål og innspill
- Alltid referere til hvilke dokumenter du henter informasjon fra
- Svare på norsk (bokmål)

Hvis informasjonen ikke finnes i dokumentene, si tydelig at dette ikke er dokumentert \
i saksmaterialet og unngå å spekulere.

Kontekst fra saksdokumentene:
{context}"""

_QA_TEMPLATE = """Svar på følgende spørsmål basert på saksdokumentene:

{question}

Gi et presist og informativt svar. Oppgi alltid hvilke dokumenter du refererer til."""

_CONFLICT_TEMPLATE = """Analyser saksdokumentene for potensielle konflikter og \
problemstillinger knyttet til: **{topic}**

Strukturer svaret slik:
1. **Hva planforslaget sier** om dette temaet
2. **Potensielle konflikter** eller problemstillinger
3. **Hvem som kan berøres**
4. **Relevante hensyn** som bør adresseres i høringsinnspill

Vær objektiv og faktabasert. Referer til konkrete dokumenter der det er mulig."""

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
            temperature=0.1,
            max_tokens=2048,
        )

    def _load_vectorstore(self) -> Chroma:
        return Chroma(
            persist_directory=self.case_config["vector_store_dir"],
            embedding_function=self._embeddings,
            collection_name=self.case_config["collection_name"],
        )

    def _retrieve(self, query: str, k: int = 5) -> list:
        return self._vectorstore.similarity_search(query, k=k)

    def _format_context(self, docs: list) -> tuple[str, list[str]]:
        parts, sources = [], []
        for doc in docs:
            source = doc.metadata.get("source", "Ukjent dokument")
            page = doc.metadata.get("page", "")
            label = Path(source).name + (f", side {page + 1}" if page != "" else "")
            parts.append(f"[Fra: {label}]\n{doc.page_content}")
            if label not in sources:
                sources.append(label)
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

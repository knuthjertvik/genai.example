"""
Plansaksanalyse – webapp for å stille spørsmål til plansaksdokumenter.

Bruk:
    streamlit run app.py
"""

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.config import CASES, CONFLICT_TOPICS
from src.rag import RAGSystem, vector_store_exists

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Plansaksanalyse",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; }
    .source-tag {
        display: inline-block;
        background: #e8edf2;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.8rem;
        color: #4a5568;
        margin: 2px;
    }
    .warning-box {
        background: #fff8e1;
        border-left: 4px solid #f9a825;
        padding: 12px 16px;
        border-radius: 4px;
        margin-bottom: 1rem;
    }
    .hearing-output {
        background: #f0fff4;
        border-left: 4px solid #38a169;
        padding: 16px;
        border-radius: 4px;
        white-space: pre-wrap;
        font-family: inherit;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🏗️ Plansaksanalyse")
    st.caption("Forstå og analyser plansaksdokumenter fra Oslo kommune")
    st.divider()

    # Find cases with a built index
    ready_cases = {k: v for k, v in CASES.items() if vector_store_exists(k)}
    all_cases = list(CASES.keys())

    if not ready_cases:
        st.error("Ingen saker er indeksert ennå.")
        st.info(
            "Legg PDF-filer i `data/sinsenveien_11/` og kjør:\n\n"
            "```\npython scripts/build_index.py\n```"
        )
        st.stop()

    selected_case = st.selectbox(
        "📁 Velg sak",
        options=list(ready_cases.keys()),
        format_func=lambda k: CASES[k]["name"],
    )
    case_cfg = CASES[selected_case]

    st.caption(case_cfg.get("description", ""))
    st.divider()

    st.link_button(
        "📮 Send inn høringsinnspill",
        case_cfg["official_url"],
        use_container_width=True,
        type="primary",
    )

    st.divider()
    st.caption(
        f"Saksnr: {case_cfg.get('case_number', '–')}  \n"
        f"Kommune: {case_cfg.get('municipality', '–')}"
    )
    st.caption("Drevet av Azure OpenAI + LangChain")


# ---------------------------------------------------------------------------
# Load RAG system (cached per case)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Laster dokumentindeks...")
def get_rag(case_key: str) -> RAGSystem:
    return RAGSystem(case_key)


try:
    rag = get_rag(selected_case)
except Exception as exc:
    st.error(f"Kunne ikke laste dokumentindeks: {exc}")
    st.info("Sjekk at miljøvariablene i `.env` er korrekte og at indeksen er bygget.")
    st.stop()


# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.title(f"📋 {case_cfg['name']}")

tab_qa, tab_conflicts, tab_hearing, tab_about = st.tabs(
    [
        "💬 Still spørsmål",
        "⚠️ Konflikter",
        "✏️ Høringsinnspill",
        "📎 Om saken",
    ]
)


# ===========================================================================
# TAB 1 – Q&A CHAT
# ===========================================================================
with tab_qa:
    st.subheader("Still spørsmål til saksdokumentene")
    st.caption(
        "Spør om hva som helst i planforslaget. "
        "Svarene er basert direkte på de opplastede saksdokumentene."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Render existing chat
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📄 Kilder", expanded=False):
                    for src in msg["sources"]:
                        st.markdown(f'<span class="source-tag">{src}</span>', unsafe_allow_html=True)

    # Chat input
    if prompt := st.chat_input("Hva vil du vite om planforslaget?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Søker i dokumentene..."):
                result = rag.query(prompt, chat_history=st.session_state.messages[:-1])
            st.markdown(result["answer"])
            if result["sources"]:
                with st.expander("📄 Kilder", expanded=False):
                    for src in result["sources"]:
                        st.markdown(f'<span class="source-tag">{src}</span>', unsafe_allow_html=True)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"],
            }
        )

    if st.session_state.messages:
        if st.button("🗑️ Nullstill samtalen", key="clear_chat"):
            st.session_state.messages = []
            st.rerun()

    # Starter questions
    if not st.session_state.messages:
        st.markdown("**Forslag til spørsmål:**")
        starter_questions = [
            "Hva er de viktigste endringene i planforslaget?",
            "Hvilken byggehøyde er foreslått?",
            "Hva sier planforslaget om parkering?",
            "Hvordan påvirkes naboeiendommene?",
            "Hva er høringsfristen?",
        ]
        cols = st.columns(len(starter_questions))
        for col, question in zip(cols, starter_questions):
            with col:
                if st.button(question, key=f"starter_{question}", use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": question})
                    with st.spinner("Søker..."):
                        result = rag.query(question)
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": result["answer"],
                            "sources": result["sources"],
                        }
                    )
                    st.rerun()


# ===========================================================================
# TAB 2 – CONFLICT ANALYSIS
# ===========================================================================
with tab_conflicts:
    st.subheader("Konflikter og problemstillinger")
    st.caption(
        "Velg et tema for å se en AI-drevet analyse av potensielle konflikter "
        "basert på saksdokumentene."
    )

    topic = st.selectbox("Velg tema", CONFLICT_TOPICS, key="conflict_topic")

    if st.button("🔍 Analyser tema", type="primary", key="analyze_btn"):
        with st.spinner(f"Analyserer «{topic}»..."):
            result = rag.analyze_conflicts(topic)

        st.markdown(
            f'<div class="warning-box">{result["answer"]}</div>',
            unsafe_allow_html=True,
        )

        if result["sources"]:
            with st.expander("📄 Kildereferanser"):
                for src in result["sources"]:
                    st.markdown(f'<span class="source-tag">{src}</span>', unsafe_allow_html=True)

    st.divider()
    st.info(
        "💡 **Tips:** Analysen viser hva dokumentene sier om temaet og peker på "
        "mulige konfliktpunkter. Bruk fanen «Høringsinnspill» for å formulere "
        "dine egne merknader."
    )


# ===========================================================================
# TAB 3 – HEARING RESPONSE GENERATOR
# ===========================================================================
with tab_hearing:
    st.subheader("Skriv høringsinnspill")
    st.caption(
        "Beskriv din bekymring eller merknad, så hjelper AI deg med å formulere "
        "et strukturert høringsinnspill."
    )

    col_input, col_output = st.columns([1, 1], gap="large")

    with col_input:
        concern = st.text_area(
            "Din bekymring eller merknad",
            height=160,
            placeholder=(
                "F.eks: Jeg er bekymret for at det nye bygget vil kaste skygge på "
                "hagen og terrassen min store deler av dagen, og at dette ikke er "
                "tilstrekkelig utredet i planforslaget..."
            ),
            key="hearing_concern",
        )

        hearing_type = st.selectbox(
            "Type innspill",
            [
                "Protest / innsigelse mot planforslaget",
                "Forslag til endringer i planforslaget",
                "Spørsmål og behov for avklaring",
                "Støtte til planforslaget med merknader",
            ],
            key="hearing_type",
        )

        include_refs = st.checkbox(
            "Inkluder referanser til saksdokumentene",
            value=True,
            key="hearing_refs",
        )

        generate_btn = st.button(
            "✍️ Generer høringsinnspill",
            type="primary",
            disabled=not concern.strip(),
            key="generate_hearing",
        )

    with col_output:
        if generate_btn and concern.strip():
            with st.spinner("Formulerer høringsinnspill..."):
                result = rag.generate_hearing_response(
                    concern=concern,
                    hearing_type=hearing_type,
                    include_references=include_refs,
                )

            st.subheader("Ditt utkast")
            st.markdown(
                f'<div class="hearing-output">{result["answer"]}</div>',
                unsafe_allow_html=True,
            )

            st.download_button(
                "⬇️ Last ned som tekstfil",
                data=result["answer"],
                file_name=f"horingsinnspill_{selected_case}.txt",
                mime="text/plain",
                key="download_hearing",
            )

            st.markdown(
                '<div class="warning-box">'
                "⚠️ <strong>Husk:</strong> Tilpass utkastet til din egen situasjon. "
                "Fyll inn [NAVN], [ADRESSE] og andre plassholdere. "
                "Signer med fullt navn og adresse."
                "</div>",
                unsafe_allow_html=True,
            )

            st.link_button(
                "📮 Send inn via Oslo kommunes portal",
                case_cfg["official_url"],
                use_container_width=True,
                type="primary",
            )

            if result["sources"]:
                with st.expander("📄 Dokumenter brukt som grunnlag"):
                    for src in result["sources"]:
                        st.markdown(f'<span class="source-tag">{src}</span>', unsafe_allow_html=True)
        else:
            st.info("👈 Fyll inn din bekymring til venstre og klikk «Generer høringsinnspill».")


# ===========================================================================
# TAB 4 – ABOUT THE CASE
# ===========================================================================
with tab_about:
    st.subheader(f"Om {case_cfg['name']}")

    col_info, col_links = st.columns([2, 1])

    with col_info:
        st.markdown(
            f"""
### Planforslaget
{case_cfg.get("description", "")}

### Hva er en reguleringsplan?
En reguleringsplan fastsetter arealbruk og vilkår for bruk og vern av arealer,
bebyggelse og anlegg i kommunen. Reguleringsplaner er rettslig bindende for
fremtidig bruk av berørte arealer.

### Din rett til å uttale deg
Alle som er berørt av planforslaget har rett til å sende inn høringsinnspill.
Fristen fremgår av kommunens kunngjøring. Innspill som er sendt innen fristen
skal behandles av kommunen og besvares i det videre planarbeidet.

### Slik bruker du dette verktøyet
1. **💬 Still spørsmål** – Skriv spørsmål i chat-feltet og få svar basert på dokumentene
2. **⚠️ Konflikter** – Velg et tema for automatisk analyse av mulige konflikter
3. **✏️ Høringsinnspill** – Beskriv din bekymring og få hjelp til å formulere innspillet
4. **📮 Send inn** – Bruk lenken til Oslo kommunes portal for å sende inn innspillet ditt

### Om verktøyet
Dette er et åpent og kritisk analyseverktøy for innbyggere. Svarene er basert
på saksdokumentene som er lastet opp. Verktøyet er ikke tilknyttet Oslo kommune
og representerer ikke myndighetenes syn.
"""
        )

    with col_links:
        st.info(
            f"**Saksnummer:** {case_cfg.get('case_number', '–')}  \n"
            f"**Type:** Reguleringsplan  \n"
            f"**Kommune:** {case_cfg.get('municipality', '–')}  \n"
            f"**Status:** Til offentlig ettersyn"
        )

        st.link_button(
            "📮 Gå til kommunens portal",
            case_cfg["official_url"],
            use_container_width=True,
            type="primary",
        )

        st.link_button(
            "🔗 Oslo kommunes plankart",
            "https://od2.pbe.oslo.kommune.no/kart/",
            use_container_width=True,
        )

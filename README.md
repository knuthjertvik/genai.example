# GenAI Workshop + Plansaksanalyse

Dette repoet inneholder to ting:

| Del | Beskrivelse |
|-----|-------------|
| **Workshop** | Hands-on notebook om RAG og MCP med Azure OpenAI |
| **Plansaksanalyse** | Webapp for å stille spørsmål til plansaksdokumenter |

---

## Plansaksanalyse – webapp

En Streamlit-app der innbyggere kan analysere plansaksdokumenter, finne
konflikter og få hjelp til å skrive høringsinnspill.

### Funksjoner
- 💬 **Spørsmål & svar** – still spørsmål og få svar med kildehenvisning
- ⚠️ **Konflikter** – automatisk analyse av potensielle problemstillinger
- ✏️ **Høringsinnspill** – AI-hjulpet formulering av innspill
- 📮 **Innsending** – direktelenke til kommunens høringsportal

### Mappestruktur

```
app.py                        ← Streamlit-appen (kjørbar)
src/
  config.py                   ← Konfigurasjon per plansak
  rag.py                      ← RAG-logikk (søk + LLM)
scripts/
  build_index.py              ← Indekseringsscript (kjøres lokalt)
data/
  sinsenveien_11/             ← Legg PDF-filene her (gitignorert)
vector_store/
  sinsenveien_11/             ← Ferdigbygd indeks (committes til git)
```

### Oppsett (lokal utvikling)

```bash
# 1. Klon og opprett virtuelt miljø
git clone <repo-url>
cd genai.example
python3 -m venv .venv
source .venv/bin/activate   # macOS/Linux

# 2. Installer avhengigheter
pip install -r requirements.txt

# 3. Kopier og fyll inn API-nøkler
cp .env.example .env
# Rediger .env med dine Azure OpenAI-verdier

# 4. Legg PDF-filer i data/sinsenveien_11/
#    (disse er gitignorert og lagres kun lokalt)

# 5. Bygg vektordatabasen (kjøres én gang, eller ved oppdatering av dokumenter)
python scripts/build_index.py --case sinsenveien_11

# 6. Start appen
streamlit run app.py
```

### Deploy til Streamlit Cloud (gratis)

1. **Push indeksen til git** etter at du har kjørt `build_index.py`:
   ```bash
   git add vector_store/
   git commit -m "Legg til vektordatabase for Sinsenveien 11"
   git push
   ```

2. **Opprett konto** på [share.streamlit.io](https://share.streamlit.io) (gratis med GitHub-login)

3. **Deploy appen:**
   - Klikk «New app»
   - Velg dette repoet og `app.py` som hovedfil
   - Under «Advanced settings → Secrets», legg inn innholdet fra `.env`:
     ```
     AZURE_OPENAI_CHAT_MODEL = "..."
     AZURE_OPENAI_ENDPOINT = "..."
     AZURE_OPENAI_API_KEY = "..."
     AZURE_OPENAI_API_VERSION = "..."
     AZURE_OPENAI_EMBED_MODEL = "..."
     AZURE_OPENAI_EMBEDDING_API_KEY = "..."
     AZURE_OPENAI_EMBEDDING_ENDPOINT = "..."
     AZURE_OPENAI_EMBEDDING_API_VERSION = "..."
     ```
   - Klikk «Deploy»

4. Del URL-en (f.eks. `https://dittapp.streamlit.app`) med vennene dine.

### Legge til ny plansak

1. Legg til ny konfigurasjon i `src/config.py` under `CASES`
2. Opprett mappe `data/<saksnøkkel>/` og legg inn PDF-filer
3. Kjør `python scripts/build_index.py --case <saksnøkkel>`
4. Commit `vector_store/<saksnøkkel>/` og push

---

## Workshop – Jupyter Notebook

| Teknologi | Beskrivelse |
|-----------|-------------|
| **RAG** | Retrieval Augmented Generation – snakk med egne dokumenter |
| **MCP** | Model Context Protocol – koble AI til verktøy og API-er |

### Kom i gang med workshopen

1. Åpne `workshop_ai_trends.ipynb` i VS Code
2. Opprett virtuelt miljø og installer avhengigheter (se over)
3. Velg `.venv` som kernel og kjør cellene

### Filer
- 📄 `workshop_ai_trends.ipynb` – selve workshoppen
- 📁 `data/` – eksempel-PDFer for RAG-delen
- 🔑 `.env.example` – mal for miljøvariabler

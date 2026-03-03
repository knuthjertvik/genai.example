#!/usr/bin/env python3
"""
Builds the ChromaDB vector index from PDF documents.

Run this locally after placing PDF files in the correct data folder.
The resulting vector_store/ folder should be committed to git so that
Streamlit Cloud can serve the app without re-processing documents.

Usage:
    python scripts/build_index.py --case sinsenveien_11
    python scripts/build_index.py          # builds all configured cases
"""

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

# Allow importing from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from rich.console import Console
from rich.progress import track

load_dotenv()
console = Console()


def build_index(case_key: str) -> None:
    from src.config import CASES
    from langchain_openai import AzureOpenAIEmbeddings
    from langchain_chroma import Chroma
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    import pymupdf4llm

    case = CASES[case_key]
    data_dir = Path(case["data_dir"])
    vector_store_dir = Path(case["vector_store_dir"])

    console.rule(f"[bold blue]{case['name']}")

    if not data_dir.exists():
        console.print(f"[red]Mappe ikke funnet: {data_dir}[/red]")
        console.print("[yellow]Opprett mappen og legg til PDF-filer:[/yellow]")
        console.print(f"  mkdir -p {data_dir}")
        console.print(f"  # Kopier PDF-filene til {data_dir}/")
        return

    pdf_files = sorted(data_dir.glob("**/*.pdf"))
    if not pdf_files:
        console.print(f"[red]Ingen PDF-filer funnet i {data_dir}[/red]")
        return

    console.print(f"[green]Fant {len(pdf_files)} PDF-filer[/green]")

    # Parse PDFs and split into chunks.
    # 800 chars (~200 tokens) fits one regulatory paragraph tightly, giving
    # focused embeddings that rank precisely against specific fact questions
    # (§-references, building heights, parking counts, etc.).
    # pymupdf4llm produces clean \n\n paragraph breaks so the splitter respects
    # natural boundaries. Rate-limit hits are handled by the retry loop below.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )

    all_chunks = []
    failed = []

    for pdf_path in track(pdf_files, description="Leser PDF-filer..."):
        try:
            # Derive doc_type from the first subfolder under data_dir.
            # e.g. data/sinsenveien_11/plandok/xx.pdf  → "plandok"
            #      data/sinsenveien_11/merknader/xx.pdf → "merknader"
            #      data/sinsenveien_11/xx.pdf           → "plandokument" (root fallback)
            relative = pdf_path.relative_to(data_dir)
            doc_type = relative.parts[0] if len(relative.parts) > 1 else "plandokument"

            # pymupdf4llm preserves tables as markdown, handles multi-column layout,
            # and produces cleaner text than raw get_text() for regulatory documents.
            pages = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)
            for page_data in pages:
                text = (page_data.get("text") or "").strip()
                page_num = page_data.get("metadata", {}).get("page", 0)
                if text:
                    chunks = splitter.create_documents(
                        [text],
                        metadatas=[
                            {
                                "source": str(pdf_path),
                                "filename": pdf_path.name,
                                "page": page_num,
                                "doc_type": doc_type,
                            }
                        ],
                    )
                    all_chunks.extend(chunks)
        except Exception as exc:
            failed.append((pdf_path.name, str(exc)))

    if failed:
        console.print(f"[yellow]Advarsel: {len(failed)} filer kunne ikke leses:[/yellow]")
        for name, err in failed:
            console.print(f"  • {name}: {err}")

    if not all_chunks:
        console.print("[red]Ingen tekst funnet. Sjekk at PDF-filene inneholder lesbar tekst.[/red]")
        return

    from collections import Counter
    type_counts = Counter(c.metadata["doc_type"] for c in all_chunks)
    console.print(f"[green]Laget {len(all_chunks)} tekstbiter totalt[/green]")
    for dtype, count in sorted(type_counts.items()):
        console.print(f"  [dim]{dtype}: {count} biter[/dim]")

    # Initialize Azure OpenAI embeddings
    console.print("[blue]Kobler til Azure OpenAI embeddings...[/blue]")
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.environ["AZURE_OPENAI_EMBED_MODEL"],
        azure_endpoint=os.environ["AZURE_OPENAI_EMBEDDING_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_EMBEDDING_API_KEY"],
        api_version=os.environ["AZURE_OPENAI_EMBEDDING_API_VERSION"],
    )

    # Remove existing vector store to rebuild fresh
    if vector_store_dir.exists():
        shutil.rmtree(vector_store_dir)
    vector_store_dir.mkdir(parents=True)

    console.print(f"[blue]Bygger vektordatabase i {vector_store_dir} ...[/blue]")
    console.print("[dim](Dette kan ta noen minutter — rate-limit håndteres automatisk)[/dim]")

    from openai import RateLimitError

    BATCH_SIZE = 50  # chunks per embedding call (~50 × 375 tokens = ~18 750 tokens/call)
    MAX_RETRIES = 6
    RETRY_WAIT = 65  # seconds — slightly over the 60 s window Azure requires

    # Create an empty collection first, then add in controlled batches.
    db = Chroma(
        collection_name=case["collection_name"],
        embedding_function=embeddings,
        persist_directory=str(vector_store_dir),
    )

    total_batches = (len(all_chunks) + BATCH_SIZE - 1) // BATCH_SIZE
    indexed = 0

    for batch_num, start in enumerate(range(0, len(all_chunks), BATCH_SIZE), 1):
        batch = all_chunks[start : start + BATCH_SIZE]
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                db.add_documents(batch)
                indexed += len(batch)
                console.print(
                    f"  [green]Batch {batch_num}/{total_batches}[/green] "
                    f"({indexed}/{len(all_chunks)} biter)"
                )
                break
            except RateLimitError:
                if attempt == MAX_RETRIES:
                    console.print(f"[red]Maks forsøk nådd for batch {batch_num}. Avbryter.[/red]")
                    raise
                console.print(
                    f"  [yellow]Rate limit nådd (batch {batch_num}, forsøk {attempt}/{MAX_RETRIES}). "
                    f"Venter {RETRY_WAIT}s...[/yellow]"
                )
                time.sleep(RETRY_WAIT)

    console.print(f"[bold green]✓ Ferdig! {indexed} biter indeksert.[/bold green]")
    console.print(f"[green]Lagret i: {vector_store_dir}[/green]")
    console.print()
    console.print("[yellow]Neste steg – commit og push vector store til git:[/yellow]")
    console.print(f"  git add {vector_store_dir}")
    console.print(f"  git commit -m 'Oppdater vektordatabase for {case_key}'")
    console.print("  git push")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bygg ChromaDB-indeks fra PDF-dokumenter i en plansak"
    )
    parser.add_argument(
        "--case",
        help="Saksnøkkel, f.eks. sinsenveien_11 (utelat for å bygge alle)",
        default=None,
    )
    args = parser.parse_args()

    from src.config import CASES

    cases_to_build = [args.case] if args.case else list(CASES.keys())

    for key in cases_to_build:
        if key not in CASES:
            console.print(f"[red]Ukjent sak: '{key}'[/red]")
            console.print(f"Tilgjengelige saker: {', '.join(CASES.keys())}")
            sys.exit(1)
        build_index(key)


if __name__ == "__main__":
    main()

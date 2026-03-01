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
    import fitz  # PyMuPDF

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

    # Parse PDFs and split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
    )

    all_chunks = []
    failed = []

    for pdf_path in track(pdf_files, description="Leser PDF-filer..."):
        try:
            doc = fitz.open(str(pdf_path))
            for page_num, page in enumerate(doc):
                text = page.get_text().strip()
                if text:
                    chunks = splitter.create_documents(
                        [text],
                        metadatas=[
                            {
                                "source": str(pdf_path),
                                "filename": pdf_path.name,
                                "page": page_num,
                            }
                        ],
                    )
                    all_chunks.extend(chunks)
            doc.close()
        except Exception as exc:
            failed.append((pdf_path.name, str(exc)))

    if failed:
        console.print(f"[yellow]Advarsel: {len(failed)} filer kunne ikke leses:[/yellow]")
        for name, err in failed:
            console.print(f"  • {name}: {err}")

    if not all_chunks:
        console.print("[red]Ingen tekst funnet. Sjekk at PDF-filene inneholder lesbar tekst.[/red]")
        return

    console.print(f"[green]Laget {len(all_chunks)} tekstbiter totalt[/green]")

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
    console.print("[dim](Dette kan ta noen minutter for mange dokumenter)[/dim]")

    Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory=str(vector_store_dir),
        collection_name=case["collection_name"],
    )

    console.print(f"[bold green]✓ Ferdig! {len(all_chunks)} biter indeksert.[/bold green]")
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

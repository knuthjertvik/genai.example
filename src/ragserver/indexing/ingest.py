from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable

from rich import print

from ..config import get_settings
from ..vectorstore import Document, VectorStore
from .chunker import chunk_text, iter_source_files

REPO_URL = "https://github.com/jlowin/fastmcp.git"


def sync_repo(repo_dir: Path) -> None:
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    if repo_dir.exists():
        print(f"[cyan]Updating {repo_dir}...[/cyan]")
        subprocess.run(["git", "-C", str(repo_dir), "pull", "--ff-only"], check=True)
    else:
        print(f"[cyan]Cloning fastmcp into {repo_dir}...[/cyan]")
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(repo_dir)], check=True)


def build_documents(files: Iterable[Path], repo_root: Path) -> list[Document]:
    documents: list[Document] = []
    for file_path in files:
        text = file_path.read_text(encoding="utf-8")
        relative_path = file_path.relative_to(repo_root).as_posix()
        for idx, chunk in enumerate(chunk_text(text)):
            doc_id = f"{relative_path}#chunk{idx}"
            documents.append(
                Document(
                    doc_id=doc_id,
                    text=chunk,
                    metadata={
                        "path": relative_path,
                        "chunk": idx,
                        "repo": "fastmcp",
                    },
                )
            )
    return documents


def ingest() -> None:
    settings = get_settings()
    repo_dir = settings.index_root
    sync_repo(repo_dir)

    files = list(iter_source_files(repo_dir))
    print(f"[green]Indexing {len(files)} files...[/green]")
    docs = build_documents(files, repo_dir)

    store = VectorStore(settings)
    store.add(docs)
    print(f"[green]Stored {len(docs)} chunks in Chroma at {settings.vector_db_path}[/green]")


def main() -> None:
    ingest()


if __name__ == "__main__":
    main()

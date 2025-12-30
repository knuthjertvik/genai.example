from __future__ import annotations

from pathlib import Path
from typing import Iterable

import tiktoken


DEFAULT_MAX_TOKENS = 600
DEFAULT_OVERLAP = 120


def iter_source_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_dir():
            continue
        if path.suffix.lower() in {".py", ".md", ".rst", ".txt", ".toml", ".json", ".yaml", ".yml"}:
            parts = list(path.relative_to(root).parts)
            if any(part.startswith(".") for part in parts):
                continue
            yield path


def chunk_text(text: str, tokenizer_name: str = "cl100k_base", max_tokens: int = DEFAULT_MAX_TOKENS, overlap: int = DEFAULT_OVERLAP) -> list[str]:
    tokenizer = tiktoken.get_encoding(tokenizer_name)
    tokens = tokenizer.encode(text)
    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(tokenizer.decode(chunk_tokens))
        if end == len(tokens):
            break
        start = max(0, end - overlap)
    return chunks

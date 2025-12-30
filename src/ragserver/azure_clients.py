from collections.abc import Sequence
from typing import Any
from urllib.parse import urlparse

from openai import AzureOpenAI

from .config import Settings


def _clean_endpoint(endpoint: str) -> str:
    parsed = urlparse(endpoint)
    if parsed.scheme and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}"
    return endpoint.rstrip("/")


def get_client(settings: Settings, *, for_embedding: bool = False) -> AzureOpenAI:
    raw_endpoint = (
        settings.azure_embed_endpoint
        if for_embedding and settings.azure_embed_endpoint
        else settings.azure_endpoint
    )
    endpoint = _clean_endpoint(raw_endpoint)
    api_key = settings.azure_embed_api_key if for_embedding and settings.azure_embed_api_key else settings.azure_api_key
    api_version = settings.azure_embed_api_version or settings.azure_api_version
    return AzureOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=api_version,
    )


def embed_texts(texts: Sequence[str], settings: Settings) -> list[list[float]]:
    client = get_client(settings, for_embedding=True)
    response = client.embeddings.create(
        input=list(texts),
        model=settings.azure_embed_model,
    )
    return [item.embedding for item in response.data]


def chat_with_context(question: str, contexts: Sequence[str], settings: Settings) -> str:
    client = get_client(settings)
    context_block = "\n\n".join(contexts)
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": "You are a helpful engineer answering questions about the FastMCP codebase. Use the provided context snippets and cite file paths when relevant.",
        },
        {
            "role": "user",
            "content": f"Context:\n{context_block}\n\nQuestion: {question}",
        },
    ]
    completion = client.chat.completions.create(
        model=settings.azure_chat_model,
        messages=messages,
    )
    return completion.choices[0].message.content or ""

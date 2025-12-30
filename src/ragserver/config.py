from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore", env_file=".env", env_file_encoding="utf-8")

    azure_endpoint: str = Field(..., alias="AZURE_OPENAI_ENDPOINT")
    azure_api_key: str = Field(..., alias="AZURE_OPENAI_API_KEY")
    azure_embed_api_key: str | None = Field(default=None, alias="AZURE_OPENAI_EMBEDDING_API_KEY")
    azure_embed_endpoint: str | None = Field(default=None, alias="AZURE_OPENAI_EMBEDDING_ENDPOINT")
    azure_api_version: str = Field("2024-08-01-preview", alias="AZURE_OPENAI_API_VERSION")
    azure_chat_model: str = Field("gpt-4o", alias="AZURE_OPENAI_CHAT_MODEL")
    azure_embed_model: str = Field("text-embedding-3-large", alias="AZURE_OPENAI_EMBED_MODEL")
    azure_embed_api_version: str | None = Field(default=None, alias="AZURE_OPENAI_EMBEDDING_API_VERSION")
    github_token: str | None = Field(default=None, alias="GITHUB_TOKEN")
    index_root: Path = Field(Path(".cache/fastmcp"), alias="INDEX_ROOT")
    vector_db_path: Path = Field(Path("data/chroma"), alias="VECTOR_DB_PATH")


def get_settings() -> Settings:
    return Settings()  # type: ignore[arg-type]

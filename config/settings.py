from pathlib import Path
from functools import lru_cache
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Local LLM via Ollama
    ollama_base_url: str = "http://localhost:11434/v1"

    @field_validator("ollama_base_url", mode="before")
    @classmethod
    def ensure_protocol(cls, v: str) -> str:
        """Ensure the Ollama URL always has an http:// prefix.

        Prevents httpx.UnsupportedProtocol when the env var is set as
        ``localhost:11434/v1`` without a scheme.
        """
        if v and not v.startswith(("http://", "https://")):
            v = f"http://{v}"
        return v
    ollama_model: str = "llama3.2"
    # Optional vision model for Stage 2 answer refinement (e.g. llama3.2-vision)
    ollama_vision_model: str = ""

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "megarag_pages"

    # ColQwen
    colqwen_model: str = "vidore/colqwen2-v1.0"
    colqwen_device: str = "mps"

    # Local paths
    data_dir: Path = Path("data")
    pages_dir: Path = Path("data/pages")
    raw_dir: Path = Path("data/raw")
    kg_db_path: Path = Path("data/db/knowledge_graph.db")

    def ensure_dirs(self) -> None:
        for d in [self.data_dir, self.pages_dir, self.raw_dir, self.kg_db_path.parent]:
            d.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT_DIR / ".env", extra="ignore")

    data_dir: Path = ROOT_DIR / "data"

    # Qdrant: a URL for the server (docker), otherwise embedded local mode on disk.
    qdrant_url: str | None = None
    qdrant_path: Path = ROOT_DIR / "data" / "qdrant"

    embed_model: str = "bge-m3"  # dense embeddings, served by Ollama
    # Query embeddings on the CPU (~35 ms) keep the GPU free for the LLM, so Ollama does not swap models.
    embed_query_on_cpu: bool = True
    ollama_keep_alive: str = "1h"
    sparse_model: str = "Qdrant/bm25"

    ollama_url: str = "http://127.0.0.1:11434"
    llm_model: str = "qwen3.5:4b"
    llm_temperature: float = 0.1
    llm_num_ctx: int = 4096  # 6 passages + prompt + answer fit; keeps the KV cache small

    # Retrieval / guardrails
    chat_top_k: int = 6
    rrf_k: int = 60
    min_dense_score: float = 0.43  # calibrated on eval/qa.jsonl: lowest answerable question scores 0.47
    max_question_chars: int = 1000

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"


@lru_cache
def get_settings() -> Settings:
    return Settings()

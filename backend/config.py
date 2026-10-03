"""Centralized configuration for HalluciGuard."""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"

    # Embeddings
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # Documents
    documents_dir: str = "data/documents"

    # Retrieval
    top_k: int = 5

    # Verification Thresholds
    similarity_threshold: float = 0.70
    high_similarity_threshold: float = 0.85
    low_similarity_threshold: float = 0.40

    # NLI
    nli_model: str = "cross-encoder/nli-deberta-v3-small"
    nli_confidence_threshold: float = 0.80

    # Frontend → Backend
    backend_url: str = "http://localhost:8000"

    # MLflow
    mlflow_tracking_uri: str = "./mlruns"
    mlflow_experiment_name: str = "HalluciGuard"

    @property
    def documents_path(self) -> Path:
        return Path(self.documents_dir)


settings = Settings()

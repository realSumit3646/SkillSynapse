from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=("backend/.env", ".env"), env_file_encoding="utf-8", extra="ignore")

    gemini_api_key: Optional[str] = None
    gemini_chat_model: str = "gemini-3.1-flash-lite-preview"
    gemini_embedding_model: str = "models/text-embedding-004"
    use_gemini_embeddings: bool = False

    sentence_transformer_model: str = "all-MiniLM-L6-v2"
    max_output_skills: int = 20
    cluster_distance_threshold: float = 0.38
    allow_llm_cluster_naming: bool = True


settings = Settings()

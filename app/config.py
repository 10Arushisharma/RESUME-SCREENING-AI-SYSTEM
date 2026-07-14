from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=Path(__file__).resolve().parent.parent / ".env", extra="ignore")

    model_name: str = "all-MiniLM-L6-v2"
    max_upload_size_mb: int = 10
    allowed_extensions: str = ".pdf,.doc,.docx,.txt,.rtf,.md,.jpg,.jpeg,.png,.bmp,.tif,.tiff"
    embedding_dimensions: int = 384
    semantic_weight: float = 0.4
    skills_weight: float = 0.25
    experience_weight: float = 0.2
    education_weight: float = 0.1
    projects_weight: float = 0.05
    certifications_weight: float = 0.0

    @property
    def allowed_extensions_list(self) -> list[str]:
        return [item.strip().lower() for item in self.allowed_extensions.split(",") if item.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()

from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Service Metadata
    PROJECT_NAME: str = "Crucible Diagnostics Engine"
    API_V1_STR: str = "/api/v1"
    ENVIRONMENT: str = "development"

    # Redis Broker & Backend Configuration
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None

    @property
    def REDIS_URL(self) -> str:
        """Constructs the fully qualified Redis connection URI."""
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    # Celery Concurrency & Lifetime Controls
    CELERY_WORKER_CONCURRENCY: int = 4
    CELERY_TASK_RESULT_EXPIRES: int = 3600  # Stored analytical results expire after 1 hour

    # Pydantic Settings Configuration
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )


@lru_cache
def get_settings() -> Settings:
    """
    Returns a cached singleton instance of Settings.
    Prevents repeated disk reads or environment re-parsing across requests.
    """
    return Settings()
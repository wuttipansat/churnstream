from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):

    app_name: str = "ChurnStream"
    app_env: str = "local"
    debug: bool = True

    mongodb_uri: str
    mongodb_database: str = "churnstream"
    mongodb_collection: str = "customers"

    mongodb_server_selection_timeout_ms: int = 5000
    mongodb_connect_timeout_ms: int = 5000
    mongodb_socket_timeout_ms: int = 10000
    mongodb_max_pool_size: int = 20

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

@lru_cache
def get_settings() -> Settings:
    return Settings()



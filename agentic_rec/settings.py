from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AGENTIC_REC_", env_file=".env", extra="ignore"
    )

    # data
    movielens_1m_url: str = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
    data_dir: str = "data"

    # lancedb
    lance_db_path: str = "lance_db"
    items_table_name: str = "items"
    users_table_name: str = "users"
    embedder_name: str = "lightonai/DenseOn"
    reranker_name: str = "lightonai/LateOn"
    reranker_type: str = "pylate"

    # llm
    llm_model: str = "cerebras:llama3.1-8b"

    @property
    def items_parquet(self) -> str:
        """Path to processed items parquet file."""
        return f"{self.data_dir}/ml-1m/items.parquet"

    @property
    def users_parquet(self) -> str:
        """Path to processed users parquet file."""
        return f"{self.data_dir}/ml-1m/users.parquet"

    @property
    def events_parquet(self) -> str:
        """Path to processed events parquet file."""
        return f"{self.data_dir}/ml-1m/events.parquet"


settings = Settings()

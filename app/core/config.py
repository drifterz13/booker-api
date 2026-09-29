from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    chroma_api_key: str
    chroma_tenant: str
    chroma_database: str

    database_url: str = "postgresql://postgres:admin@mysecret:5432/booker"

    openai_api_key: str = Field()

    rustfs_access_key: str = "rustfs"
    rustfs_secret_key: str = "secret"

from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )


class StorageConfig(Settings):
    aws_access_key_id: str = Field(validation_alias="S3_ACCESS_KEY", default="rustfs")
    aws_secret_key: str = Field(validation_alias="S3_SECRET_KEY", default="secret")
    s3_endpoint_url: str = Field(default="http://localhost:9000")
    s3_region_name: str = Field(default="eu-central-1")
    s3_bucket_name: str = Field(default="booker")


class DatabaseConfig(Settings):
    database_url: str = "postgresql+psycopg://admin:mysecret@localhost:5432/booker"


class CorsConfig(Settings):
    cors_allowed_origins: list[str] = Field(default_factory=lambda: ["*"], min_length=1)


class EmbeddingConfig(Settings):
    openai_api_key: SecretStr


class ChatConfig(EmbeddingConfig):
    chat_model_name: str = Field(default="gpt-4o-mini", pattern=r"^[\w.-]+$")
    chat_timeout_seconds: float = Field(default=120, gt=0)

    @field_validator("openai_api_key")
    @classmethod
    def validate_api_key(cls, value: SecretStr) -> SecretStr:
        key = value.get_secret_value().strip()
        if not key:
            raise ValueError("An OpenAI API key is required for chat")
        return SecretStr(key)


class IngestionConfig(Settings):
    ingestion_workers: int = Field(default=1, ge=1)


class ObservabilityConfig(Settings):
    phoenix_collector_endpoint: str | None = None
    phoenix_project_name: str = Field(default="booker-local", min_length=1)

    @field_validator("phoenix_collector_endpoint")
    @classmethod
    def normalize_endpoint(cls, value: str | None) -> str | None:
        return value.strip().rstrip("/") or None if value is not None else None


class IntegrationTestConfig(Settings):
    test_database_url: str = (
        "postgresql+psycopg://admin:mysecret@localhost:5432/booker-test"
    )
    test_s3_bucket_name: Literal["booker-test"] = "booker-test"

    @field_validator("test_database_url")
    @classmethod
    def validate_test_database(cls, value: str) -> str:
        try:
            url = make_url(value)
        except ArgumentError as error:
            raise ValueError(
                "A valid PostgreSQL test database URL is required"
            ) from error
        if (
            url.get_backend_name() not in {"postgres", "postgresql"}
            or url.database != "booker-test"
        ):
            raise ValueError(
                "Integration tests require the PostgreSQL booker-test database"
            )
        return value

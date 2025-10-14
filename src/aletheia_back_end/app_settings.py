"""Application configuration."""

from functools import lru_cache
from typing import Any

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from azure.cosmos import PartitionKey


class Settings(BaseSettings):
    """Default settings."""

    # Database settings
    db_type: str = "postgresql"
    db_user: str = "postgres"
    db_password: str = ""
    db_host: str = "127.0.0.1"
    db_port: str = "5432"
    db_name: str = "backendapi"
    database_url: str = ""

    # Azure Cosmos DB settings
    cosmos_url: str = ""
    cosmos_key: str = ""
    vector_embeddings_path: str = Field(default="vector1")
    database_name: str = ""
    container_name: str = ""
    cosmos_container_properties: dict[str, Any] = {"partition_key": PartitionKey(path="/id")}
    vector_search_fields: dict[str, Any] = {
        "field": {"field": ["value"]},
        "text_field": "text",
        "embedding_field": "vector1"
    }
    cosmos_database_properties: dict[str, Any] = {}
    vector_embedding_policy: dict[str, Any] = {}

    # Azure OpenAI settings
    azure_openai_api_key: SecretStr = Field(default=SecretStr(""))
    azure_openai_endpoint: str = ""
    azure_openai_api_version: str = ""
    azure_openai_deployment: str = ""
    azure_openai_embeddings_dimensions: int = Field(default=1536)

    # Azure OpenAI embeddings settings
    azure_openai_embeddings_model_api_key: SecretStr = Field(default=SecretStr("None"))
    azure_openai_embeddings_model_endpoint: str = ""
    azure_openai_embeddings_model_deployment: str = ""
    azure_openai_embeddings_model_api_version: str = ""

    # OpenAI settings
    openai_model_api_key: SecretStr = Field(default=SecretStr("None"))
    openai_model_endpoint: str = ""
    openai_model_deployment: str = ""
    openai_model_api_version: str = ""

    # OpenAI embeddings settings
    openai_embeddings_model_api_key: SecretStr = Field(default=SecretStr("None"))
    openai_embeddings_model_endpoint: str = ""
    openai_embeddings_model_deployment: str = ""
    openai_embeddings_model_api_version: str = ""
    openai_embeddings_model_dimensions: int = Field(default=1536)

    # --- Logging Settings ---
    LOG_LEVEL: str = "INFO"  # Default to INFO for general use
    LOG_FILE_ENABLED: bool = True  # Enable/disable file logging
    LOG_FILE_NAME: str = "aletheia_back_end_log_file.log"
    LOG_FILE_MAX_BYTES: int = 10485760  # 10 MB
    LOG_FILE_BACKUP_COUNT: int = 5  # Keep 5 backup files
    LOG_TO_CONSOLE: bool = True  # Enable/disable console logging
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # --- Workflow Settings ---
    active_workflow: str = ""
    rag_workflow_config_path: str = ""
    consistency_check_workflow_config_path: str = ""
    fact_check_workflow_config_path: str = ""
    query_expander_workflow_config_path: str = ""

    # Pydantic V2 way to configure settings
    model_config = SettingsConfigDict(
        env_file="src/aletheia_back_end/.env",
        extra="ignore",  # Ignore extra environment variables not defined in the class
        env_file_encoding="utf-8",  # Optional: specify encoding
    )


@lru_cache
def retrieve_settings(**kwargs: Any) -> Settings:
    """Helper function for creating the configuration.

    Args:
        **kwargs (Any): Keyword arguments to override the default behavior of the settings.

    Returns:
        Settings: Updated Settings object.
    """
    settings: Settings = Settings(**kwargs)
    settings.database_url = (
        f"{settings.db_type}://"
        f"{settings.db_user}:"
        f"{settings.db_password}@"
        f"{settings.db_host}:"
        f"{settings.db_port}/"
        f"{settings.db_name}"
    )
    return settings


# Instantiate the settings object at the module level.
settings = retrieve_settings()

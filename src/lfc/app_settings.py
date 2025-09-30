"""Application configuration."""

from functools import lru_cache
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Default settings."""

    # Speechmatics settings
    speechmatics_api_key: str = ""

    # Pydantic V2 way to configure settings
    model_config = SettingsConfigDict(
        env_file="lfc/.env",
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
    return settings


# Instantiate the settings object at the module level.
settings = retrieve_settings()

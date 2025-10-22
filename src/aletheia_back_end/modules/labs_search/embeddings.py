"""Embedding model configuration for vector search."""

from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings

from aletheia_back_end.app_settings import settings

# from openai import AsyncAzureOpenAI


def get_openai_embeddings() -> OpenAIEmbeddings:
    """Initialize and return OpenAI embeddings model.

    Returns:
        OpenAIEmbeddings: Configured embeddings model instance.

    Raises:
        ValueError: If required environment variables are missing.
    """
    if not all(
        [
            settings.openai_embeddings_model_deployment,
            settings.openai_embeddings_model_api_key,
            settings.openai_embeddings_model_dimensions,
        ]
    ):
        raise ValueError("Missing OpenAI environment variables")

    return OpenAIEmbeddings(
        model=settings.openai_embeddings_model_deployment,  # text-embedding-3-small
        api_key=settings.openai_embeddings_model_api_key,
        dimensions=settings.openai_embeddings_model_dimensions,
        chunk_size=1,
    )  # type: ignore[call-arg]


def get_azure_openai_embeddings() -> AzureOpenAIEmbeddings:
    """Initialize and return Azure OpenAI embeddings model.

    Returns:
        AzureOpenAIEmbeddings: Configured embeddings model instance.

    Raises:
        ValueError: If required environment variables are missing.
    """
    if not all(
        [
            settings.azure_openai_embeddings_model_deployment,
            settings.azure_openai_embeddings_model_api_version,
            settings.azure_openai_embeddings_model_endpoint,
            settings.azure_openai_embeddings_model_api_key,
            settings.azure_openai_embeddings_model_dimensions
        ]
    ):
        raise ValueError("Missing Azure OpenAI environment variables")

    return AzureOpenAIEmbeddings(
        azure_deployment=settings.azure_openai_embeddings_model_deployment,
        dimensions=settings.azure_openai_embeddings_model_dimensions,
        api_version=settings.azure_openai_embeddings_model_api_version,
        azure_endpoint=settings.azure_openai_embeddings_model_endpoint,
        openai_api_key=settings.azure_openai_embeddings_model_api_key,
        chunk_size=1,
    )  # type: ignore[call-arg]

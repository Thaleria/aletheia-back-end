"""Azure Cosmos DB vector store setup for LangChain."""

from typing import Any, Optional
import asyncio

from azure.cosmos import CosmosClient
from azure.cosmos.exceptions import CosmosHttpResponseError
from fastapi import Depends
from langchain_azure_ai.vectorstores import AzureCosmosDBNoSqlVectorSearch
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from aletheia_back_end.app_settings import settings
from aletheia_back_end.modules.labs_search.embeddings import get_azure_openai_embeddings, get_openai_embeddings
from aletheia_back_end.modules.labs_search.vector_store_interface import VectorStoreInterface
from aletheia_back_end.utils.logging_config import get_configured_logger

# Set up logging
logger = get_configured_logger(__name__)

# vector embedding policy to specify vector details
vector_embedding_policy = {
    "vectorEmbeddings": [
        {
            "path": "/" + settings.vector_embeddings_path,
            "dataType": "float32",
            "distanceFunction": "cosine",
            "dimensions": settings.azure_openai_embeddings_model_dimensions,
        }
    ]
}


# vector index policy to specify vector details
indexing_policy = {
    "indexingMode": "consistent",
    "automatic": True,
    "includedPaths": [{"path": "/*"}],
    "excludedPaths": [
        {"path": '/"_etag"/?'},
        {"path": "/" + settings.vector_embeddings_path + "/*"},
    ],
    "fullTextIndexes": [],
    "vectorIndexes": [
        {
            "path": "/" + settings.vector_embeddings_path,
            "type": "diskANN",
            "quantizationByteSize": 128,
            "indexingSearchListSize": 100,
        }
    ],
}


class AzureCosmosDBVectorStoreAdapter(VectorStoreInterface):
    """Adapter for AzureCosmosDBNoSqlVectorSearch to conform to
    VectorStoreInterface.
    """

    def __init__(
        self,
        cosmos_client: CosmosClient,
        embedding: Embeddings,
        vector_embedding_policy: dict[str, Any],
        indexing_policy: dict[str, Any],
        cosmos_container_properties: dict[str, Any],
        cosmos_database_properties: dict[str, Any],
        vector_search_fields: dict[str, Any],
        database_name: str,
        container_name: str,
        search_type: str = "hybrid",
    ):
        """Initializes the Azure Cosmos DB vector store adapter.

        This adapter initializes the underlying AzureCosmosDBNoSqlVectorSearch
        instance and applies predefined policies for vector embedding,
        indexing, and full-text search.

        Args:
            cosmos_client (CosmosClient): An initialized Azure Cosmos DB
                client.
            embedding (Embeddings): The embedding model instance.
            vector_embedding_policy (Dict[str, Any]): Vector embedding policy
                for the container.
            indexing_policy (Dict[str, Any]): Indexing policy for the
                container, including vector indexes.
            cosmos_container_properties (Dict[str, Any]): Properties for the
                Cosmos DB container.
            cosmos_database_properties (Dict[str, Any]): Properties for the
                Cosmos DB database.
            vector_search_fields (Dict[str, Any]): Fields used for vector
                search.
            database_name (str): The Cosmos DB database name.
            container_name (str): The Cosmos DB container name for the vector
                store.
            search_type (str): The type of search to perform
                (e.g., "similarity", "hybrid"). Defaults to "hybrid".
        """
        self._vector_store = AzureCosmosDBNoSqlVectorSearch(
            cosmos_client=cosmos_client,
            database_name=database_name,
            container_name=container_name,
            embedding=embedding,
            vector_embedding_policy=vector_embedding_policy,
            indexing_policy=indexing_policy,
            cosmos_container_properties=cosmos_container_properties,
            cosmos_database_properties=cosmos_database_properties,
            vector_search_fields=vector_search_fields,
            search_type=search_type
        )
        logger.info("AzureCosmosDBVectorStoreAdapter initialized.")

    async def search(self, query: str, party_id: Optional[str | int] = None, top_k: int = 10) -> Any:
        """Performs a similarity search using the Azure Cosmos DB vector store
        based on the input query.

        Args:
            query (str): The input query string to search for.
            party_id (Optional[str | int]): An optional political party ID to
                filter the search results. Defaults to None.
            top_k (int): The number of top similar documents to retrieve.
                Defaults to 5.

        Returns:
            Any: A list of documents retrieved from the vector store.
        """
        logger.info(
            f"AzureCosmosDBVectorStoreAdapter: Searching for query '{query}' with top_k='{top_k}' for party_id='{party_id}'"
        )
        cosmos_filter = None
        print(f"party_id to search for: {party_id}")
        if party_id:
            # Check the type and apply quotes for strings ---
            if isinstance(party_id, (str, bytes)):
                # If it's a string (like 'cpb'), enclose the value in single quotes.
                cosmos_filter = f"c.metadata.partyId = '{party_id}'"
            else:
                # If it's a number (like 16), use the value directly.
                cosmos_filter = f"c.metadata.partyId = {party_id}"

            return await asyncio.to_thread(
                self._vector_store.similarity_search,
                query=query,
                k=top_k,
                where=cosmos_filter
            )
        else:
            print(f"No party_id: {party_id}")
            return await asyncio.to_thread(
                self._vector_store.similarity_search,
                query=query,
                k=top_k
            )


def initiate_cosmosdb_vectorstore(documents: list[Document]) -> Any:
    """Initializes and populates an Azure Cosmos DB vector store with indexed
    documents.

    This function connects to Azure Cosmos DB using the configured settings,
    generates embeddings for the input documents using OpenAI embeddings,
    and then inserts these documents along with their embeddings into the
    specified Cosmos DB container. It also configures vector search, indexing,
    and container properties.

    Args:
        documents (list[Document]): A list of `Document` objects to be indexed
            in the vector store. Each document should contain `page_content`
            and optionally `metadata`.

    Returns:
        AzureCosmosDBNoSqlVectorSearch: Configured vector store instance.

    Raises:
        ValueError: If no documents are provided for indexing.
        RuntimeError: If there is an error during the initialization or
            document insertion process with Cosmos DB. This includes
            `CosmosHttpResponseError` and other unexpected exceptions.
    """
    if not documents:
        raise ValueError("No documents provided for indexing")

    try:
        cosmos_client = CosmosClient(settings.cosmos_url, settings.cosmos_key)
        # insert the documents in AzureCosmosDBNoSqlVectorSearch with their embeddings
        vector_store = AzureCosmosDBNoSqlVectorSearch.from_documents(
            documents=documents,
            embedding=get_azure_openai_embeddings(),
            cosmos_client=cosmos_client,
            database_name=settings.db_name,
            container_name=settings.container_name,
            vector_embedding_policy=vector_embedding_policy,
            indexing_policy=indexing_policy,
            cosmos_container_properties=settings.cosmos_container_properties,
            cosmos_database_properties={},
            vector_search_fields=settings.vector_search_fields,
            full_text_search_enabled=True,
        )
        logger.info("Cosmos DB vector store initialized successfully")
        return vector_store
    except CosmosHttpResponseError as e:
        logger.error("Cosmos DB HTTP error: %s", e)
        raise RuntimeError(f"Failed to initialize Cosmos DB vector store: {e}") from e
    except Exception as e:
        logger.error("Unexpected error initializing Cosmos DB: %s", e)
        raise RuntimeError(f"Failed to initialize Cosmos DB vector store: {e}") from e


def get_vector_store(
    embedding_model: Any,  # Depends(get_openai_embeddings)
) -> VectorStoreInterface:
    """Returns a previously configured Azure Cosmos DB vector store instance.

    This function serves as a factory or dependency injector, returning an
    initialized instance of `AzureCosmosDBVectorStoreAdapter` that conforms to
    the `VectorStoreInterface`.
    It configures the vector store using global application settings (e.g.,
    Cosmos DB URL, key, database, and container names) and integrates with the
    provided embedding model.

    Args:
        embedding_model (Any): An embedding model instance, injected.

    Returns:
        VectorStoreInterface: A configured instance of 
            `AzureCosmosDBVectorStoreAdapter` which adheres to the
            `VectorStoreInterface`.
    """
    logger.info("Creating AzureCosmosDBVectorStoreAdapter instance.")
    cosmos_client = CosmosClient(settings.cosmos_url, settings.cosmos_key)
    return AzureCosmosDBVectorStoreAdapter(
        embedding=embedding_model,
        cosmos_client=cosmos_client,
        database_name=settings.database_name,
        container_name=settings.container_name,
        vector_embedding_policy=vector_embedding_policy,
        indexing_policy=indexing_policy,
        cosmos_container_properties=settings.cosmos_container_properties,
        cosmos_database_properties=settings.cosmos_database_properties,
        vector_search_fields=settings.vector_search_fields,
        search_type="hybrid",
    )
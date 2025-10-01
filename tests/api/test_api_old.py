# """Api unit tests."""

# from typing import Any
# from unittest.mock import AsyncMock, MagicMock, patch

# import pytest
# from fastapi.testclient import TestClient
# from langchain_core.documents import Document
# from azure.cosmos import PartitionKey

# from ckb_back_end.app import app
# from ckb_back_end.modules.labs_nlp.azure_client import get_llm_client
# from ckb_back_end.modules.labs_search.cosmos_db import get_vector_store
# from ckb_back_end.modules.labs_search.embeddings import get_openai_embeddings

# client = TestClient(app)


# @pytest.fixture
# def mock_data() -> dict[str, Any]:
#     """Provides mock input data for chat endpoint tests.

#     Returns:
#         A dictionary containing the mock chat input structure.
#     """
#     return {
#         "messages": [{"content": "Mock data example", "role": "user"}],
#         "context": {
#             "overrides": {
#                 "top": 3,
#                 "temperature": 0.3,
#                 "minimum_reranker_score": 0,
#                 "minimum_search_score": 0,
#                 "retrieval_mode": "hybrid",
#                 "semantic_ranker": True,
#                 "semantic_captions": False,
#                 "suggest_followup_questions": False,
#                 "use_oid_security_filter": False,
#                 "use_groups_security_filter": False,
#                 "vector_fields": ["embedding"],
#                 "use_gpt4v": False,
#                 "gpt4v_input": "textAndImages",
#             }
#         },
#         "session_state": None,
#     }


# @pytest.fixture
# def mock_settings_fixture() -> MagicMock:
#     """Provides mock settings for Cosmos DB configuration.

#     Returns:
#         A MagicMock instance simulating settings with Cosmos DB attributes.
#     """
#     mock = MagicMock()
#     mock.cosmos_url = "mock-url"
#     mock.cosmos_key = "mock-key"
#     mock.vector_embeddings_path = "mock-vector_embeddings_path"
#     mock.database_name = "mock-db"
#     mock.container_name = "mock-container"
#     mock.cosmos_vector_property = "embedding"
#     mock.cosmos_container_properties = {"partition_key": PartitionKey(path="/id")}
#     mock.vector_search_fields = {
#         "field": {"field": ["value"]},
#         "text_field": "text",
#         "embedding_field": "vector1",
#     }
#     mock.cosmos_database_properties = {}
#     mock.vector_embedding_policy = {}
#     mock.openai_embeddings_dimensions = 1536
#     mock.azure_openai_endpoint = "mock-azure_openai_endpoint"
#     mock.azure_openai_api_key = "mock-azure_openai_api_key"
#     mock.openai_embeddings_model = "text-embedding-ada-003"
#     mock.azure_openai_deployment = "mock-azure_openai_deployment"
#     mock.openai_api_version = "mock-openai_api_version"
#     mock.openai_embeddings_model_deployment = "mock-openai_embeddings_model_deployment"

#     return mock


# # @pytest.fixture # TODO: To create in a future iteration
# # def mock_data_empty_query():
# #     data = mock_data()
# #     data["messages"] = [{"content": ""}]
# #     return data


# @pytest.fixture
# def mock_query_processor() -> MagicMock:
#     """Provides a mock query processor.

#     Configures `process` to return a predefined "Mocked rewritten query response".

#     Returns:
#         A MagicMock instance simulating a query processor.
#     """
#     mock = MagicMock()
#     mock.process.return_value = "Mocked rewritten query response"
#     return mock


# @pytest.fixture
# def mock_retrieved_documents() -> list[Document]:
#     """Provides a list of mock Document objects.

#     Returns:
#         list[Document]: A list of sample `Document` objects for testing.
#     """
#     return [
#         Document(
#             page_content="Content of document 1", metadata={"id": "doc1", "score": 0.9}
#         ),
#         Document(
#             page_content="Content of document 2", metadata={"id": "doc2", "score": 0.8}
#         ),
#     ]


# @pytest.fixture
# def mock_vector_store_adapter(mock_retrieved_documents: list[Document]) -> MagicMock:
#     """Provides a mock of the VectorStoreInterface.

#     Args:
#         mock_retrieved_documents (list[Document]): A pytest fixture providing
#             a list of mock `Document` objects for the `search` method's return.

#     Returns:
#         MagicMock: A configured mock object simulating the vector store adapter.
#     """
#     mock_adapter = MagicMock()
#     # Ensure the mock's search method is an async mock if the real one is async
#     mock_adapter.search = AsyncMock(return_value=mock_retrieved_documents)
#     return mock_adapter


# @pytest.fixture
# def mock_azure_client() -> MagicMock:
#     """Provides a mock Azure OpenAI client.

#     Returns:
#         A MagicMock instance simulating AzureOpenAI client.
#     """
#     mock = MagicMock()
#     mock.process.return_value = "Mocked LLM response"
#     return mock


# @pytest.fixture
# def mock_llm_client() -> MagicMock:
#     """Provides a mock LLM client.

#     Configures `rag_llm_call` to return a predefined "Generated response".

#     Returns:
#         A MagicMock instance simulating an LLM client.
#     """
#     mock = MagicMock()
#     mock.rag_llm_call.return_value = "Mocked LLM response"
#     return mock


# def test_chat_endpoint_success(
#     mock_data: Any,
#     mock_settings_fixture: MagicMock,
#     mock_vector_store_adapter: MagicMock,
#     mock_llm_client: Any,
#     mock_query_processor: MagicMock
# ) -> None:
#     """Test Case: Verifies successful processing of a chat request with mocked  vector store and LLM.

#     Verifies the `/v1/chat` endpoint correctly handles a valid request. It
#     confirms API interaction with mocked vector store and LLM dependencies,
#     ensuring a successful response with expected content. The test uses
#     FastAPI `dependency_overrides` and `unittest.mock.patch` for dependency
#     control and test isolation.

#     Args:
#         mock_data (Any): Mock input data for the chat request.
#         mock_settings_fixture (MagicMock): Mock application settings.
#         mock_vector_store_adapter (MagicMock): Mock vector store adapter.
#         mock_llm_client (Any): Mock LLM client.

#     Dependencies Mocked:
#     - `get_openai_embeddings`: Overridden to prevent actual embedding model
#         initialization.
#     - `get_vector_store`: Overridden to provide `mock_vector_store_adapter`.
#     - `get_llm_client`: Overridden to provide `mock_llm_client`.
#     - `CosmosClient`: Patched to bypass actual client creation.
#     - `settings` (across modules): Patched to use `mock_settings_fixture` for
#         configuration.
#     - `AzureOpenAIEmbeddings`: Patched to bypass actual client creation.

#     Test Flow:
#     1.  FastAPI dependency overrides are configured.
#     2.  `unittest.mock.patch` contexts are applied for module-level mocks.
#     3.  A POST request is sent to `/v1/chat`.

#     **Assertions:**
#     1.  **HTTP Status:** Asserts response status is `200 OK`.
#     2.  **Response Payload:** Asserts JSON response matches expected `ChatOut`
#         structure.
#     3.  **Vector Store Interaction:** Asserts
#     `mock_vector_store_adapter.search` was called once with specific arguments.
#     4.  **LLM Interaction:** Asserts `mock_llm_client.call_llm` was called
#         once with specific arguments.
#     """
#     app.dependency_overrides[get_openai_embeddings] = lambda: MagicMock()
#     app.dependency_overrides[get_vector_store] = lambda: mock_vector_store_adapter
#     app.dependency_overrides[get_llm_client] = lambda: mock_llm_client

#     with (
#         patch(
#             "ckb_back_end.modules.labs_search.cosmos_db.CosmosClient",
#             return_value=MagicMock(),
#         ),
#         patch(
#             "ckb_back_end.modules.labs_search.cosmos_db.settings",
#             new=mock_settings_fixture,
#         ),
#         patch(
#             "ckb_back_end.modules.labs_search.embeddings.settings",
#             new=mock_settings_fixture,
#         ),
#         patch(
#             "ckb_back_end.modules.labs_nlp.azure_client.settings", new=mock_settings_fixture
#         ),
#         patch("langchain_openai.AzureOpenAIEmbeddings", return_value=MagicMock()),
#     ):
#         response = client.post("/v1/chat", json=mock_data)
#         assert response.status_code == 200
#         assert response.json() == {
#             "delta": {"content": "Generated response", "role": "assistant"}
#         }
#         mock_vector_store_adapter.search.assert_called_once_with(
#             "Mock data example", top_k=3, search_type="similarity"
#         )
#         mock_llm_client.rag_llm_call.assert_called_once_with(
#             message="Mock data example",
#             context="Retrieval context: Content of document 1 Content of document 2",
#             temperature=0.3,
#         )


# # TODO tests

# # test_chat_endpoint_empty_query:
# # User sends an empty query. Define how the API should respond (e.g., 400 Bad Request with an error message).

# # test_chat_endpoint_no_retrieved_documents:
# # Tests the case where the vector store returns no relevant documents. Graceful fallback (e.g., a generic "I don't know" message or still call the LLM with an empty context).

# # test_chat_endpoint_vector_store_error:
# # Rrror occurs during the vector store search to ensure your API returns an appropriate 500 error.

# # test_chat_endpoint_llm_error: Simulates an error during the LLM call to ensure your API returns an appropriate 500 error.

# # test_chat_endpoint_invalid_json_input: Tests FastAPI's built-in Pydantic validation for malformed or incorrect JSON inputs, expecting a 422 status code.

# # TODO: drafts for future functions
# # Helper function to apply common patches
# # def apply_common_patches(mocker, mock_settings, mock_llm_client, mock_cosmos_search_return_value):
# #     mocker.patch("ckb_back_end.modules.labs_search.cosmos_db.get_vector_store", return_value=MagicMock())
# #     mocker.patch("tests.simulation_functions.AzureLLMClient", return_value=mock_llm_client)
# #     mocker.patch("tests.simulation_functions.get_llm_client", return_value=mock_llm_client)
# #     mocker.patch("ckb_back_end.modules.labs_search.cosmos_db.CosmosClient", return_value=MagicMock())
# #     mocker.patch("ckb_back_end.modules.labs_search.cosmos_db.settings", new=mock_settings)
# #     mocker.patch("ckb_back_end.modules.labs_search.embeddings.settings", new=mock_settings)
# #     mocker.patch("tests.simulation_functions.settings", new=mock_settings)
# #     mocker.patch("langchain_openai.AzureOpenAIEmbeddings", return_value=MagicMock())
# #     return mocker.patch("langchain_azure_ai.vectorstores.AzureCosmosDBNoSqlVectorSearch.search", return_value=mock_cosmos_search_return_value)


# # def test_chat_endpoint_empty_query(mock_data, mock_settings_fixture, mock_llm_client, mocker):
# #     """Tests chat request with an empty user query."""
# #     mock_data["messages"][0]["content"] = ""
# #     mock_cosmos_search = apply_common_patches(mocker, mock_settings_fixture, mock_llm_client, []) # No documents for empty query

# #     response = client.post("/v1/chat", json=mock_data)

# #     assert response.status_code == 400 # Or whatever your API returns for empty queries
# #     assert "error" in response.json() # Check for an error message or specific error structure


# # def test_chat_endpoint_no_retrieved_documents(mock_data, mock_settings_fixture, mock_llm_client, mocker):
# #     """Tests chat request when no documents are retrieved from the vector store."""
# #     mock_cosmos_search = apply_common_patches(mocker, mock_settings_fixture, mock_llm_client, []) # Empty list of documents

# #     response = client.post("/v1/chat", json=mock_data)

# #     assert response.status_code == 200 # Still 200, but content should indicate no answer
# #     assert response.json()["delta"]["content"] == "Generated response" # Or "I couldn't find an answer based on the provided context." if your LLM handles it this way
# #     mock_cosmos_search.assert_called_once_with("Mock data example", top_k=3, search_type="similarity")
# #     # Assert LLM was called, possibly with an empty context or a specific "no docs" context
# #     mock_llm_client.call_llm.assert_called_once()
# #     assert "Retrieval context: " == mock_llm_client.call_llm.call_args.kwargs["context"]


# # def test_chat_endpoint_vector_store_error(mock_data, mock_settings_fixture, mock_llm_client, mocker):
# #     """Tests chat request when the vector store search raises an exception."""
# #     # Make the search method raise an exception
# #     mock_cosmos_search = apply_common_patches(mocker, mock_settings_fixture, mock_llm_client, [])
# #     mock_cosmos_search.side_effect = Exception("Cosmos DB connection failed")

# #     response = client.post("/v1/chat", json=mock_data)

# #     assert response.status_code == 500 # Internal Server Error
# #     assert "error" in response.json()
# #     assert "Cosmos DB connection failed" in response.json()["error"] # Check for specific error message


# # def test_chat_endpoint_llm_error(mock_data, mock_retrieved_documents, mock_settings_fixture, mock_llm_client, mocker):
# #     """Tests chat request when the LLM call raises an exception."""
# #     # Make the LLM call raise an exception
# #     mock_llm_client.call_llm.side_effect = Exception("LLM service unavailable")
# #     apply_common_patches(mocker, mock_settings_fixture, mock_llm_client, mock_retrieved_documents)

# #     response = client.post("/v1/chat", json=mock_data)

# #     assert response.status_code == 500 # Internal Server Error
# #     assert "error" in response.json()
# #     assert "LLM service unavailable" in response.json()["error"] # Check for specific error message

# # def test_chat_endpoint_invalid_json_input(mocker):
# #     """Tests chat endpoint with malformed JSON input."""
# #     # No need for mocks here, FastAPI's Pydantic validation should catch this
# #     response = client.post("/v1/chat", content="this is not json", headers={"Content-Type": "application/json"})
# #     assert response.status_code == 422 # Unprocessable Entity

# #     response = client.post("/v1/chat", json={"messages": "not a list"}) # Invalid messages field
# #     assert response.status_code == 422

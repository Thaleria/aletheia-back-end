from unittest.mock import AsyncMock
from fastapi.testclient import TestClient
import pytest
from typing import Any

from aletheia_back_end.app import app
from aletheia_back_end.api import chat
from aletheia_back_end.modules.workflows.rag_graph import get_rag_workflow_app

client = TestClient(app)

@pytest.fixture
def mock_data() -> dict[str, Any]:
    """Provides mock input data for chat endpoint tests.

    Returns:
        A dictionary containing the mock chat input structure.
    """
    return {
        "messages": [{"content": "Mock data example", "role": "user"}],
        "context": {
            "overrides": {
                "top": 3,
                "temperature": 0.3,
                "minimum_reranker_score": 0,
                "minimum_search_score": 0,
                "retrieval_mode": "hybrid",
                "semantic_ranker": True,
                "semantic_captions": False,
                "suggest_followup_questions": False,
                "use_oid_security_filter": False,
                "use_groups_security_filter": False,
                "vector_fields": ["embedding"],
                "use_gpt4v": False,
                "gpt4v_input": "textAndImages",
            }
        },
        "session_state": None,
    }


@pytest.fixture(autouse=True)
def mock_rag_workflow() -> Any:
    """Automatically overrides the RAG workflow dependency with a mock.

    This fixture replaces the real `get_rag_workflow_app` dependency
    with an `AsyncMock` that returns a fixed output, ensuring that
    tests do not call external services.

    Yields:
        Any: This is a setup/teardown fixture.
    """
    mock = AsyncMock()
    mock.ainvoke.return_value = {"output": "Mocked answer"}
    app.dependency_overrides[get_rag_workflow_app] = lambda: mock
    yield mock
    app.dependency_overrides.clear()


def test_post_chat(mock_data: dict[str, Any], mock_rag_workflow: AsyncMock) -> None:
    """Tests the application endpoint with mocked RAG workflow.

    Sends a POST request to the chat endpoint with the provided
    mock data and verifies that the mocked RAG workflow output
    is returned in the response.

    Args:
        mock_data (dict[str, Any]): The mock request payload.
    """

    response = client.post("/v1/chat", json=mock_data)
    assert response.status_code == 200
    data = response.json()
    assert data["delta"]["content"] == "Mocked answer"

    mock_rag_workflow.ainvoke.assert_awaited_once_with(
            {'query': 'Mock data example'}  # TODO: Eventually it will have more fields and they will need to be put here. But for now the function is only taking the query
        )

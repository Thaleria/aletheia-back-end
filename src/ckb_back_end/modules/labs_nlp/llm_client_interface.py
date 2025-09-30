"""LLM client abstract class for LLM providers."""

from abc import ABC, abstractmethod
from typing import Any

from ckb_back_end.utils.logging_config import get_configured_logger

# Set up logging
logger = get_configured_logger(__name__)


class LLMClientInterface(ABC):
    """Abstract Base Class defining the interface for any LLM client.

    All concrete LLM client implementations (e.g., AzureLLMClient,
    OpenAILLMClient) must implement the methods defined here.
    """

    @abstractmethod
    async def rag_llm_call(self, query: str, context: str, prompt: str) -> str:
        """Abstract method to call an LLM with a given query and context.

        Implementations should handle the specifics of the LLM API call.

        Args:
            query (str): The user's input query for the LLM.
            context (str): The RAG context to be provided to the LLM.
            prompt (str): The prompt template to be used for the LLM call.

        Returns:
            str: The LLM-generated response content.
        """
        pass

    @property
    @abstractmethod
    def llm(
        self,
    ) -> (
        Any
    ):  # This is Any as there could be different LLM objects (OpenAI, Azure... etc)
        """The underlying LLM object."""
        raise NotImplementedError

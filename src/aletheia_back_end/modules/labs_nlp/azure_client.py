"""Service to use AzureOpenAI model."""

from typing import Any
import os

from langchain_openai import AzureChatOpenAI

from aletheia_back_end.app_settings import settings
from aletheia_back_end.modules.labs_nlp.llm_client_interface import LLMClientInterface
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from aletheia_back_end.utils.utils import load_prompt_template

default_prompt = load_prompt_template(os.path.join(os.path.dirname(__file__), "default_prompt.txt"))


class AzureLLMClient(LLMClientInterface):
    """Client for interacting with Azure OpenAI LLM services.

    This class implements the `LLMClientInterface` to provide a concrete
    implementation for making chat completion calls to Azure OpenAI.
    """
    @property
    def llm(self) -> Any:
        """The underlying LLM object."""
        return self._llm

    def __init__(
            self,
            temperature: float = 0.,
            max_tokens: int = 5000,
            timeout: int = 30
    ):
        """Initializes the Azure LLM client with settings from the
        configuration.

        Args:
            temperature: The sampling temperature for the LLM. Defaults to 0.5.
            max_tokens: The maximum number of tokens to generate. Defaults to 500.
            timeout: The request timeout in seconds. Defaults to 30.
        """
        self._llm = AzureChatOpenAI(
                azure_endpoint=settings.azure_openai_endpoint,
                api_key=settings.azure_openai_api_key,
                azure_deployment=settings.azure_openai_deployment,
                api_version=settings.azure_openai_api_version,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout
            )

    async def rag_llm_call(self, query: str, context: str, prompt: str = default_prompt) -> str:
        """Calls an Azure Chat OpenAI endpoint with a given message, context and prompt.

        This method initializes an AzureOpenAI client using configured settings
        and sends a chat completion request to the LLM. It constructs the
        messages with a system role providing the RAG context and a human role
        with the user's query.

        Args:
            query (str): The user's input query for the LLM.
            context (str): The RAG context to be provided to the LLM, derived
                from retrieved documents.
            prompt (str): A string template for the prompt to be sent to the LLM.

        Returns:
            str: The LLM-generated response content. This will be an error
                message string if an exception occurs during the LLM call.
        """
        try:
            prompt_template = ChatPromptTemplate.from_messages(
                [
                    ("system", prompt),
                    ("human", "query: {query}\nContext: {context}"),
                ]
            )

            # Use the passed-in prompt_template and llm
            rag_chain = prompt_template | self.llm | StrOutputParser()

            llm_output = await rag_chain.ainvoke({"query": query, "context": context})

            if isinstance(llm_output, str):
                return llm_output
            else:  # Handle unexpected return types
                return str(llm_output)
        except Exception as e:
            return f"Error: {str(e)}"


def get_llm_client() -> LLMClientInterface:
    """Provides an instance of an LLM client.

    This function serves as a factory or dependency injector, returning an
    initialized instance of `AzureLLMClient` that conforms to the
    `LLMClientInterface`.

    Returns:
        LLMClientInterface: An initialized LLM client object, adhering to the
            `LLMClientInterface`.
    """
    return AzureLLMClient()

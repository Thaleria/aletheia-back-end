import os
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models import BaseChatModel  # Import BaseChatModel for type hinting
from abc import ABC, abstractmethod
from typing import Any
from ckb_back_end.utils.logging_config import get_configured_logger
from ckb_back_end.utils.utils import load_prompt_template

query_rewriter_prompt = load_prompt_template(os.path.join(os.path.dirname(__file__), "query_rewriter_prompt.txt"))

# Set up logging
logger = get_configured_logger(__name__)


class QueryProcessor(ABC):
    """Abstract base class for query processing.

    Defines the interface for transforming a query string.
    """
    @abstractmethod
    async def process(self, query: str) -> str:
        """
        Takes an input query string and returns a processed query string.

        Args:
            query (str): Input query string to be processed.

        Returns:
            str: Processed query string.
        """
        pass


class QueryRewriter(QueryProcessor):
    """Rewrites an original query using an LLM to improve retrieval.

    The class takes the user query and uses an LLM to reformulate it, aiming
    for a more specific and detailed query that is likely to retrieve more
    relevant information in a RAG system.
    """
    def __init__(self, llm: BaseChatModel, prompt: str = query_rewriter_prompt):
        """Initializes the QueryRewriter.

        Args:
            llm: An instance of LangChain's BaseChatModel, used for
                rewriting the query.
            prompt: A string template for the prompt to be sent to the LLM.

        Raises:
            TypeError: If the 'llm' argument is not an instance of
                LangChain's BaseChatModel.
        """

        if not isinstance(llm, BaseChatModel):
            raise TypeError("The 'llm' argument must be an instance of LangChain's BaseChatModel.")
        self.llm = llm

        self.llm_template_string = prompt

        self.prompt_template = PromptTemplate(
            input_variables=["original_query"],
            template=self.llm_template_string
        )

        self.query_rewriter_chain = self.prompt_template | self.llm

    async def process(self, query: str) -> Any:
        """Rewrites the original query using an LLM to improve retrieval.

        This method sends the original query to the configured LLM with a
        specific prompt to generate a more effective query for retrieval.
        In case of an error during the LLM call, it falls back to returning
        the original query.

        Args:
            query (str): The original query string to be rewritten.

        Returns:
            str: The rewritten query string, or the original query if an
                error occurs during rewriting.
        """
        logger.debug(f"\nOriginal query: '{query}'")

        try:
            response = await self.query_rewriter_chain.ainvoke({"original_query": query})  # The LC chain returns an AIMessage object
            processed_query = response.content
            logger.debug(f"Rewritten query: '{processed_query}'")
        except Exception as e:
            logger.debug(f"Error during query rewriting: {e}")
            processed_query = query  # Fallback to original query on error
            logger.debug(f"Falling back to original query: '{processed_query}'")

        return processed_query


class QueryStepBack(QueryProcessor):  # TODO: Idea of making a class that makes the query to have broader context.
    """Generate a step-back query to retrieve broader context."""


class QueryDecomposer(QueryProcessor):  # TODO: Idea of making a class that forms a step by step plan based on the query.
    """Decomposing the query into a series of steps."""


class QueryExpander(QueryProcessor):  # TODO: Idea of broadening the original query to capture more relevant information, by generating multiple similar queries based on the user's initial input
    """Generates multiple similar queries based on the original one."""

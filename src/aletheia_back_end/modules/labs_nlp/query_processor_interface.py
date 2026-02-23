from langchain_core.prompts import PromptTemplate
from langchain_core.language_models import BaseChatModel  # Import BaseChatModel for type hinting
from abc import ABC, abstractmethod
from typing import Any
from aletheia_back_end.utils.logging_config import get_configured_logger

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
    def __init__(self, llm: BaseChatModel, prompt: str):
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


class QueryExpander(QueryProcessor):
    """Generates multiple similar queries based on the original one using an
    LLM to improve retrieval.

    The class takes the user query and uses an LLM to generate n queries based
    on it, aiming to retrieve more relevant information in a RAG system.
    """
    def __init__(self, llm: BaseChatModel, prompt: str, num_queries: int = 3):
        """Initializes the QueryExpander.

        Args:
            llm: An instance of LangChain's BaseChatModel, used for
                expanding the query.
            prompt: A string template for the prompt to be sent to the LLM.
            num_queries: Number of expanded queries to generate.

        Raises:
            TypeError: If the 'llm' argument is not an instance of
                LangChain's BaseChatModel.
        """

        if not isinstance(llm, BaseChatModel):
            raise TypeError("The 'llm' argument must be an instance of LangChain's BaseChatModel.")
        self.llm = llm
        self.num_queries = num_queries

        self.llm_template_string = prompt

        self.prompt_template = PromptTemplate(
            input_variables=["original_query", "num_queries"],  # TODO: This is not really doing anything
            template=self.llm_template_string
        )

        self.query_expander_chain = self.prompt_template | self.llm

    async def process(self, query: str) -> Any:
        """Expands the original query using an LLM to improve retrieval.

        This method sends the original query to the configured LLM with a
        specific prompt to generate several queries for retrieval.
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
            response = await self.query_expander_chain.ainvoke({"num_queries": self.num_queries, "original_query": query})  # The LC chain returns an AIMessage object
            expanded_queries = response.content
            logger.debug(f"Expanded queries: '{expanded_queries}'")
        except Exception as e:
            logger.debug(f"Error during query expanding: {e}")
            expanded_queries = query  # Fallback to original query on error
            logger.debug(f"Falling back to original query: '{expanded_queries}'")

        return expanded_queries

    def _parse_llm_response(self, response: str) -> list[str]:
        """Parse LLM response into a list of query variations.

        Args:
            response (str): Raw LLM output (e.g., numbered list of queries).

        Returns:
            List[str]: List of parsed query variations.
        """
        lines = response.strip().split("\n")
        queries = []
        for line in lines:
            # Expected format like "1. Some query text"
            if line.strip() and line[0].isdigit() and "." in line:
                query_text = line[line.index(".") + 1:].strip()
                if query_text:
                    queries.append(query_text)
        return queries

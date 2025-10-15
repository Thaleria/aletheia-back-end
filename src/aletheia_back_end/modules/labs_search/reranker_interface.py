from operator import itemgetter
from langchain_core.language_models import BaseChatModel
from abc import ABC, abstractmethod
from typing import Optional
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from aletheia_back_end.utils.logging_config import get_configured_logger

# Set up logging
logger = get_configured_logger(__name__)


class RerankerInterface(ABC):
    """
    Abstract Base Class (Interface) of a reranker for a RAG setting.
    """
    @abstractmethod
    async def rerank(self, documents: list[Document], original_query: str, top_k: int = 5) -> list[Document]:
        """Retrieves the top_k documents from a vector store.

        Args:
            documents (list[Document]): A list of LangChain Document objects
                to be reranked.
            original_query (str): The user's original query string.
            top_k (int): The number of top relevant documents to retrieve.
                Defaults to 5.

        Returns:
            list[Document]: A list of LangChain Document objects representing
                the retrieved content.
        """
        pass

    @abstractmethod
    def build_rag_context(self, documents: list[Document]) -> str:
        """Builds a context string from a list of documents.

        Args:
            documents (list[Document]): A list of Document objects.

        Returns:
            str: A single string representing the combined content of all
                documents.
        """
        pass


# TODO: These classes could be in a different file called like langchain_retriever.py as these are made for langchain based documents and ecosystem.
# That way these file would just be for the abstract class and the langchain_retriever.py for the langchain based classes
class Reranker(RerankerInterface):
    """A reranker that reranks documents using an LLM.

    This reranker extends the `BasicRetriever` by adding a reranking step.
    After an initial retrieval, it uses an LLM to score the relevance of each
    document to the query, and then returns the top-scoring documents.
    """
    def __init__(self,
                 llm: BaseChatModel,  # TODO: Verify if the BaseChatModel really is needed, the LLM's may come in different types
                 prompt: str,
                 top_k: int = 5) -> None:
        """Initializes the RerankingRetriever.

        Args:
            llm (BaseChatModel): The LLM object used for reranking documents.
            top_k (int): The number of top reranked documents to
                return. Defaults to 5.

        Raises:
            TypeError: If the 'llm' argument is not an instance of
                LangChain's BaseChatModel.
        """
        self.top_k = top_k

        if not isinstance(llm, BaseChatModel):
            raise TypeError("The 'llm' argument must be an instance of LangChain's BaseChatModel.")
        self.llm = llm
        self.prompt_template = PromptTemplate(
            input_variables=["query", "doc"],
            template=prompt
        )
        self.llm_chain = self.prompt_template | self.llm

    async def _get_llm_relevance_score(self, query: str, doc_content: str) -> float:
        """Method to get a relevance score from the LLM for a single document.

        This method uses an LLM to give a numerical relevance score based on
        the document's content and the query. The LLM's output is then parsed.
        The score is clamped between 1.0 and 10.0.

        Args:
            query (str): The original query string.
            doc_content (str): The content of the document to be scored.

        Returns:
            float: The relevance score (1.0-10.0) assigned by the LLM.
                Returns 0.0 if the score cannot be parsed.
        """
        input_data = {"query": query, "doc": doc_content}
        try:
            llm_score = float((await self.llm_chain.ainvoke(input_data)).content)  # The self.llm_chain.invoke(input_data) result comes in the AIMessage format from LangChain
            # logger.debug('Debug llm_score: ', llm_score)
            clamped_score = max(1.0, min(10.0, llm_score))
            # In case it's needed, here's the code to use regex
            # match = re.search(r'\d+(\.\d+)?', response_content)

            logger.info(f"The doc_content: {doc_content} had a score of {clamped_score}.")
            return clamped_score  # Clamping the score

        except ValueError:
            logger.warning(f"Warning: Could not parse score from LLM response: '{llm_score}'. Defaulting to 0.")
            return 0.0  # Default score if parsing fails

    async def rerank(self, documents: list[Document], query: str, k: Optional[int] = None) -> list[Document]:
        """Reranks the documents revlevance, regarding the user's query, using
        an LLM.

        This method takes a list of documents and uses an LLM to score their
        relevance to the original user query. It then sorts the documents by
        their scores and returns the top_k most relevant ones.

        Args:
            documents (list[Document]): A list of LangChain Document objects
                to be reranked.
            query (str): The query string.
            k (int, optional): The number of final reranked documents to return.
                If None, uses `self.top_k`.

        Returns:
            list[Document]: A list of reranked documents.
        """
        # Number of final documents to return
        final_k = k if k is not None else self.top_k

        # Rerank the retrieved documents using the LLM
        logger.debug('\nDebug, helpful to get an idea if the LLM is scoring decently:')
        scored_docs = []
        for doc in documents:
            score = await self._get_llm_relevance_score(query, doc.page_content)
            logger.debug("\nDocument content:\n", doc.page_content)
            logger.debug("Relevance Score: ", score)
            scored_docs.append((doc, score))

        # Sort documents by their relevance score in descending order
        reranked_docs_with_scores = sorted(scored_docs, key=itemgetter(1), reverse=True)
        logger.debug(f"Document contents with their scores: {reranked_docs_with_scores}")

        # Extract only the Document objects and return the top_n
        final_reranked_documents = [doc for doc, _ in reranked_docs_with_scores[:final_k]]
        logger.debug(f"\nThe Rekanker returned {len(final_reranked_documents)} reranked documents.")
        logger.info(f"Final retrieval documents with the highest score: {final_reranked_documents}")

        return final_reranked_documents  # To debug, also return scored_docs

    def build_rag_context(self, documents: list[Document]) -> str:
        """Builds a context string from a list of documents.

        This function iterates through a list of `Document` objects, extracts
        the `page_content` from each, and concatenates them into a single
        string. The resulting string is prefixed with "Context taken from the
        retrieved documents: " to clearly indicate its purpose in a RAG system.

        Args:
            documents (list[Document]): A list of Document objects, each
                containing a page_content attribute that is used to build
                the context.

        Returns:
            str: A single string representing the combined content of all
                documents.Returns an empty string if the input `documents`
                list is empty.
        """
        if not documents:
            return ""  # Handle empty list case

        # Extract page_content from each document
        # TODO: Write Notion task for this
        document_contents = ['\n'.join([str(doc.metadata), doc.page_content]) for doc in documents]  # Include metadata in the context?
        context = "\n\n".join(document_contents)

        return f"Context taken from the retrieved documents: {context}"

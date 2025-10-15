import os
from operator import itemgetter
from langchain_core.language_models import BaseChatModel
from abc import ABC, abstractmethod
from typing import Optional
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from .vector_store_interface import VectorStoreInterface
from aletheia_back_end.utils.logging_config import get_configured_logger

# Set up logging
logger = get_configured_logger(__name__)


class RetrieverInterface(ABC):
    """
    Abstract Base Class (Interface) of a retriever for a RAG setting.
    """
    @abstractmethod
    async def retrieve(self, query: str, top_k: int = 5, party_id: Optional[str | int] = None) -> list[Document]:
        """Retrieves the top_k documents from a vector store.

        Args:
            query (str): The user's query string.
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


class BasicRetriever(RetrieverInterface):
    """A basic retriever that fetches documents from a vector store using
    similarity search.
    """
    def __init__(self, vector_store: VectorStoreInterface, top_k: int = 5) -> None:
        """Initializes the BasicRetriever.

        Args:
            vector_store (VectorStoreInterface): The vector store instance.
            top_k (int): Number of top documents to retrieve. Defaults to 5.
        """
        self.top_k = top_k
        self.vector_store = vector_store

    async def retrieve(self, query: str, k: Optional[int] = None, party_id: Optional[str | int] = None ) -> list[Document]:
        """Retrieves documents from the vector store based on the query.

        Performs a similarity search to find the most relevant documents.

        Args:
            query (str): The query string.
            k v: The number of top documents to retrieve.
                If None, `self.top_k` is used.

        Returns:
            list[Document]: A list of retrieved documents. Returns an empty
                list if the search fails.
        """
        k = k if k is not None else self.top_k
        # This vector_store uses the search function from the abstract class, which then uses the similarity search function of Cosmos DB
        logger.info(f"Performing similarity search for query: {query} to retrieve {k} documents about party_id: {party_id}")
        try:
            retrieved_docs = await self.vector_store.search(query, top_k=k, party_id=party_id)
            logger.debug(f"Debug: Retrieved {len(retrieved_docs)} documents")
            return retrieved_docs
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []

    def build_rag_context(self, documents: list[Document]) -> str:
        """Builds a context string from a list of documents.

        This function iterates through a list of `Document` objects, extracts
        the `page_content` from each and concatenates them into a string.
        The resulting string is prefixed with "Retrieval context: " to clearly
        indicate its purpose in a RAG system.

        Args:
            documents (list[Document]): A list of Document objects, each
                containing a page_content attribute that is used to build the
                context.

        Returns:
            str: A single string representing the combined content of all
                documents.
                Returns an empty string if the documents' list is empty.
        """
        if not documents:
            return ""  # Handle empty list case

        # Extract page_content from each document
        document_contents = [doc.page_content for doc in documents]
        context = "\n\n".join(document_contents)
        logger.info(f"Context: {context}")

        return f"Retrieval context: {context}"


# TODO: These classes could be in a different file called like langchain_retriever.py as these are made for langchain based documents and ecosystem.
# That way these file would just be for the abstract class and the langchain_retriever.py for the langchain based classes
class RerankingRetriever(BasicRetriever):
    """A retriever that fetches documents and then reranks them using an LLM.

    This retriever extends the `BasicRetriever` by adding a reranking step.
    After an initial retrieval, it uses an LLM to score the relevance of each
    document to the query, and then returns the top-scoring documents.
    """
    def __init__(self,
                 vector_store: VectorStoreInterface,
                 llm: BaseChatModel,  # TODO: Verify if the BaseChatModel really is needed, the LLM's may come in different types
                 prompt: str,
                 top_k_initial: int = 10,
                 top_n_reranked: int = 5) -> None:
        """Initializes the RerankingRetriever.

        Args:
            vector_store (VectorStoreInterface): The vector store instance.
            llm (BaseChatModel): The LLM object used for reranking documents.
            top_k_initial (int): The number of documents to retrieve in the
                initial search before reranking. Defaults to 10.
            top_n_reranked (int): The number of top reranked documents to
                return. Defaults to 5.

        Raises:
            TypeError: If the 'llm' argument is not an instance of
                LangChain's BaseChatModel.
        """
        super().__init__(vector_store)
        self.top_k_initial = top_k_initial
        self.top_n_reranked = top_n_reranked

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

    async def retrieve(self, query: str, k: Optional[int] = None, party_id: Optional[str | int] = None) -> list[Document]:
        """Performs initial retrieval and then reranks the results using an LLM.

        This method first retrieves a larger set of documents using the
        vector store's similarity search and then uses the LLM to rerank
        these documents based on their relevance to the query. Finally,
        it returns the top `k` reranked documents.

        Args:
            query (str): The query string.
            k (int, optional): The number of final reranked documents to return.
                If None, uses `self.top_n_reranked`.

        Returns:
            list[Document]: A list of reranked documents.
        """
        # Number of final documents to return
        final_k = k if k is not None else self.top_n_reranked

        # Step 1: Initial retrieval (call parent's retrieve method)
        initial_retrieved_docs = await super().retrieve(query, k=self.top_k_initial, party_id=party_id)  # This retrieve calls the parent's retrieve function which calls the vector store search function which calls the Cosmos DB similarity search function
        logger.debug(f"\nInitial retrieval: {len(initial_retrieved_docs)} documents.")

        if not initial_retrieved_docs:
            logger.debug("No documents retrieved initially for reranking.")
            return []

        # TODO: Wouldn't it make sense to first have all the retrieved documents and only then figure out which are the most important ones? It's easier to compare relatively then absolutely
        # Step 2: Rerank the retrieved documents using the LLM
        logger.debug('\nDebug, helpful to get an idea if the LLM is scoring decently:')
        scored_docs = []
        for doc in initial_retrieved_docs:
            score = await self._get_llm_relevance_score(query, doc.page_content)
            logger.debug("\nDocument content:\n", doc.page_content)
            logger.debug("Relevance Score: ", score)
            scored_docs.append((doc, score))

        # Sort documents by their relevance score in descending order
        reranked_docs_with_scores = sorted(scored_docs, key=itemgetter(1), reverse=True)
        logger.debug(f"Document contents with their scores: {reranked_docs_with_scores}")

        # Extract only the Document objects and return the top_n
        final_reranked_documents = [doc for doc, _ in reranked_docs_with_scores[:final_k]]
        logger.debug(f"\nRerankingRetriever returned {len(final_reranked_documents)} reranked documents.")
        logger.info(f"Final retrieval documents with the highest score: {final_reranked_documents}")

        return final_reranked_documents  # To debug, also return scored_docs

    def build_rag_context(self, documents: list[Document]) -> str:
        """Builds a context string from a list of documents.

        This function iterates through a list of `Document` objects, extracts the
        `page_content` from each, and concatenates them into a single string.
        The resulting string is prefixed with "Retrieval context: " to clearly
        indicate its purpose in a RAG system.

        Args:
            documents (list[Document]): A list of Document objects, each containing
                a page_content attribute that is used to build the context.

        Returns:
            str: A single string representing the combined content of all
                documents, prefixed with "Retrieval context: ".
                Returns an empty string if the input `documents` list is empty.
        """
        if not documents:
            return ""  # Handle empty list case

        # Extract page_content from each document
        # TODO: Write Notion task for this
        document_contents = ['\n'.join([str(doc.metadata), doc.page_content]) for doc in documents]
        context = "\n\n".join(document_contents)

        return f"Context taken from the retrieved documents: {context}"

"""Vector store abstract class for vector store providers."""

from abc import ABC, abstractmethod

# Assuming LangChain's Document will be passed around.
# If you want to completely decouple, you'd define your own DocumentChunk class here as well.
from langchain_core.documents import Document


class VectorStoreInterface(ABC):
    """Abstract Base Class (Interface) for a RAG Vector Store.

    Defines the contract for vector store operations required by the RAG application.
    """

    @abstractmethod
    async def search(self, query: str, top_k: int = 5) -> list[Document]:
        """Performs a similarity search in the vector store and retrieves the
        top_k documents.

        Args:
            query (str): The user's query string.
            top_k (int): The number of top relevant documents to retrieve.
                Defaults to 5.

        Returns:
            list[Document]: A list of Document objects representing the
            retrieved content.
        """
        pass

    # Possible future functions
    # @abstractmethod
    # async def add_documents(self, documents: List[Document]):
    #     """Adds documents to the vector store."""
    #     pass

    # @abstractmethod
    # async def delete_documents(self, document_ids: List[str]):
    #     """Deletes documents by their IDs from the vector store."""
    #     pass

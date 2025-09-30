""""RAG workflow definition."""

from .state import GraphState
from langgraph.graph import StateGraph, END
from functools import partial
from typing import Any
from aletheia_back_end.modules.workflows.rag.nodes import retrieve_node, generate_node
from aletheia_back_end.modules.labs_nlp.llm_client_interface import LLMClientInterface
from aletheia_back_end.modules.labs_nlp.query_processor_interface import QueryProcessor
from aletheia_back_end.modules.labs_search.retriever_interface import RetrieverInterface
from aletheia_back_end.modules.labs_nlp.azure_client import get_llm_client
from aletheia_back_end.modules.labs_nlp.query_processor_interface import QueryRewriter
from aletheia_back_end.modules.labs_search.cosmos_db import get_vector_store
from aletheia_back_end.modules.labs_search.embeddings import get_azure_openai_embeddings, get_openai_embeddings
from aletheia_back_end.modules.labs_search.retriever_interface import RerankingRetriever


def get_rag_app(
    llm_client: LLMClientInterface,
    retriever: RetrieverInterface,
    query_processor: QueryProcessor
) -> Any:
    """Creates and returns a compiled LangGraph workflow for RAG.

    This function sets up the LangGraph workflow by defining its nodes and
    edges, and injecting the necessary dependencies (LLM client, retriever,
    and query processor). It then compiles the graph, ready to be used.

    Args:
        llm_client (LLMClientInterface): An initialized LLM client adhering
            to the `LLMClientInterface`, used for generating responses.
        retriever (RetrieverInterface): An initialized retriever adhering to
            the `RetrieverInterface`, used for fetching relevant documents.
        query_processor (QueryProcessor): An initialized query processor
            adhering to the `QueryProcessor` interface, used for transforming
            the input query.

    Returns:
        Any: A compiled LangGraph application instance, ready to be invoked.
    """
    # Define the graph
    workflow = StateGraph(GraphState)

    # Add nodes using functools.partial to pass the external dependencies
    workflow.add_node(
        "retrieve_node",
        partial(retrieve_node, retriever=retriever, query_processor=query_processor)
    )
    workflow.add_node(
        "generate_node",
        partial(generate_node, llm_client=llm_client)
    )

    # Set up edges
    workflow.add_edge("retrieve_node", "generate_node")
    workflow.add_edge("generate_node", END)

    # Set the entry point
    workflow.set_entry_point("retrieve_node")

    # Compile the graph
    app = workflow.compile()

    return app


# Instantiate the RAG workflow
def get_rag_workflow_app() -> Any:
    llm_client = get_llm_client()
    llm = llm_client.llm
    vector_store = get_vector_store(embedding_model=get_azure_openai_embeddings())
    retriever = RerankingRetriever(vector_store=vector_store, llm=llm)
    query_rewriter = QueryRewriter(llm=llm)
    return get_rag_app(llm_client, retriever, query_rewriter)

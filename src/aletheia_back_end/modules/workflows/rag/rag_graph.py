""""RAG workflow definition."""

from .state import GraphState
from langgraph.graph import StateGraph, END
from functools import partial
from typing import Any
from aletheia_back_end.modules.workflows.rag.nodes import retrieve_node, generate_node
from aletheia_back_end.modules.labs_nlp.llm_client_interface import LLMClientInterface
from aletheia_back_end.modules.labs_nlp.query_processor_interface import QueryProcessor
from aletheia_back_end.modules.labs_search.retriever_interface import RetrieverInterface

from aletheia_back_end.utils.config_builders import (
    load_workflow_core_components_config,
    build_llm_client,
    build_vector_store,
    build_retriever,
    build_query_processor,
)


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
def get_rag_workflow_app(config_path: str = "src/aletheia_back_end/config/rag_workflow_config.yml") -> Any:
    config = load_workflow_core_components_config(config_path)

    # Get the config values from the YAML file
    llm_client = build_llm_client(config["llm"])
    llm = llm_client.llm

    vector_store = build_vector_store(config["vector_store"])
    retriever = build_retriever(config["retriever"], llm, vector_store)
    query_rewriter = build_query_processor(config["query_processor"], llm)

    return get_rag_app(llm_client, retriever, query_rewriter)

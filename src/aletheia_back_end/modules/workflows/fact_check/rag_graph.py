""""RAG workflow definition."""

from .state import GraphState
from langgraph.graph import StateGraph, END
from functools import partial
from typing import Any
from aletheia_back_end.app_settings import settings
from aletheia_back_end.modules.workflows.fact_check.nodes import retrieve_node, generate_node
from aletheia_back_end.modules.labs_nlp.llm_client_interface import LLMClientInterface
from aletheia_back_end.modules.labs_nlp.query_processor_interface import QueryProcessor
from aletheia_back_end.modules.labs_search.retriever_interface import RetrieverInterface

from aletheia_back_end.utils.config_builders import (
    load_workflow_core_components_config,
    load_nodes_config,
    build_llm_client,
    build_vector_store,
    build_retriever,
    build_query_processor,
)


def get_compiled_workflow(
    llm_client: LLMClientInterface,
    retriever: RetrieverInterface,
    query_processor: QueryProcessor,
    prompt: str
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
        partial(generate_node, llm_client=llm_client, prompt=prompt)
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
    components_config = load_workflow_core_components_config(path=settings.fact_check_workflow_config_path)
    nodes_config = load_nodes_config(path=settings.fact_check_workflow_config_path)

    # Get the config values from the YAML file
    llm_client = build_llm_client(config=components_config["llm"])
    llm = llm_client.llm

    vector_store = build_vector_store(config=components_config["vector_store"])
    retriever = build_retriever(config=components_config["retriever"], vector_store=vector_store, llm=llm)
    query_rewriter = build_query_processor(config=components_config["query_processor"], llm=llm)
    generate_node_prompt = nodes_config["generate_node"]["prompt"]

    return get_compiled_workflow(llm_client=llm_client, retriever=retriever, query_processor=query_rewriter, prompt=generate_node_prompt)

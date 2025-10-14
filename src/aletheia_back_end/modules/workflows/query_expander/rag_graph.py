""""RAG workflow definition."""

from .state import GraphState
from langgraph.graph import StateGraph, END
from functools import partial
from typing import Any
from aletheia_back_end.app_settings import settings
from aletheia_back_end.modules.workflows.query_expander.nodes import expand_queries_node, retrieve_node, gather_and_rerank_node, generate_node
from aletheia_back_end.modules.labs_nlp.llm_client_interface import LLMClientInterface
from aletheia_back_end.modules.labs_nlp.query_processor_interface import QueryProcessor
from aletheia_back_end.modules.labs_search.retriever_interface import RetrieverInterface
from aletheia_back_end.modules.labs_search.reranker_interface import RerankerInterface

from aletheia_back_end.utils.config_builders import (
    load_workflow_core_components_config,
    load_nodes_config,
    build_llm_client,
    build_vector_store,
    build_retriever,
    build_query_processor,
    build_reranker
)


def get_compiled_workflow(
    llm_client: LLMClientInterface,
    retriever: RetrieverInterface,
    query_processor: QueryProcessor,
    reranker: RerankerInterface,
    num_queries: int,
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
        num_queries (int): The number of expanded queries to generate.
        prompt (str): The prompt template to be used in the generate node.

    Returns:
        Any: A compiled LangGraph application instance, ready to be invoked.
    """
    # Define the graph
    workflow = StateGraph(GraphState)

    # Add nodes using functools.partial to pass the external dependencies
    workflow.add_node(
        "expand_queries_node",
        partial(expand_queries_node, query_processor=query_processor)
    )

    # For each expanded query, retrieve info from the vector store
    for i in range(num_queries):
        node_name = f"retrieve_branch_{i}"
        workflow.add_node(
            node_name,
            partial(retrieve_node, index=i, retriever=retriever)
        )

    # rerank the different contexts retrieved for each node and gather them into one context. Rerank its importance based on the original query
    workflow.add_node(
        "gather_and_rerank_node",
        partial(gather_and_rerank_node, reranker=reranker)
    )

    # Final node, which generates the output based on the gathered context
    workflow.add_node(
        "generate_node",
        partial(generate_node, llm_client=llm_client, prompt=prompt)
    )

    # Set up edges
    for i in range(3):
        workflow.add_edge("expand_queries_node", f"retrieve_branch_{i}")
        workflow.add_edge(f"retrieve_branch_{i}", "gather_and_rerank_node")
    workflow.add_edge("gather_and_rerank_node", "generate_node")
    workflow.add_edge("generate_node", END)

    # Set the entry point
    workflow.set_entry_point("expand_queries_node")

    # Compile the graph
    app = workflow.compile()

    return app


# Instantiate the RAG workflow
def get_rag_workflow_app() -> Any:
    components_config = load_workflow_core_components_config(settings.query_expander_workflow_config_path)
    nodes_config = load_nodes_config(settings.query_expander_workflow_config_path)

    # Get the config values from the YAML file
    llm_client = build_llm_client(components_config["llm"])
    llm = llm_client.llm

    vector_store = build_vector_store(components_config["vector_store"])
    retriever = build_retriever(components_config["retriever"], vector_store, llm)
    query_processor = build_query_processor(components_config["query_processor"], llm)
    reranker = build_reranker(components_config["reranker"], llm)

    generate_node_prompt = nodes_config["generate_node"]["prompt"]
    num_queries = components_config["query_processor"]["params"]["num_queries"]

    return get_compiled_workflow(llm_client, retriever, query_processor,
                                 reranker, num_queries, generate_node_prompt)

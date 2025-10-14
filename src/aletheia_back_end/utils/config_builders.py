import yaml
from typing import Any, Dict, Optional
from langchain_core.language_models import BaseChatModel

from aletheia_back_end.modules.labs_nlp.azure_client import get_azure_llm_client
from aletheia_back_end.modules.labs_nlp.openai_client import get_openai_llm_client
from aletheia_back_end.modules.labs_search.cosmos_db import get_vector_store
from aletheia_back_end.modules.labs_search.embeddings import (
    get_azure_openai_embeddings,
    get_openai_embeddings,
)

from aletheia_back_end.modules.labs_search.retriever_interface import RetrieverInterface, RerankingRetriever, BasicRetriever
from aletheia_back_end.modules.labs_nlp.query_processor_interface import QueryProcessor, QueryRewriter, QueryExpander
from aletheia_back_end.modules.labs_nlp.llm_client_interface import LLMClientInterface
from aletheia_back_end.modules.labs_search.vector_store_interface import VectorStoreInterface
from aletheia_back_end.modules.labs_search.reranker_interface import RerankerInterface, Reranker


def build_llm_client(config: Dict[str, Any]) -> LLMClientInterface:
    # TODO: In the end pass only **params to the functions, instead of specifying every parameter. This is just for ease of debugging
    params = config.get("params", {})
    temperature = params.get("temperature")
    max_tokens = params.get("max_tokens")
    timeout = params.get("timeout")
    if config["type"] == "AzureOpenAIClient":
        return get_azure_llm_client(temperature, max_tokens, timeout)
    elif config["type"] == "OpenAIClient":
        return get_openai_llm_client(temperature, max_tokens, timeout)
    else:
        raise ValueError(f"Unknown LLM type {config['type']}")


def build_vector_store(config: Dict[str, Any]) -> VectorStoreInterface:
    emb_cfg = config["params"].get("embeddings", {})
    emb_type = emb_cfg.get("type")

    if emb_type == "AzureOpenAIEmbeddings":
        embedding_model = get_azure_openai_embeddings()
    elif emb_type == "OpenAIEmbeddings":
        embedding_model = get_openai_embeddings()
    else:
        raise ValueError(f"Unknown embedding type {emb_type}")

    return get_vector_store(embedding_model=embedding_model)


def build_retriever(config: Dict[str, Any],
                    vector_store: VectorStoreInterface,
                    llm: Optional[BaseChatModel] = None) -> RetrieverInterface:
    params = config.get("params", {})
    top_k = params.get("top_k_initial")
    if config["type"] == "BasicRetriever":
        return BasicRetriever(vector_store, top_k)
    elif config["type"] == "RerankingRetriever":
        top_n_reranked = params.get("top_n_reranked")
        prompt = params.get("prompt")
        return RerankingRetriever(vector_store, llm, prompt, top_k, top_n_reranked)
    else:
        raise ValueError(f"Unknown retriever type {config['type']}")


def build_query_processor(config: Dict[str, Any], llm: BaseChatModel) -> QueryProcessor:
    params = config.get("params", {})
    prompt = params.get("prompt")
    if config["type"] == "QueryExpander":
        num_queries = params.get("num_queries")
        return QueryExpander(llm, prompt, num_queries)
    elif config["type"] == "QueryRewriter":
        return QueryRewriter(llm, prompt)
    else:
        raise ValueError(f"Unknown query processor type {config['type']}")


def build_reranker(config: Dict[str, Any], llm: BaseChatModel) -> RerankerInterface:
    params = config.get("params", {})
    prompt = params.get("prompt")
    return Reranker(llm, prompt)


def load_workflow_core_components_config(path: str) -> Any:
    with open(path, "r") as f:
        return yaml.safe_load(f)["core_components"]


def load_nodes_config(path: str) -> Any:
    with open(path, "r") as f:
        return yaml.safe_load(f)["nodes"]

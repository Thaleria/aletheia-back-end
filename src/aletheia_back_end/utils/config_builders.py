import yaml
from typing import Any, Dict
from langchain_core.language_models import BaseChatModel

from aletheia_back_end.modules.labs_nlp.azure_client import get_azure_llm_client
from aletheia_back_end.modules.labs_nlp.openai_client import get_openai_llm_client
from aletheia_back_end.modules.labs_search.cosmos_db import get_vector_store
from aletheia_back_end.modules.labs_search.embeddings import (
    get_azure_openai_embeddings,
    get_openai_embeddings,
)

from aletheia_back_end.modules.labs_search.retriever_interface import RetrieverInterface, RerankingRetriever
from aletheia_back_end.modules.labs_nlp.query_processor_interface import QueryProcessor, QueryRewriter
from aletheia_back_end.modules.labs_nlp.llm_client_interface import LLMClientInterface
from aletheia_back_end.modules.labs_search.vector_store_interface import VectorStoreInterface


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


def build_retriever(config: Dict[str, Any], llm: BaseChatModel, vector_store: VectorStoreInterface) -> RetrieverInterface:
    params = config.get("params", {})
    prompt = params.get("prompt")
    top_k_initial = params.get("top_k_initial")
    top_n_reranked = params.get("top_n_reranked")
    return RerankingRetriever(vector_store, llm, prompt, top_k_initial, top_n_reranked)


def build_query_processor(config: Dict[str, Any], llm: BaseChatModel) -> QueryProcessor:
    params = config.get("params", {})
    prompt = params.get("prompt")
    return QueryRewriter(llm, prompt)


def load_workflow_core_components_config(path: str) -> Any:
    with open(path, "r") as f:
        return yaml.safe_load(f)["core_components"]


def load_workflows_config(path: str) -> Any:
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    workflows = {wf["name"]: wf for wf in config.get("workflows", [])}

    return workflows

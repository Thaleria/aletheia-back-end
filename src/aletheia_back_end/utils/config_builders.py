import yaml
from pathlib import Path
from typing import Any, Dict

from aletheia_back_end.modules.labs_nlp.azure_client import get_azure_llm_client
from aletheia_back_end.modules.labs_nlp.openai_client import get_openai_llm_client
from aletheia_back_end.modules.labs_search.cosmos_db import get_vector_store
from aletheia_back_end.modules.labs_search.embeddings import (
    get_azure_openai_embeddings,
    get_openai_embeddings,
)
from aletheia_back_end.modules.labs_search.retriever_interface import RerankingRetriever
from aletheia_back_end.modules.labs_nlp.query_processor_interface import QueryRewriter


def build_llm_client(config: Dict[str, Any]):
    params = config.get("params", {})
    if config["type"] == "AzureOpenAIClient":
        return get_azure_llm_client(**params)
    elif config["type"] == "OpenAIClient":
        return get_openai_llm_client(**params)
    else:
        raise ValueError(f"Unknown LLM type {config['type']}")


def build_vector_store(config: Dict[str, Any]):
    emb_cfg = config["params"].get("embeddings", {})
    emb_type = emb_cfg.get("type")

    if emb_type == "AzureOpenAIEmbeddings":
        embedding_model = get_azure_openai_embeddings()
    elif emb_type == "OpenAIEmbeddings":
        embedding_model = get_openai_embeddings()
    else:
        raise ValueError(f"Unknown embedding type {emb_type}")

    return get_vector_store(embedding_model=embedding_model)


def build_retriever(config: Dict[str, Any], llm, vector_store):
    params = config.get("params", {})
    return RerankingRetriever(vector_store=vector_store, llm=llm, **params)


def build_query_processor(config: Dict[str, Any], llm):
    params = config.get("params", {})
    return QueryRewriter(llm=llm, **params)


def load_workflow_config(path: str = "config/rag_workflow_config.yml") -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)["rag_workflow"]

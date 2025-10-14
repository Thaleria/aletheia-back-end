from typing import Any
from aletheia_back_end.app_settings import settings

from aletheia_back_end.modules.workflows.rag.rag_graph import get_rag_workflow_app
from aletheia_back_end.modules.workflows.fact_check.rag_graph import get_rag_workflow_app as get_rag_fact_check_workflow_app
from aletheia_back_end.modules.workflows.consistency_check.rag_graph import get_rag_workflow_app as get_rag_consistency_check_workflow_app


# Define a function to load the prompt, making the logic reusable
def load_prompt_template(file_path: str) -> str:
    """Loads a prompt template from a file located in the same directory."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: Prompt file not found at {file_path}")
        return ""  # Or handle the error as appropriate


def get_config_rag_workflow_app() -> Any:
    active_workflow = settings.active_workflow

    if active_workflow == "rag":
        return get_rag_workflow_app()
    elif active_workflow == "fact_check":
        return get_rag_fact_check_workflow_app()
    elif active_workflow == "consistency_check":
        return get_rag_consistency_check_workflow_app()
    else:
        raise ValueError(f"Unknown workflow name {active_workflow}")

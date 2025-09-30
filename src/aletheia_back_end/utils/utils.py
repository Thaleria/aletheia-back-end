# Define a function to load the prompt, making the logic reusable
def load_prompt_template(file_path: str) -> str:
    """Loads a prompt template from a file located in the same directory."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: Prompt file not found at {file_path}")
        return ""  # Or handle the error as appropriate

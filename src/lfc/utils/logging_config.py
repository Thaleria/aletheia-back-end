import logging
from .config import FAILED_CHUNKS_DIR, OUTPUT_DIR


def setup_dirs() -> None:
    """Create necessary directories if they don't exist."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    FAILED_CHUNKS_DIR.mkdir(exist_ok=True)


def setup_logging() -> None:
    """Configures the application's global logging."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(OUTPUT_DIR / 'transcription.log'),
            logging.StreamHandler()
        ]
    )


def get_configured_logger(name: str) -> logging.Logger:
    """Get a configured logger for a specific module.

    Returns a logger instance that inherits the global configuration set by
    "setup_logging".
    The logger is namespaced to the provided module name.

    Args:
        name (str): The name of the logger, the module's `__name__`.

    Returns:
        logging.Logger: A configured logger instance.
    """
    return logging.getLogger(name)


# Initialize directories and logging when the module is imported
setup_dirs()
setup_logging()

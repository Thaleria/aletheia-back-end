"""Logging configuration."""

import logging
import os
from logging.handlers import RotatingFileHandler

from ckb_back_end.app_settings import settings


def setup_logging() -> None:
    """Configures the application's global logging.

    This function sets up the root logger with handlers and a logging level
    based on the application's global `settings` object. It should be called
    once at the application's entry point to ensure consistent logging across
    all modules.

    The configuration includes:
    - Creating a 'logs' directory at the project root if it doesn't exist.
    - Clearing any pre-existing root logger handlers to prevent duplicate log entries.
    - Setting the logging level (e.g., DEBUG, INFO) from `settings.LOG_LEVEL`.
    - Applying a log format string from `settings.LOG_FORMAT`.
    - Configuring a `RotatingFileHandler` if `settings.LOG_FILE_ENABLED` is True.
      This handler writes logs to a file, with rotation based on size and backup
      count defined in `settings`. This is typically used for local development
      or environments where direct file access is acceptable.
    - Configuring a `StreamHandler` if `settings.LOG_TO_CONSOLE` is True.
      This handler directs logs to standard output/error, which is the
      recommended approach for containerized environments (e.g., Docker, Kubernetes)
      as container orchestration platforms collect these streams.
    """
    # Ensure log directory exists relative to the project root
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )
    log_dir = os.path.join(project_root, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Clear existing handlers to prevent duplicate logs if called multiple times
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Get the desired log level from settings, convert string to logging constant
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    # Create a formatter using the format string from settings
    formatter = logging.Formatter(settings.LOG_FORMAT)

    # List to hold active handlers
    configured_handlers: list[logging.Handler] = []

    # Configure File Handler
    if settings.LOG_FILE_ENABLED:
        log_file_path = os.path.join(log_dir, settings.LOG_FILE_NAME)
        file_handler = RotatingFileHandler(
            filename=log_file_path,
            maxBytes=settings.LOG_FILE_MAX_BYTES,
            backupCount=settings.LOG_FILE_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        configured_handlers.append(file_handler)

    # Configure Stream Handler
    if settings.LOG_TO_CONSOLE:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        configured_handlers.append(stream_handler)

    # Apply the handlers and level to the root logger
    logging.basicConfig(level=log_level, handlers=configured_handlers)

    logging.info(
        f"Logging configured with level: {settings.LOG_LEVEL}, "
        f"file logging: {settings.LOG_FILE_ENABLED}, "
        f"console logging: {settings.LOG_TO_CONSOLE}."
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

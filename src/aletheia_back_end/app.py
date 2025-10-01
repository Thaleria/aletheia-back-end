"""Main entry point for application."""

from fastapi import FastAPI

from aletheia_back_end.api.chat import chat_router
from aletheia_back_end.api.lfc import lfc_router
from aletheia_back_end.utils.logging_config import get_configured_logger

app: FastAPI = FastAPI()
app.include_router(chat_router)
# app.include_router(lfc_router)

if __name__ == "__main__":
    import uvicorn

    try:
        uvicorn.run("app:app", host="127.0.0.1", port=8001, reload=True)
    except Exception as e:
        logger = get_configured_logger(__name__)
        logger.error(f"Failed to start application: {str(e)}")
        raise

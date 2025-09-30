import requests
from httpx import HTTPStatusError
from lfc.utils.logging_config import get_configured_logger
from datetime import datetime
from pydub import AudioSegment
from .config import FAILED_CHUNKS_DIR, OUTPUT_DIR
from typing import Optional, Dict, Any

logger = get_configured_logger(__name__)


def send_to_rag(RAG_API_URL: str, transcribed_text: str) -> Optional[Dict[str, Any]]:
    """Sends a transcribed text chunk to the RAG API for processing.

    This function posts the transcribed text to a RAG (Retrieval-Augmented
    Generation) API endpoint, which is used for fact-checking. The function
    includes a `try-except` block to handle potential network and HTTP status
    errors.

    Args:
        RAG_API_URL (str): The URL of the RAG API endpoint.
        transcribed_text (str): The text chunk to be sent for processing.

    Returns:
        Optional[Dict[str, Any]]: The JSON response from the RAG API if the request
            is successful, otherwise `None` if an error occurs.
    """
    payload = {
        "messages": [
            {"role": "assistant", "content": transcribed_text}
        ],
        "context": {
            "overrides": {
                "top": 5,
                "temperature": 0.5,
                "minimum_reranker_score": 0.0,
                "minimum_search_score": 0.0,
                "retrieval_mode": "hybrid",
                "semantic_ranker": False,
                "semantic_captions": False,
                "suggest_followup_questions": False,
                "use_oid_security_filter": False,
                "use_groups_security_filter": False,
                "vector_fields": [],
                "use_gpt4v": False,
                "gpt4v_input": ""
            }
        }
    }

    try:
        response = requests.post(RAG_API_URL, json=payload)
        response.raise_for_status()  # Raises HTTPStatusError for bad responses
        return response.json()
    except HTTPStatusError as e:
        logger.error(f"HTTP error sending to RAG: {e.response.status_code} - {e.response.text}")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error sending to RAG: {e}")
        return None


def save_failed_chunk(chunk: AudioSegment, chunk_index: int, error: str):
    """Saves a failed audio chunk as a WAV file and logs the error.

    This utility function is for debugging and analysis when something fails.
    It takes an audio chunk that failed to transcribe, saves it as a WAV file,
    and creates a corresponding text file with the error details. The files
    are saved in the directory specified by `FAILED_CHUNKS_DIR`.

    Args:
        chunk (AudioSegment): The `pydub` AudioSegment object representing the
            audio data.
        chunk_index (int): The sequential index of the chunk.
        error (str): The error message associated with the transcription
            failure.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    failed_chunk_path = FAILED_CHUNKS_DIR / f"failed_chunk_{chunk_index}_{timestamp}.wav"
    chunk.export(failed_chunk_path, format="wav")
    error_log_path = FAILED_CHUNKS_DIR / f"failed_chunk_{chunk_index}_{timestamp}.error.txt"
    error_log_path.write_text(f"Error: {error}\nTimestamp: {timestamp}", encoding="utf-8")
    logger.error(f"Saved failed chunk {chunk_index} to {failed_chunk_path}")


def save_chunk(chunk: AudioSegment, chunk_index: int):
    """Saves an audio chunk as a WAV file.

    This utility function is for debugging and analysis to check the
    transcription corresponds to the audio.
    It takes an audio chunk, saves it as a WAV file, and creates a
    corresponding text file with the transcription. The files are saved in the
    directory specified by OUTPUT_DIR.

    Args:
        chunk (AudioSegment): The `pydub` AudioSegment object representing the
            audio data.
        chunk_index (int): The sequential index of the chunk.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    chunk_path = OUTPUT_DIR / f"chunk_{chunk_index}_{timestamp}.wav"
    chunk.export(chunk_path, format="wav") 
    logger.info(f"Saved chunk {chunk_index} to {chunk_path}")

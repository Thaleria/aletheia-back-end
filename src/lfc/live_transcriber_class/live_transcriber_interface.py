"""LLM client abstract class for LLM providers."""

from abc import ABC, abstractmethod
from typing import Any

from lfc.utils.logging_config import get_configured_logger

# Set up logging
logger = get_configured_logger(__name__)


class LiveTranscriberInterface(ABC):
    """Abstract base class defining the interface for a live audio transcriber.

    All concrete transcriber implementations (e.g., Speechmatics, Deepgram) must
    implement the methods defined in this interface.
    """

    @abstractmethod
    def _setup_client(self) -> Any:
        """Initializes and configures the client for the transcription service.

        This method should set up the necessary connection settings and client
        instance for the specific transcription provider.

        Returns:
            Any: The configured client or connection object.
        """
        pass

    @abstractmethod
    def _run_stream(self) -> None:
        """Manages the audio streaming process.

        This method is responsible for capturing audio from a source, such as
        an M3U8 stream, and preparing it for transcription.
        Subclasses should implement the logic to feed the audio into a buffer
        or processing pipeline.
        """
        pass

    @abstractmethod
    def _process_chunk_with_retry(self, chunk: Any, chunk_index: int) -> bool:
        """Processes a single audio chunk with built-in retry logic.

        This method should send the provided audio chunk to the transcription
        service.
        Implementation should include a retry mechanism.

        Args:
            chunk (Any): The audio chunk to be transcribed. The type will
                depend on the specific transcriber's implementation (e.g.,
                `pydub.AudioSegment`).
            chunk_index (int): The sequential index of the audio chunk.

        Returns:
            bool: `True` if the chunk was transcribed successfully, `False`
                otherwise.
        """
        pass

    @abstractmethod
    def _main_transcription_loop(self) -> None:
        """Executes the main transcription loop.

        This method orchestrates the process of reading audio data,
        creating chunks, and calling the `_process_chunk_with_retry` method
        for each chunk. It should handle the continuous flow of transcription.
        """
        pass

    @abstractmethod
    def start(self) -> None:
        """Starts the live transcription process."""
        pass

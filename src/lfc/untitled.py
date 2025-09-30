import io
import os
import time
import threading
import subprocess
import tempfile
import random
import speechmatics
from httpx import HTTPStatusError
from requests import RequestException
from typing import List, Tuple, Any
from pydub import AudioSegment
from lfc.utils.logging_config import get_configured_logger
from lfc.utils.config import (
    M3U8_STREAM_URL, SPEECHMATICS_API_KEY, LANGUAGE, CONNECTION_URL,
    RAG_API_URL, MAX_RETRIES, INITIAL_RETRY_DELAY, MAX_RETRY_DELAY,
    CHUNK_COOLDOWN, CHUNK_BYTES, OUTPUT_DIR
)
from lfc.utils.utils import send_to_rag, save_failed_chunk
from lfc.live_transcriber_class.live_transcriber_interface import LiveTranscriberInterface

logger = get_configured_logger(__name__)


class SpeechmaticsTranscriberSpeakers(LiveTranscriberInterface):
    """Manages the live audio transcription process from an M3U8 stream.

    This class handles the end-to-end process of streaming audio, chunking it,
    transcribing it using the Speechmatics API, and optionally sending the
    transcript to a RAG API for fact-checking.

    Args:
        use_rag (bool): Whether to use the RAG API for fact-checking. Defaults to False.
    """

    def __init__(self, use_rag: bool = False) -> None:
        self.use_rag: bool = use_rag
        self._audio_buffer: io.BytesIO = io.BytesIO()
        self._buffer_lock: threading.Lock = threading.Lock()
        self._current_chunk_transcript: List[Tuple[str, dict[str, str]]] = []
        self._transcript_lock: threading.Lock = threading.Lock()
        self._ws: speechmatics.client.WebsocketClient = self._setup_client()

    def _handle_partial_transcript(self, msg: dict[str, Any]) -> None:
        """Handles and logs partial transcript updates with speaker info."""
        print("msg: ", msg)
        if 'results' in msg:
            with self._transcript_lock:
                transcript_parts = []
                for result in msg['results']:
                    speaker_id = result.get('speaker')  # Extract the speaker
                    content = result.get('alternatives')[0].get('content')  # Get the word
                    if speaker_id and content:
                        transcript_parts.append((f"[{speaker_id}] {content}", "partial"))

                if transcript_parts:
                    self._current_chunk_transcript.extend(transcript_parts)
                    logger.debug(f"Partial: {' '.join([t[0] for t in transcript_parts])}")

    def _handle_full_transcript(self, msg: dict[str, Any]) -> None:
        """Handles and logs final transcript updates with speaker info.

        Args:
            msg (dict[str, Any]): The full transcript message from the
                Speechmatics API.
        """
        # print("Full msg: ", msg)
        if 'results' in msg:
            formatted_segments = []
            with self._transcript_lock:
                previous_speaker = None

                for item in msg['results']:
                    alternatives_dict = item['alternatives'][0]
                    content = alternatives_dict['content']
                    current_speaker = alternatives_dict['speaker']

                    if current_speaker != previous_speaker:  # Gotta make sure that if S1 speaks, then S2 speaks then S1 again, it should be [S1]: ... [S2]: ... [S1]: ...
                        formatted_segments.append([current_speaker, content])
                        previous_speaker = current_speaker
                    else:  # Same speaker, append to the last segment
                        if formatted_segments:
                            formatted_segments[-1][1] += f" {content}"
                        else:
                            # This handles the case where the first word of a new chunk
                            # has the same speaker as the last word of the previous chunk
                            formatted_segments.append([current_speaker, content])

                # Format the final output
                for speaker, text in formatted_segments:
                    formatted_dict = {}
                    formatted_dict[speaker] = text
                    self._current_chunk_transcript.append(("full", formatted_dict))
                    logger.debug(f"Full: {formatted_dict}")
        else:
            logger.warning("Received full transcript message without 'results' field.")

    def _setup_client(self) -> speechmatics.client.WebsocketClient:
        """Initializes and configures the Speechmatics WebSocket client.

        Returns:
            speechmatics.client.WebsocketClient: The configured WebSocket client.
        """
        ws = speechmatics.client.WebsocketClient(
            speechmatics.models.ConnectionSettings(
                url=CONNECTION_URL,
                auth_token=SPEECHMATICS_API_KEY,
            )
        )
        ws.add_event_handler(
            event_name=speechmatics.models.ServerMessageType.AddPartialTranscript,
            event_handler=self._handle_partial_transcript,
        )
        ws.add_event_handler(
            event_name=speechmatics.models.ServerMessageType.AddTranscript,
            event_handler=self._handle_full_transcript,
        )
        return ws

    def _get_complete_transcript_with_speakers(self) -> str:
        """
        Concatenates all full transcripts for the current chunk, preserving speaker turn order.

        Returns:
            str: A single string containing all full transcripts for the chunk,
                each on a new line.
        """
        with self._transcript_lock:
            full_transcripts = []
            previous_speaker = None

            for type_, speakers_content in self._current_chunk_transcript:

                if type_ == "full":
                    for current_speaker, content in speakers_content.items():
                        if current_speaker != previous_speaker:
                            full_transcripts.append([current_speaker, content])
                            previous_speaker = current_speaker
                        else:
                            # Same speaker: append content to the last entry
                            if full_transcripts:
                                previous_content = full_transcripts[-1][1]
                                new_content = f"{previous_content} {content}".strip()
                                full_transcripts[-1][1] = new_content

            # Format the final list of turns into a single string with newlines
            formatted_output = ""
            for speaker, content in full_transcripts:
                formatted_output += f"[{speaker}]: {content}\n"

            return formatted_output.strip()

    def _run_stream(self) -> None:
        """Streams audio from M3U8 to an in-memory buffer using FFmpeg.

        Raises:
            FileNotFoundError: If FFmpeg is not installed or not in the system's PATH.
            IOError: If the FFmpeg stdout pipe cannot be created.
        """
        logger.info(f"Streaming from: {M3U8_STREAM_URL}")
        try:
            process = subprocess.Popen([
                "ffmpeg",
                "-i", M3U8_STREAM_URL,
                "-vn",
                "-acodec", "pcm_s16le",
                "-ac", "1",
                "-ar", "16000",
                "-f", "s16le",
                "-"
            ], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

            # Check if the stdout pipe was successfully created
            if process.stdout is None:
                raise IOError("Failed to create FFmpeg stdout pipe.")

            while True:
                data = process.stdout.read(4096)
                if not data:
                    break
                with self._buffer_lock:
                    self._audio_buffer.write(data)
        except FileNotFoundError:
            logger.critical("FFmpeg is not installed or not in the system's PATH. Please install it to continue.")
            raise
        except Exception as e:
            logger.error(f"FFmpeg stream error: {e}")
        finally:
            if process and process.poll() is None:
                process.terminate()
                process.wait()

    def _process_chunk_with_retry(self, chunk: AudioSegment, chunk_index: int) -> bool:
        """Processes an audio chunk with retry logic for rate limits.

        Args:
            chunk (AudioSegment): The audio segment to process.
            chunk_index (int): The index of the current chunk.

        Returns:
            bool: True if the chunk was successfully processed and transcribed,
                False otherwise.
        """
        retry_count = 0
        while retry_count < MAX_RETRIES:
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_chunk:
                    chunk.export(temp_chunk.name, format="wav")
                    save_failed_chunk(chunk, chunk_index, 'Debugging')

                with open(temp_chunk.name, 'rb') as audio_file:
                    self._current_chunk_transcript = []  # Reset for new chunk
                    self._ws.run_synchronously(
                        audio_file,
                        speechmatics.models.TranscriptionConfig(  # Define transcription parameters
                            language=LANGUAGE,
                            enable_partials=False,
                            max_delay=2,
                            operating_point="enhanced",
                            diarization='speaker',
                            speaker_diarization_config={
                                "speaker_sensitivity": 0.7,
                                # "max_speakers": 4
                                }
                        ),
                        speechmatics.models.AudioSettings(),
                    )

                complete_transcript = self._get_complete_transcript_with_speakers()
                print("transcription: ", complete_transcript)
                if complete_transcript:
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    out_path = OUTPUT_DIR / f"chunk_{chunk_index}_{ts}.txt"
                    out_path.write_text(complete_transcript, encoding="utf-8")
                    logger.info(f"Saved: {out_path.name}")
                    logger.debug(f"Transcript: {complete_transcript}")
                    # print("transcription: ", complete_transcript)  # For real-time monitoring in console

                    if self.use_rag:
                        # Call RAG API right after saving
                        try:
                            logger.debug("DEBUG: complete_transcript: ", complete_transcript)
                            result = send_to_rag(RAG_API_URL, complete_transcript)
                            # result = send_to_rag(RAG_API_URL=RAG_API_URL, transcribed_text="Tell me about back pain.")  # For testing purposes, using a static query
                            logger.info(f"Fact-check result for chunk {chunk_index}: {result}")
                        except RequestException as e:
                            logger.error(f"Network error sending chunk {chunk_index} to RAG API: {e}")
                        except HTTPStatusError as e:
                            logger.error(f"HTTP error sending chunk {chunk_index} to RAG API: {e.response.status_code} - {e.response.text}")
                    return True
                return False

            except Exception as e:
                error_msg = str(e)
                if "No worker can be scheduled" in error_msg:
                    retry_count += 1
                    if retry_count < MAX_RETRIES:
                        delay = min(INITIAL_RETRY_DELAY * (2 ** (retry_count - 1)), MAX_RETRY_DELAY)
                        jitter = random.uniform(0, 0.1 * delay)
                        total_delay = delay + jitter  # Using exponential backoff with jitter is the industry standard for interacting with web APIs to handle rate limits gracefully
                        logger.warning(f"Rate limit hit, retrying in {total_delay:.1f}s (attempt {retry_count}/{MAX_RETRIES})")
                        time.sleep(total_delay)
                        continue
                    else:
                        logger.error(f"Max retries reached for chunk {chunk_index}")
                        save_failed_chunk(chunk, chunk_index, "Max retries reached")
                        return False
                elif isinstance(e, HTTPStatusError) and e.response.status_code == 401:
                    logger.error('Invalid API key - Check your SPEECHMATICS_API_KEY!')
                    return False
                else:
                    logger.error(f"Transcription error on chunk {chunk_index}: {error_msg}")
                    save_failed_chunk(chunk, chunk_index, error_msg)
                    return False
            finally:
                if 'temp_chunk' in locals() and os.path.exists(temp_chunk.name):
                    os.remove(temp_chunk.name)  # Ensure temporary file cleanup
        return False

    def _main_transcription_loop(self) -> None:
        """Main loop for processing the audio stream and transcribing chunks.

        This loop continuously reads from the audio buffer, creates audio chunks,
        and sends them to the transcriber for processing.

        Raises:
            KeyboardInterrupt: When the user presses Ctrl+C to shut down the process.
            Exception: For any other errors during the main loop.
        """
        logger.info("Buffering WAV audio stream and starting transcription process...")

        chunk_index = 0
        while True:
            try:
                time.sleep(1)  # Give time to fill. Prevents busy-looping (CPU waste checking empty buffer)
                with self._buffer_lock:
                    # print("DEBUG: audio_buffer.tell(): ", audio_buffer.tell())
                    # first_size = audio_buffer.tell()

                    # print("DEBUG: audio_buffer.read(): ", audio_buffer.read())

                    # The `if` statement checks if the buffer has accumulated enough data to form a full chunk.
                    # If the current buffer size is less than the required chunk size, it continues the loop,
                    # waiting for more data to be written by the FFmpeg thread.
                    if self._audio_buffer.tell() < CHUNK_BYTES:
                        continue

                    # Move the cursor to the beginning of the buffer.
                    # This is crucial for starting the read operation from the start of the accumulated data.
                    self._audio_buffer.seek(0)

                    # Read the full chunk of data for processing. This reads `chunk_bytes` from the beginning of the buffer.
                    raw_data = self._audio_buffer.read(CHUNK_BYTES)

                    # Read the rest of the data in the buffer. The cursor is now at the end of the first chunk,
                    # so this reads all the remaining, unprocessed data from that point to the end.
                    remaining_data = self._audio_buffer.read()

                    # Create a new, clean `io.BytesIO` buffer to hold the data for the next loop iteration.
                    new_buffer = io.BytesIO()

                    # Write the `remaining_data` to the newly created buffer. This effectively "shifts" the unprocessed data
                    # to the beginning of the new buffer, ready to be added to in the next loop.
                    new_buffer.write(remaining_data)
                    self._audio_buffer = new_buffer

                    # Create an `AudioSegment` object from the `raw_data`. This object is used to
                    # prepare the audio chunk for transcription
                    chunk = AudioSegment(
                        data=raw_data,
                        sample_width=2,  # 16-bit PCM
                        frame_rate=16000,
                        channels=1
                    )

                    # Increment the chunk index to keep track of the current chunk number.
                    chunk_index += 1

                logger.info(f"Processing chunk {chunk_index} ({len(chunk)}ms)")

                if not self._process_chunk_with_retry(chunk, chunk_index):
                    logger.warning("Skipping to next chunk...")
                    continue

                logger.debug(f"Waiting {CHUNK_COOLDOWN}s before next chunk...")
                time.sleep(CHUNK_COOLDOWN)

            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt. Shutting down...")
                break
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                time.sleep(5)

    def start(self) -> None:
        """Starts the live transcription process.

        This method launches the FFmpeg stream in a separate thread to
        continuously fill the audio buffer and then starts the main
        transcription loop to process the buffered audio.
        """
        ffmpeg_thread = threading.Thread(target=self._run_stream, daemon=True)
        ffmpeg_thread.start()
        self._main_transcription_loop()


class SpeechmaticsTranscriber(LiveTranscriberInterface):
    """Manages the live audio transcription process from an M3U8 stream.

    This class handles the end-to-end process of streaming audio, chunking it,
    transcribing it using the Speechmatics API, and optionally sending the
    transcript to a RAG API for fact-checking.

    Args:
        use_rag (bool): Whether to use the RAG API for fact-checking. Defaults to False.
    """

    def __init__(self, use_rag: bool = False) -> None:
        self.use_rag: bool = use_rag
        self._audio_buffer: io.BytesIO = io.BytesIO()
        self._buffer_lock: threading.Lock = threading.Lock()
        self._current_chunk_transcript: List[Tuple[str, str]] = []
        self._transcript_lock: threading.Lock = threading.Lock()
        self._ws: speechmatics.client.WebsocketClient = self._setup_client()

    def _handle_partial_transcript(self, msg: dict[str, Any]) -> None:
        """Handles and logs partial transcript updates.

        Args:
            msg (dict[str, Any]): The partial transcript message from the
                Speechmatics API.
        """
        with self._transcript_lock:
            self._current_chunk_transcript.append(("partial", msg['metadata']['transcript']))
        logger.debug(f"Partial: {msg['metadata']['transcript']}")

    def _handle_full_transcript(self, msg: dict[str, Any]) -> None:
        """Handles and logs final transcript updates.

        Args:
            msg (dict[str, Any]): The full transcript message from the
                Speechmatics API.
        """
        with self._transcript_lock:
            self._current_chunk_transcript.append(("full", msg['metadata']['transcript']))
        logger.debug(f"Full: {msg['metadata']['transcript']}")

    def _setup_client(self) -> speechmatics.client.WebsocketClient:
        """Initializes and configures the Speechmatics WebSocket client.

        Returns:
            speechmatics.client.WebsocketClient: The configured WebSocket client.
        """
        ws = speechmatics.client.WebsocketClient(
            speechmatics.models.ConnectionSettings(
                url=CONNECTION_URL,
                auth_token=SPEECHMATICS_API_KEY,
            )
        )
        ws.add_event_handler(
            event_name=speechmatics.models.ServerMessageType.AddPartialTranscript,
            event_handler=self._handle_partial_transcript,
        )
        ws.add_event_handler(
            event_name=speechmatics.models.ServerMessageType.AddTranscript,
            event_handler=self._handle_full_transcript,
        )
        return ws

    def _get_complete_transcript(self) -> str:
        """Concatenates all full transcripts for the current chunk.

        Returns:
            str: A single string containing all full transcripts for the chunk,
                separated by spaces.
        """
        with self._transcript_lock:
            full_transcripts = [text for type_, text in self._current_chunk_transcript if type_ == "full"]
            return " ".join(full_transcripts)

    def _run_stream(self) -> None:
        """Streams audio from M3U8 to an in-memory buffer using FFmpeg.

        Raises:
            FileNotFoundError: If FFmpeg is not installed or not in the system's PATH.
            IOError: If the FFmpeg stdout pipe cannot be created.
        """
        logger.info(f"Streaming from: {M3U8_STREAM_URL}")
        try:
            process = subprocess.Popen([
                "ffmpeg",
                "-i", M3U8_STREAM_URL,
                "-vn",
                "-acodec", "pcm_s16le",
                "-ac", "1",
                "-ar", "16000",
                "-f", "s16le",
                "-"
            ], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

            # Check if the stdout pipe was successfully created
            if process.stdout is None:
                raise IOError("Failed to create FFmpeg stdout pipe.")

            while True:
                data = process.stdout.read(4096)
                if not data:
                    break
                with self._buffer_lock:
                    self._audio_buffer.write(data)
        except FileNotFoundError:
            logger.critical("FFmpeg is not installed or not in the system's PATH. Please install it to continue.")
            raise
        except Exception as e:
            logger.error(f"FFmpeg stream error: {e}")
        finally:
            if process and process.poll() is None:
                process.terminate()
                process.wait()

    def _process_chunk_with_retry(self, chunk: AudioSegment, chunk_index: int) -> bool:
        """Processes an audio chunk with retry logic for rate limits.

        Args:
            chunk (AudioSegment): The audio segment to process.
            chunk_index (int): The index of the current chunk.

        Returns:
            bool: True if the chunk was successfully processed and transcribed,
                False otherwise.
        """
        retry_count = 0
        while retry_count < MAX_RETRIES:
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_chunk:
                    chunk.export(temp_chunk.name, format="wav")

                with open(temp_chunk.name, 'rb') as audio_file:
                    self._current_chunk_transcript = []  # Reset for new chunk
                    self._ws.run_synchronously(
                        audio_file,
                        speechmatics.models.TranscriptionConfig(  # Define transcription parameters
                            language=LANGUAGE,
                            enable_partials=False,
                            max_delay=2,
                            operating_point="enhanced",
                        ),
                        speechmatics.models.AudioSettings(),
                    )

                complete_transcript = self._get_complete_transcript()
                if complete_transcript:
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    out_path = OUTPUT_DIR / f"chunk_{chunk_index}_{ts}.txt"
                    out_path.write_text(complete_transcript, encoding="utf-8")
                    logger.info(f"Saved: {out_path.name}")
                    logger.debug(f"Transcript: {complete_transcript}")

                    if self.use_rag:
                        # Call RAG API right after saving
                        try:
                            print("DEBUG: complete_transcript: ", complete_transcript)
                            result = send_to_rag(RAG_API_URL, complete_transcript)
                            # result = send_to_rag(RAG_API_URL=RAG_API_URL, transcribed_text="Tell me about back pain.")  # For testing purposes, using a static query
                            logger.info(f"Fact-check result for chunk {chunk_index}: {result}")
                        except RequestException as e:
                            logger.error(f"Network error sending chunk {chunk_index} to RAG API: {e}")
                        except HTTPStatusError as e:
                            logger.error(f"HTTP error sending chunk {chunk_index} to RAG API: {e.response.status_code} - {e.response.text}")
                    return True
                return False

            except Exception as e:
                error_msg = str(e)
                if "No worker can be scheduled" in error_msg:
                    retry_count += 1
                    if retry_count < MAX_RETRIES:
                        delay = min(INITIAL_RETRY_DELAY * (2 ** (retry_count - 1)), MAX_RETRY_DELAY)
                        jitter = random.uniform(0, 0.1 * delay)
                        total_delay = delay + jitter  # Using exponential backoff with jitter is the industry standard for interacting with web APIs to handle rate limits gracefully
                        logger.warning(f"Rate limit hit, retrying in {total_delay:.1f}s (attempt {retry_count}/{MAX_RETRIES})")
                        time.sleep(total_delay)
                        continue
                    else:
                        logger.error(f"Max retries reached for chunk {chunk_index}")
                        save_failed_chunk(chunk, chunk_index, "Max retries reached")
                        return False
                elif isinstance(e, HTTPStatusError) and e.response.status_code == 401:
                    logger.error('Invalid API key - Check your SPEECHMATICS_API_KEY!')
                    return False
                else:
                    logger.error(f"Transcription error on chunk {chunk_index}: {error_msg}")
                    save_failed_chunk(chunk, chunk_index, error_msg)
                    return False
            finally:
                if 'temp_chunk' in locals() and os.path.exists(temp_chunk.name):
                    os.remove(temp_chunk.name)  # Ensure temporary file cleanup
        return False

    def _main_transcription_loop(self) -> None:
        """Main loop for processing the audio stream and transcribing chunks.

        This loop continuously reads from the audio buffer, creates audio chunks,
        and sends them to the transcriber for processing.

        Raises:
            KeyboardInterrupt: When the user presses Ctrl+C to shut down the process.
            Exception: For any other errors during the main loop.
        """
        logger.info("Buffering WAV audio stream and starting transcription process...")

        chunk_index = 0
        while True:
            try:
                time.sleep(1)  # Give time to fill. Prevents busy-looping (CPU waste checking empty buffer)
                with self._buffer_lock:
                    # print("DEBUG: audio_buffer.tell(): ", audio_buffer.tell())
                    # first_size = audio_buffer.tell()

                    # print("DEBUG: audio_buffer.read(): ", audio_buffer.read())

                    # The `if` statement checks if the buffer has accumulated enough data to form a full chunk.
                    # If the current buffer size is less than the required chunk size, it continues the loop,
                    # waiting for more data to be written by the FFmpeg thread.
                    if self._audio_buffer.tell() < CHUNK_BYTES:
                        continue

                    # Move the cursor to the beginning of the buffer.
                    # This is crucial for starting the read operation from the start of the accumulated data.
                    self._audio_buffer.seek(0)

                    # Read the full chunk of data for processing. This reads `chunk_bytes` from the beginning of the buffer.
                    raw_data = self._audio_buffer.read(CHUNK_BYTES)

                    # Read the rest of the data in the buffer. The cursor is now at the end of the first chunk,
                    # so this reads all the remaining, unprocessed data from that point to the end.
                    remaining_data = self._audio_buffer.read()

                    # Create a new, clean `io.BytesIO` buffer to hold the data for the next loop iteration.
                    new_buffer = io.BytesIO()

                    # Write the `remaining_data` to the newly created buffer. This effectively "shifts" the unprocessed data
                    # to the beginning of the new buffer, ready to be added to in the next loop.
                    new_buffer.write(remaining_data)
                    self._audio_buffer = new_buffer

                    # Create an `AudioSegment` object from the `raw_data`. This object is used to
                    # prepare the audio chunk for transcription
                    chunk = AudioSegment(
                        data=raw_data,
                        sample_width=2,  # 16-bit PCM
                        frame_rate=16000,
                        channels=1
                    )

                    # Increment the chunk index to keep track of the current chunk number.
                    chunk_index += 1

                logger.info(f"Processing chunk {chunk_index} ({len(chunk)}ms)")

                if not self._process_chunk_with_retry(chunk, chunk_index):
                    logger.warning("Skipping to next chunk...")
                    continue

                logger.debug(f"Waiting {CHUNK_COOLDOWN}s before next chunk...")
                time.sleep(CHUNK_COOLDOWN)

            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt. Shutting down...")
                break
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                time.sleep(5)

    def start(self) -> None:
        """Starts the live transcription process.

        This method launches the FFmpeg stream in a separate thread to
        continuously fill the audio buffer and then starts the main
        transcription loop to process the buffered audio.
        """
        ffmpeg_thread = threading.Thread(target=self._run_stream, daemon=True)
        ffmpeg_thread.start()
        self._main_transcription_loop()

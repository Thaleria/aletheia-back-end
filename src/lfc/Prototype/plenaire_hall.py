import os
import sys
# This ways it can run uv run lfc.Prototype.BBC_live
# Other way of running without defining the abs path is python -m lfc.Prototype.BBC_live
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import subprocess
import tempfile
import json
import requests
from pydub import AudioSegment
import speechmatics
from httpx import HTTPStatusError
import time
import threading
import io
import random
from datetime import datetime
from typing import List, Tuple
from lfc.utils.logging_config import setup_logging, get_configured_logger, OUTPUT_DIR
from app_settings import settings

# Set up logging
setup_logging()
logger = get_configured_logger(__name__)

# === CONFIGURATION ===
M3U8_STREAM_URL = "http://127.0.0.1:8000/master.m3u8"
CHUNK_DURATION_SEC = 40  # Audio is processed in 40-second chunks
OVERLAP_SEC = 1  # A 1-second overlap between chunks ensures continuity in transcription
SPEECHMATICS_API_KEY = settings.speechmatics_api_key
LANGUAGE = "en"
CONNECTION_URL = "wss://eu2.rt.speechmatics.com/v2"  # WebSocket URL for Speechmatics' real-time transcription API.
RAG_API_URL = "http://127.0.0.1:8002/v1/chat"

# Retry configuration
MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 2  # seconds
MAX_RETRY_DELAY = 32  # seconds
CHUNK_COOLDOWN = 3  # seconds to wait between chunks

if not SPEECHMATICS_API_KEY:
    raise EnvironmentError("SPEECHMATICS_API_KEY environment variable not set")

# Create a transcription client
ws = speechmatics.client.WebsocketClient(
    speechmatics.models.ConnectionSettings(
        url=CONNECTION_URL,
        auth_token=SPEECHMATICS_API_KEY,
    )
)

# === SHARED STATE ===
audio_buffer = io.BytesIO()  # in-memory buffer (BytesIO) to store raw audio data from FFmpeg.
buffer_lock = threading.Lock()  # threading lock to prevent concurrent access to audio_buffer
current_chunk_transcript: List[Tuple[str, str]] = []
transcript_lock = threading.Lock()  # Thread-safe updates to current_chunk_transcript
failed_chunks_dir = OUTPUT_DIR / "failed_chunks"  # Directory to save chunks that failed transcription
failed_chunks_dir.mkdir(exist_ok=True) 


def save_failed_chunk(chunk: AudioSegment, chunk_index: int, error: str):
    """Save failed chunk for later processing.

    Saves a failed audio chunk as a WAV file and logs the error in a text file.
    Uses a timestamp to ensure unique filenames.
    Helps debug transcription failures by preserving problematic audio."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    failed_chunk_path = failed_chunks_dir / f"failed_chunk_{chunk_index}_{timestamp}.wav"
    chunk.export(failed_chunk_path, format="wav")
    error_log_path = failed_chunks_dir / f"failed_chunk_{chunk_index}_{timestamp}.error.txt"
    error_log_path.write_text(f"Error: {error}\nTimestamp: {timestamp}", encoding="utf-8")
    logger.error(f"Saved failed chunk {chunk_index} to {failed_chunk_path}")


# Partial transcripts provide real-time feedback, while full transcripts are finalized, more accurate versions
def print_partial_transcript(msg):
    """Handle partial transcript updates."""
    with transcript_lock:
        current_chunk_transcript.append(("partial", msg['metadata']['transcript']))
    logger.debug(f"Partial: {msg['metadata']['transcript']}")


def print_transcript(msg):
    """Handle final transcript updates."""
    with transcript_lock:
        current_chunk_transcript.append(("full", msg['metadata']['transcript']))
    logger.debug(f"Full: {msg['metadata']['transcript']}")


# Register the event handlers
ws.add_event_handler(
    event_name=speechmatics.models.ServerMessageType.AddPartialTranscript,
    event_handler=print_partial_transcript,
)
ws.add_event_handler(
    event_name=speechmatics.models.ServerMessageType.AddTranscript,
    event_handler=print_transcript,
)


def run_ffmpeg_stream():
    """Run FFmpeg to stream audio from M3U8 to WAV format with specific settings (mono, 16kHz, PCM encoding)."""
    logger.info(f"Streaming from: {M3U8_STREAM_URL}")
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

    while True:
        data = process.stdout.read(4096)  # Reads FFmpeg output in 4KB chunks
        if not data:
            break
        with buffer_lock:
            audio_buffer.write(data)


def get_complete_transcript() -> str:
    """Get the complete transcript by concatenating all full transcripts."""
    with transcript_lock:
        full_transcripts = [text for type_, text in current_chunk_transcript if type_ == "full"]
        return " ".join(full_transcripts)


def send_to_rag(transcribed_text: str):
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

    response = requests.post(RAG_API_URL, json=payload)
    return response.json()


def process_chunk_with_retry(chunk: AudioSegment, chunk_index: int, conf, settings, use_rag=False) -> bool:
    """Process a chunk with retry logic."""
    retry_count = 0
    while retry_count < MAX_RETRIES:
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav") as temp_chunk:
                chunk.export(temp_chunk.name, format="wav")

                # Reset transcript list for new chunk
                global current_chunk_transcript
                current_chunk_transcript = []

                with open(temp_chunk.name, 'rb') as audio_file:
                    ws.run_synchronously(audio_file, conf, settings)  # Sends the chunk to Speechmatics via ws.run_synchronously

                # Get and save the complete transcript
                complete_transcript = get_complete_transcript()
                if complete_transcript:
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    out_path = OUTPUT_DIR / f"chunk_{chunk_index}_{ts}.txt"
                    out_path.write_text(complete_transcript, encoding="utf-8")
                    logger.info(f"Saved: {out_path.name}")
                    logger.debug(f"Transcript: {complete_transcript}")

                    if use_rag:
                         # Call RAG API right after saving
                        try:
                            print("DEBUG: complete_transcript", complete_transcript)
                            # result = send_to_rag(complete_transcript)
                            result = send_to_rag("Tell me about back pain.")  # For testing purposes, using a static query
                            logger.info(f"Fact-check result for chunk {chunk_index}: {result}")
                        except Exception as e:
                            logger.error(f"Error sending chunk {chunk_index} to RAG API: {e}")

                    return True
                else:
                    logger.warning(f"No transcript generated for chunk {chunk_index}")
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
                    save_failed_chunk(chunk, chunk_index, "Max retries reached")  # If a chunk fails repeatedly, it's saved to the failed_chunks_dir for later analysis, 
                    return False
            elif isinstance(e, HTTPStatusError) and e.response.status_code == 401:
                logger.error('Invalid API key - Check your SPEECHMATICS_API_KEY!')
                return False
            else:
                logger.error(f"Transcription error on chunk {chunk_index}: {error_msg}")
                save_failed_chunk(chunk, chunk_index, error_msg)
                return False
    return False


def stream_and_transcribe(use_rag=False):
    """Process the audio stream using Speechmatics with overlapping chunks."""
    logger.info("Buffering WAV audio stream...")

    # Define transcription parameters
    conf = speechmatics.models.TranscriptionConfig(
        language=LANGUAGE,
        enable_partials=False,
        max_delay=2,
        operating_point="enhanced",
        additional_vocab=[
            {
                "content": "Max Aardema",
                "sounds_like": [
                    "Max Arma",
                    "Max Adema",
                ]
            },
            {
                "content": "Reinder Blaauw",
                "sounds_like": [
                    "Rijn de Blauw",
                    "Reinder Blaauw",
                    "Reinder Blauw"
                ]
            },
            {
                "content": "Maikel Boon",
                "sounds_like": [
                    "Michael Boon",
                    "Michael Boon",
                    "Michael Boon"
                ]
            },
            {
                "content": "Vincent van den Born",
                "sounds_like": [
                    "Vincent van der Boorn",
                    "Vincent van den Born",
                    "Vincent van der Born"
                ]
            },
            {
                "content": "Martin Bosma",
                "sounds_like": [
                    "Martin Bosma",
                    "Martin Bosma",
                    "Martin Bosma"
                ]
            },
            {
                "content": "Willem Boutkan",
                "sounds_like": [
                    "Willem Boutkan",
                    "Willem Boutkan",
                    "Willem Boutkan"
                ]
            },
            {
                "content": "René Claassen",
                "sounds_like": [
                    "René Klaassen",
                    "René Klaassen",
                    "René Claassen"
                ]
            },
            {
                "content": "Patrick Crijns",
                "sounds_like": [
                    "Patrick Krijns",
                    "Patrick Crijns",
                    "Patrick Krijns"
                ]
            },
            {
                "content": "Marco Deen",
                "sounds_like": [
                    "Marco Deen",
                    "Marco Deen",
                    "Marco Deen"
                ]
            },
            {
                "content": "Tony van Dijck",
                "sounds_like": [
                    "Tony van Dijk",
                    "Tony Van Dijck",
                    "Tony van Dijk"
                ]
            },
            {
                "content": "Emiel van Dijk",
                "sounds_like": [
                    "Emiel van Dijk",
                    "Emiel van Dijk",
                    "Emile van Dijk"
                ]
            },
            {
                "content": "Eric Esser",
                "sounds_like": [
                    "Erik Esser",
                    "Eric Esser",
                    "Erik Esser"
                ]
            },
            {
                "content": "Chris Faddegon",
                "sounds_like": [
                    "Chris van der Gon",
                    "Chris van der Gon",
                    "Chris van de Gon"
                ]
            },
            {
                "content": "Dion Graus",
                "sounds_like": [
                    "Dion Graus",
                    "Dion Graus",
                    "Dion Graus"
                ]
            },
            {
                "content": "Peter van Haasen",
                "sounds_like": [
                    "Peter van Hazen",
                    "Peter van Hazen",
                    "Peter van Haaßen"
                ]
            },
            {
                "content": "Hidde Heutink",
                "sounds_like": [
                    "Hidde Heutink",
                    "Hidde Heutink",
                    "Hidde Heutink"
                ]
            },
            {
                "content": "Patrick van der Hoeff",
                "sounds_like": [
                    "Patrick van der Hoef",
                    "Patrick van der Hoef",
                    "Patrick van der Hoef"
                ]
            },
            {
                "content": "Léon de Jong",
                "sounds_like": [
                    "Leon de Jong",
                    "Leon de Jong",
                    "Leon de Jong"
                ]
            },
            {
                "content": "Alexander Kops",
                "sounds_like": [
                    "Alexander Kops",
                    "Alexander Kops",
                    "Alexander Kops"
                ]
            },
            {
                "content": "Gidi Markuszower",
                "sounds_like": [
                    "Gidi Marcus Zower",
                    "Gidi Markuszower",
                    "Gidi Marks-Zoër"
                ]
            },
            {
                "content": "Rachel van Meetelen",
                "sounds_like": [
                    "Rachel van Metelen",
                    "Rachel van Metalen",
                    "Rachel van Metelen"
                ]
            },
            {
                "content": "Jeremy Mooiman",
                "sounds_like": [
                    "Jeremy Mooiman",
                    "Jeremy Mooijman",
                    "Jeremy Mooiman"
                ]
            },
            {
                "content": "Edgar Mulder",
                "sounds_like": [
                    "Edgar Mulder",
                    "Edgar Mulder",
                    "Edgar Mulder"
                ]
            },
            {
                "content": "Jeanet Nijhof-Leeuw",
                "sounds_like": [
                    "Jeannette Nijhoff-Leeuw",
                    "Jeannette Nijhof",
                    "Jeanet Nijhoff Leeuw"
                ]
            },
            {
                "content": "Joeri Pool",
                "sounds_like": [
                    "Joeri Pol",
                    "Louis Paul",
                    "Joeri Paul"
                ]
            },
            {
                "content": "Dennis Ram",
                "sounds_like": [
                    "Dennis Ram",
                    "Dennis RAM",
                    "Dennis Ram"
                ]
            },
            {
                "content": "Robert Rep",
                "sounds_like": [
                    "Robert Reb",
                    "Robert Rep",
                    "Robert Rep"
                ]
            },
            {
                "content": "Raymond de Roon",
                "sounds_like": [
                    "Raymond de Roon",
                    "Raymond de Roon",
                    "Raymond de Roon"
                ]
            },
            {
                "content": "Peter Smitskam",
                "sounds_like": [
                    "Peter Smits Kam",
                    "Peter SmitsKan",
                    "Peter Smits-Scham"
                ]
            },
            {
                "content": "Folkert Thiadens",
                "sounds_like": [
                    "Volkert Thiadens",
                    "Folkert Tielens",
                    "Folkert Thiadens"
                ]
            },
            {
                "content": "Nico Uppelschoten",
                "sounds_like": [
                    "Nico Uppelschoten",
                    "Nico Opgeschoten",
                    "Nico Uppelschote"
                ]
            },
            {
                "content": "Jan Valize",
                "sounds_like": [
                    "Jan Valize",
                    "Jan Valiezen",
                    "Jan Vanliezen"
                ]
            },
            {
                "content": "Martine van der Velde",
                "sounds_like": [
                    "Martine van der Velde",
                    "Martine van der Velde",
                    "Martine van der Velden"
                ]
            },
            {
                "content": "Elmar Vlottes",
                "sounds_like": [
                    "Elmar Vlottes",
                    "Elmar Vlotjes",
                    "Elmar Vlotters"
                ]
            },
            {
                "content": "Marina Vondeling",
                "sounds_like": [
                    "Marina Vondeling",
                    "Marina Vondeling",
                    "Marina Vondeling"
                ]
            },
            {
                "content": "Henk de Vree",
                "sounds_like": [
                    "Henk de Vree",
                    "Henk de Vree",
                    "Henk de Vree"
                ]
            },
            {
                "content": "Geert Wilders",
                "sounds_like": [
                    "Geert Wilders",
                    "Geert Wilders",
                    "Geert Wilders"
                ]
            },
            {
                "content": "Laura Bromet",
                "sounds_like": [
                    "Laura Bromet",
                    "Laura Bromet",
                    "Laura Bromet"
                ]
            },
            {
                "content": "Julian Bushoff",
                "sounds_like": [
                    "Julian Boeshof",
                    "Julian Bush",
                    "Julian Boshoff"
                ]
            },
            {
                "content": "Glimina Chakor",
                "sounds_like": [
                    "Glimina Chakor",
                    "Glimina Chakor",
                    "Glimina Shakur"
                ]
            },
            {
                "content": "Geert Gabriëls",
                "sounds_like": [
                    "Geert Gabriels",
                    "Geert Gabriëls",
                    "Geert Gabriëls"
                ]
            },
            {
                "content": "Marleen Haage",
                "sounds_like": [
                    "Marleen Hagen",
                    "Marleen Haage",
                    "Marleen Hagen"
                ]
            },
            {
                "content": "Daniëlle Hirsch",
                "sounds_like": [
                    "Daniele Hirsch",
                    "Daniëlle Hirsch",
                    "Daniëlle Hirsch Habtamu"
                ]
            },
            {
                "content": "Habtamu de Hoop",
                "sounds_like": [
                    "Habtamu De Hoop",
                    "HAPTammo de Hoop",
                    "De Hoop"
                ]
            },
            {
                "content": "Barbara Kathmann",
                "sounds_like": [
                    "Barbara Katman",
                    "Barbara Kathmann",
                    "Barbara Kathmann"
                ]
            },
            {
                "content": "Jesse Klaver",
                "sounds_like": [
                    "Jesse Klaver",
                    "Jesse Klaver",
                    "Jesse Klaver"
                ]
            },
            {
                "content": "Suzanne Kröger",
                "sounds_like": [
                    "Suzanne Kreuger",
                    "Suzanne Kröger",
                    "Suzanne Kreuger"
                ]
            },
            {
                "content": "Esmah Lahlah",
                "sounds_like": [
                    "Esma Lachla",
                    "Esma Laila",
                    "Asmalachla"
                ]
            },
            {
                "content": "Tom van der Lee",
                "sounds_like": [
                    "Tom van der Lee",
                    "Tom van der Lee",
                    "Tom van de Lee"
                ]
            },
            {
                "content": "Mohammed Mohandis",
                "sounds_like": [
                    "Mohamed Mohandis",
                    "Mohammed Mohandis",
                    "Mohammed Mohandis"
                ]
            },
            {
                "content": "Songül Mutluer",
                "sounds_like": [
                    "Songul Mutluer",
                    "Song Gül Mutlu",
                    "Songül Mutluer"
                ]
            },
            {
                "content": "Jimme Nordkamp",
                "sounds_like": [
                    "Jimme Noordkamp",
                    "Yme Noord Kamp",
                    "Jimme Noordkamp"
                ]
            },
            {
                "content": "Mariëtte Patijn",
                "sounds_like": [
                    "Mariette Patijn",
                    "Mariëtte Patijn",
                    "Mariëtte Patijn"
                ]
            },
            {
                "content": "Anita Pijpelink",
                "sounds_like": [
                    "Anita Pijperlink",
                    "Anita Pijper Link",
                    "Anita Pijpelink"
                ]
            },
            {
                "content": "Kati Piri",
                "sounds_like": [
                    "Kati Piri",
                    "Kati Piri",
                    "Kati Piri"
                ]
            },
            {
                "content": "Elke Slagt-Tichelman",
                "sounds_like": [
                    "Elke Slag Tichelman",
                    "Elke Slagter Homan",
                    "Elkeslag Tiggelman"
                ]
            },
            {
                "content": "Luc Stultiens",
                "sounds_like": [
                    "Luc Stiltiens",
                    "Luuk Stultiens",
                    "Luuk Stultiens"
                ]
            },
            {
                "content": "Joris Thijssen",
                "sounds_like": [
                    "Joris Thijssen",
                    "Joris Thijssen",
                    "Joris Thijssen"
                ]
            },
            {
                "content": "Frans Timmermans",
                "sounds_like": [
                    "Frans Timmermans",
                    "Frans Timmermans",
                    "Frans Timmermans"
                ]
            },
            {
                "content": "Mikal Tseggai",
                "sounds_like": [
                    "Mikael Tjegai",
                    "Michal Chagall",
                    "Mikal Csegay"
                ]
            },
            {
                "content": "Lisa Westerveld",
                "sounds_like": [
                    "Lisa Westerveld",
                    "Lisa Westerveld",
                    "Lisa Westerveld"
                ]
            },
            {
                "content": "Raoul White",
                "sounds_like": [
                    "Raoul White",
                    "Raoul White",
                    "Raoul White"
                ]
            },
            {
                "content": "Thierry Aartsen",
                "sounds_like": [
                    "Thierry Aertsen",
                    "Thierry Aartsen",
                    "Thierry Aartsen"
                ]
            },
            {
                "content": "Bente Becker",
                "sounds_like": [
                    "Bente Bekker",
                    "Bente Becker",
                    "Benthe Bekker"
                ]
            },
            {
                "content": "Harry Bevers",
                "sounds_like": [
                    "Harry Bevers",
                    "Harry Bevers",
                    "Harry Bevers"
                ]
            },
            {
                "content": "Bart Bikkers",
                "sounds_like": [
                    "Bart Pickers",
                    "Bart Pickers",
                    "Bart Bikkers"
                ]
            },
            {
                "content": "Martijn Buijsse",
                "sounds_like": [
                    "Martijn Buijsen",
                    "Martijn Buijze",
                    "Martijn Buisse"
                ]
            },
            {
                "content": "Eric van der Burg",
                "sounds_like": [
                    "Erik van den Burg",
                    "Eric van der Burg",
                    "Eric van den Burg"
                ]
            },
            {
                "content": "Thom van Campen",
                "sounds_like": [
                    "Tom van Kampen",
                    "Thom van Campen",
                    "Tom van Campen"
                ]
            },
            {
                "content": "Rosemarijn Dral",
                "sounds_like": [
                    "Roze Marijn Dral",
                    "Rozemarijn Bral",
                    "Roosmarijn Dral"
                ]
            },
            {
                "content": "Wendy van Eijk",
                "sounds_like": [
                    "Wendy van Eyck",
                    "Wendy van Eijk",
                    "Wendy van Eyk"
                ]
            },
            {
                "content": "Ulysse Ellian",
                "sounds_like": [
                    "Ulysse Elian",
                    "Ulysse Ellian",
                    "Ulysse Ellian"
                ]
            },
            {
                "content": "Silvio Erkens",
                "sounds_like": [
                    "Silvio Erkens",
                    "Silvio Erkens",
                    "Silvio Erkens"
                ]
            },
            {
                "content": "Peter de Groot",
                "sounds_like": [
                    "Peter de Groot",
                    "Peter Groot",
                    "Peter Groot"
                ]
            },
            {
                "content": "Arend Kisteman",
                "sounds_like": [
                    "Arend Kisteman",
                    "Arend Kistje Man",
                    "Arend Kisteman"
                ]
            },
            {
                "content": "Daan de Kort",
                "sounds_like": [
                    "Daan de Kort",
                    "Daan de Kort",
                    "Daan de Kort"
                ]
            },
            {
                "content": "Claire Martens-America",
                "sounds_like": [
                    "Claire Martensmerika",
                    "Claire Martens Marika",
                    "Claire Martens-America"
                ]
            },
            {
                "content": "Wim Meulenkamp",
                "sounds_like": [
                    "Wim Meulenkamp",
                    "Wim Molenkamp",
                    "Wim Meulenkamp"
                ]
            },
            {
                "content": "Ingrid Michon-Derkzen",
                "sounds_like": [
                    "Ingrid Michon Derksen",
                    "Ingrid Michon-Derksen",
                    "Ingrid Michon-Derksen"
                ]
            },
            {
                "content": "Queeny Rajkowski",
                "sounds_like": [
                    "Queenie Rijkovski",
                    "Queenie Raykofski",
                    "Queenie Rajkowski"
                ]
            },
            {
                "content": "Simone Richardson",
                "sounds_like": [
                    "Simone Richardson",
                    "Simone Richardson",
                    "Simone Richardson"
                ]
            },
            {
                "content": "Judith Tielen",
                "sounds_like": [
                    "Judith Thielen",
                    "Judith Tielen",
                    "Judith Tielen"
                ]
            },
            {
                "content": "Hester Veltman",
                "sounds_like": [
                    "Hester Veldman",
                    "Hester Veldman",
                    "Hester Veldman"
                ]
            },
            {
                "content": "Ruud Verkuijlen",
                "sounds_like": [
                    "Ruud Verkuilen",
                    "Ruud Verkuijlen",
                    "Ruud Verkuilen"
                ]
            },
            {
                "content": "Aukje de Vries",
                "sounds_like": [
                    "Aukje de Vries",
                    "Aukje de Vries",
                    "Aukje de Vries"
                ]
            },
            {
                "content": "Dilan Yeşilgöz-Zegerius",
                "sounds_like": [
                    "Dylan Jesselgoos Segerius",
                    "Dilan Yesilgöz-Zegerius",
                    "Dilan Yeşilgöz-Zegerius"
                ]
            },
            {
                "content": "Diederik Boomsma",
                "sounds_like": [
                    "Diederik Boomsma",
                    "Diederik Boomsma",
                    "Diederik Boomsma"
                ]
            },
            {
                "content": "Faith Bruyning",
                "sounds_like": [
                    "Faith Bruining",
                    "Faith Bruining",
                    "Faith Bruining"
                ]
            },
            {
                "content": "Olger van Dijk",
                "sounds_like": [
                    "Olga van Dijk",
                    "Alger van Dijk",
                    "Olger van Dijk"
                ]
            },
            {
                "content": "Annemarie Heite",
                "sounds_like": [
                    "Annemarie Heijten",
                    "Annemarie Heitje",
                    "Annemarie Heyting"
                ]
            },
            {
                "content": "Harm Holman",
                "sounds_like": [
                    "Harm Holman",
                    "Harm Holman",
                    "Harm Holman"
                ]
            },
            {
                "content": "Folkert Idsinga",
                "sounds_like": [
                    "Volkert Itzinga",
                    "Folkert Eisinga",
                    "Folkert Idsinga"
                ]
            },
            {
                "content": "Daniëlle Jansen",
                "sounds_like": [
                    "Danielle Jansen",
                    "Daniëlle Janssen",
                    "Daniëlle Jansen"
                ]
            },
            {
                "content": "Agnes Joseph",
                "sounds_like": [
                    "Agnes Jozef",
                    "Agnes Joseph",
                    "Agnes Joseph"
                ]
            },
            {
                "content": "Isa Kahraman",
                "sounds_like": [
                    "Isa Karaman",
                    "Isa Karreman",
                    "Isa Kahraman"
                ]
            },
            {
                "content": "Willem Koops",
                "sounds_like": [
                    "Willem Koops",
                    "Willem Koops",
                    "Willem Koops"
                ]
            },
            {
                "content": "Ria de Korte",
                "sounds_like": [
                    "Ria de Korte",
                    "Ria Decorte",
                    "Ria de Korte"
                ]
            },
            {
                "content": "Bram Kouwenhoven",
                "sounds_like": [
                    "Bram Kouwenhoven",
                    "Bram Kouwenhoven",
                    "Bram Kouwenhoven"
                ]
            },
            {
                "content": "Wytske Postma",
                "sounds_like": [
                    "Witske Posma",
                    "Wytske Postma",
                    "Witske Postma"
                ]
            },
            {
                "content": "Ilse Saris",
                "sounds_like": [
                    "Ilse Sares",
                    "Ilse Saris",
                    "Ilse Saris"
                ]
            },
            {
                "content": "Jesse Six Dijkstra",
                "sounds_like": [
                    "Jesse Six Dijkstra",
                    "Jesse Dijkstra",
                    "Jesse Six-Dijkstra"
                ]
            },
            {
                "content": "Aant Jelle Soepboer",
                "sounds_like": [
                    "Eten Jelle Soepboer",
                    "Aant Jelle Soepboer",
                    "Jelle Soepboer"
                ]
            },
            {
                "content": "Nicolien van Vroonhoven",
                "sounds_like": [
                    "Nicolien van Vroonhoven",
                    "Nicolien van Vroonhoven",
                    "Nicoline van Vroonhoven"
                ]
            },
            {
                "content": "Sander van Waveren",
                "sounds_like": [
                    "Sander van Waveren",
                    "Sander van Waveren",
                    "Sander van Waveren"
                ]
            },
            {
                "content": "Merlien Welzijn",
                "sounds_like": [
                    "Merlien Welzijn",
                    "Marlien Welzijn",
                    "Merlien Welzijn"
                ]
            },
            {
                "content": "Natascha Wingelaar",
                "sounds_like": [
                    "Natasja Wingelaar",
                    "Natascha Winkelaar",
                    "Natascha Wingelaar"
                ]
            },
            {
                "content": "Mpanzu Bamenga",
                "sounds_like": [
                    "Mepanzu Bamega",
                    "MePanza BAMega",
                    "Mpanzu Bamaga"
                ]
            },
            {
                "content": "Rob Jetten",
                "sounds_like": [
                    "Rob Jetten",
                    "Rob Jetten",
                    "Rob Jetten"
                ]
            },
            {
                "content": "Jan Paternotte",
                "sounds_like": [
                    "Jan Paternotte",
                    "Jan Paternotte",
                    "Jan Paternotte"
                ]
            },
            {
                "content": "Wieke Paulusma",
                "sounds_like": [
                    "Wieke Paulusma",
                    "Wieke Paulusma",
                    "Wieke Paulusma"
                ]
            },
            {
                "content": "Anne-Marijke Podt",
                "sounds_like": [
                    "Annemarijke Pot",
                    "Annemarieke Pot",
                    "Annemarijke Pot"
                ]
            },
            {
                "content": "Ilana Rooderkerk",
                "sounds_like": [
                    "Ilana Rodekerk",
                    "Ilana Roode Kerk",
                    "Ilana Rooderkerk"
                ]
            },
            {
                "content": "Joost Sneller",
                "sounds_like": [
                    "Joost Sneller",
                    "Joost Sneller",
                    "Joost Sneller"
                ]
            },
            {
                "content": "Hans Vijlbrief",
                "sounds_like": [
                    "Hans Velbrief",
                    "Hans Fel brief",
                    "Hans Velbrief"
                ]
            },
            {
                "content": "Hanneke van der Werf",
                "sounds_like": [
                    "Hanneke van der Werf",
                    "Hanneke van der Werff",
                    "Hanneke van der Werff"
                ]
            },
            {
                "content": "Martin Oostenbrink",
                "sounds_like": [
                    "Martin Oostenbrink",
                    "Martin Oosten",
                    "Martin Oostenbrink"
                ]
            },
            {
                "content": "Cor Pierik",
                "sounds_like": [
                    "Koor Pierik",
                    "Brink Cor Pierik",
                    "Cor Pierik"
                ]
            },
            {
                "content": "Caroline van der Plas",
                "sounds_like": [
                    "Caroline van der Plas",
                    "Caroline van der Plas",
                    "Caroline van der Plas"
                ]
            },
            {
                "content": "Mariska Rikkers-Oosterkamp",
                "sounds_like": [
                    "Mariska Rikkers Oosterkamp",
                    "Mariska Rikkers Oosterkamp",
                    "Mariska Rikkers Oosterkamp"
                ]
            },
            {
                "content": "Henk Vermeer",
                "sounds_like": [
                    "Henk Vermeer",
                    "Henk Vermeer",
                    "Henk Vermeer"
                ]
            },
            {
                "content": "Marieke Wijen-Nass",
                "sounds_like": [
                    "Marike Weijenas",
                    "Marieke Wijnants",
                    "Marieke Weijenas"
                ]
            },
            {
                "content": "Claudia van Zanten",
                "sounds_like": [
                    "Claudia van Zanten",
                    "Claudia van Santen",
                    "Claudia van Zanten"
                ]
            },
            {
                "content": "Henri Bontenbal",
                "sounds_like": [
                    "Henry Bontebal",
                    "Henri Bonte Bal",
                    "Henry Bontebal"
                ]
            },
            {
                "content": "Derk Boswijk",
                "sounds_like": [
                    "Dirk Boswijk",
                    "Derk Boswijk",
                    "Derk Boswijk"
                ]
            },
            {
                "content": "Inge van Dijk",
                "sounds_like": [
                    "Inge van Dijk",
                    "Inge van Dijk",
                    "Inge van Dijk"
                ]
            },
            {
                "content": "Harmen Krul",
                "sounds_like": [
                    "Harmen Krul",
                    "Harmen Krul",
                    "Harmen Krul"
                ]
            },
            {
                "content": "Eline Vedder",
                "sounds_like": [
                    "Eline Vedder",
                    "Eline Vedder",
                    "Eline Vedder"
                ]
            },
            {
                "content": "Sandra Beckerman",
                "sounds_like": [
                    "Sandra Beckerman",
                    "Sandra Beckerman",
                    "Sandra Beckerman"
                ]
            },
            {
                "content": "Jimmy Dijk",
                "sounds_like": [
                    "Jimmy Dijk",
                    "Jimmy Dijk",
                    "Jimmy Dijk"
                ]
            },
            {
                "content": "Sarah Dobbe",
                "sounds_like": [
                    "Sarah Dobbe",
                    "Sarah Dobbe",
                    "Sara Dobben"
                ]
            },
            {
                "content": "Bart van Kent",
                "sounds_like": [
                    "Bart van Kent",
                    "Bart van Kent",
                    "Bart van Kent"
                ]
            },
            {
                "content": "Michiel van Nispen",
                "sounds_like": [
                    "Michiel van Nispen",
                    "Michiel van Nispen",
                    "Michiel van Nispen"
                ]
            },
            {
                "content": "Ismail el Abassi",
                "sounds_like": [
                    "Ismael L. Abassi",
                    "Ismaël El Agassi",
                    "Ismail El Abassi"
                ]
            },
            {
                "content": "Stephan van Baarle",
                "sounds_like": [
                    "Stefan van Baarle",
                    "Stephan van Baarle",
                    "Stefan van Baarle"
                ]
            },
            {
                "content": "Doğukan Ergin",
                "sounds_like": [
                    "Dogukan Ergin",
                    "Dogan Kan erin",
                    "Dogukan Ergin"
                ]
            },
            {
                "content": "Ines Kostić",
                "sounds_like": [
                    "Ines, Kostitsch",
                    "Inez Kostic",
                    "Ines Kostic"
                ]
            },
            {
                "content": "Esther Ouwehand",
                "sounds_like": [
                    "Esther Ouwehand",
                    "Esther Ouwehand",
                    "Esther Ouwehand"
                ]
            },
            {
                "content": "Christine Teunissen",
                "sounds_like": [
                    "Christine Teunissen",
                    "Christine Teunissen",
                    "Christine Teunissen"
                ]
            },
            {
                "content": "Thierry Baudet",
                "sounds_like": [
                    "Thierry Baudet",
                    "Thierry Baudet",
                    "Thierry Baudet"
                ]
            },
            {
                "content": "Pepijn van Houwelingen",
                "sounds_like": [
                    "Pepijn van Houwelingen",
                    "Pepijn van Houwelingen",
                    "Pepijn van Houwelingen"
                ]
            },
            {
                "content": "Gideon van Meijeren",
                "sounds_like": [
                    "Gideon van Meijeren",
                    "Gideon Vermeiren",
                    "Gideon van Meijeren"
                ]
            },
            {
                "content": "Diederik van Dijk",
                "sounds_like": [
                    "Diederik van Dijk",
                    "Diederik van Dijk",
                    "Diederik van Dijk"
                ]
            },
            {
                "content": "André Flach",
                "sounds_like": [
                    "André Vlag",
                    "André Flach",
                    "André Vlag"
                ]
            },
            {
                "content": "Chris Stoffer",
                "sounds_like": [
                    "Christopher",
                    "Christopher",
                    "Christopher"
                ]
            },
            {
                "content": "Mirjam Bikker",
                "sounds_like": [
                    "Mirjam Bikker",
                    "Mirjam Bikker",
                    "Mirjam Bikker"
                ]
            },
            {
                "content": "Don Ceder",
                "sounds_like": [
                    "Don Seder",
                    "Don Ceder",
                    "Don Ceder"
                ]
            },
            {
                "content": "Pieter Grinwis",
                "sounds_like": [
                    "Pieter Grinwis",
                    "Pieter Grinwis",
                    "Pieter Grinwis"
                ]
            },
            {
                "content": "Laurens Dassen",
                "sounds_like": [
                    "Lauwens Dassen",
                    "Laurens Dassen",
                    "Laurens Dassen"
                ]
            },
            {
                "content": "Marieke Koekkoek",
                "sounds_like": [
                    "Marieke Koekoek",
                    "Marieke Koekkoek",
                    "Marieke Koekkoek"
                ]
            },
            {
                "content": "Joost Eerdmans",
                "sounds_like": [
                    "Joost Eerdmans",
                    "Joost Eerdmans",
                    "Joost Eerdmans"
                ]
            },
            {
                "content": "Aardema",
                "sounds_like": [
                    "Aardema"
                ]
            },
            {
                "content": "Blaauw",
                "sounds_like": [
                    "Blaauw"
                ]
            },
            {
                "content": "Boon",
                "sounds_like": [
                    "Boon"
                ]
            },
            {
                "content": "van den Born",
                "sounds_like": [
                    "van den Born"
                ]
            },
            {
                "content": "Bosma",
                "sounds_like": [
                    "Bosma"
                ]
            },
            {
                "content": "Boutkan",
                "sounds_like": [
                    "Boutkan"
                ]
            },
            {
                "content": "Claassen",
                "sounds_like": [
                    "Claassen"
                ]
            },
            {
                "content": "Crijns",
                "sounds_like": [
                    "Crijns"
                ]
            },
            {
                "content": "Deen",
                "sounds_like": [
                    "Deen"
                ]
            },
            {
                "content": "van Dijck",
                "sounds_like": [
                    "van Dijck"
                ]
            },
            {
                "content": "van Dijk",
                "sounds_like": [
                    "van Dijk"
                ]
            },
            {
                "content": "Esser",
                "sounds_like": [
                    "Esser"
                ]
            },
            {
                "content": "Faddegon",
                "sounds_like": [
                    "Faddegon"
                ]
            },
            {
                "content": "Graus",
                "sounds_like": [
                    "Graus"
                ]
            },
            {
                "content": "van Haasen",
                "sounds_like": [
                    "van Haasen"
                ]
            },
            {
                "content": "Heutink",
                "sounds_like": [
                    "Heutink"
                ]
            },
            {
                "content": "van der Hoeff",
                "sounds_like": [
                    "van der Hoeff"
                ]
            },
            {
                "content": "de Jong",
                "sounds_like": [
                    "de Jong"
                ]
            },
            {
                "content": "Kops",
                "sounds_like": [
                    "Kops"
                ]
            },
            {
                "content": "Markuszower",
                "sounds_like": [
                    "Markuszower"
                ]
            },
            {
                "content": "van Meetelen",
                "sounds_like": [
                    "van Meetelen"
                ]
            },
            {
                "content": "Mooiman",
                "sounds_like": [
                    "Mooiman"
                ]
            },
            {
                "content": "Mulder",
                "sounds_like": [
                    "Mulder"
                ]
            },
            {
                "content": "Nijhof-Leeuw",
                "sounds_like": [
                    "Nijhof-Leeuw"
                ]
            },
            {
                "content": "Pool",
                "sounds_like": [
                    "Pool"
                ]
            },
            {
                "content": "Ram",
                "sounds_like": [
                    "Ram"
                ]
            },
            {
                "content": "Rep",
                "sounds_like": [
                    "Rep"
                ]
            },
            {
                "content": "de Roon",
                "sounds_like": [
                    "de Roon"
                ]
            },
            {
                "content": "Smitskam",
                "sounds_like": [
                    "Smitskam"
                ]
            },
            {
                "content": "Thiadens",
                "sounds_like": [
                    "Thiadens"
                ]
            },
            {
                "content": "Uppelschoten",
                "sounds_like": [
                    "Uppelschoten"
                ]
            },
            {
                "content": "Valize",
                "sounds_like": [
                    "Valize"
                ]
            },
            {
                "content": "van der Velde",
                "sounds_like": [
                    "van der Velde"
                ]
            },
            {
                "content": "Vlottes",
                "sounds_like": [
                    "Vlottes"
                ]
            },
            {
                "content": "Vondeling",
                "sounds_like": [
                    "Vondeling"
                ]
            },
            {
                "content": "de Vree",
                "sounds_like": [
                    "de Vree"
                ]
            },
            {
                "content": "Wilders",
                "sounds_like": [
                    "Wilders"
                ]
            },
            {
                "content": "Bromet",
                "sounds_like": [
                    "Bromet"
                ]
            },
            {
                "content": "Bushoff",
                "sounds_like": [
                    "Bushoff"
                ]
            },
            {
                "content": "Chakor",
                "sounds_like": [
                    "Chakor"
                ]
            },
            {
                "content": "Gabriëls",
                "sounds_like": [
                    "Gabriëls"
                ]
            },
            {
                "content": "Haage",
                "sounds_like": [
                    "Haage"
                ]
            },
            {
                "content": "Hirsch",
                "sounds_like": [
                    "Hirsch"
                ]
            },
            {
                "content": "de Hoop",
                "sounds_like": [
                    "de Hoop"
                ]
            },
            {
                "content": "Kathmann",
                "sounds_like": [
                    "Kathmann"
                ]
            },
            {
                "content": "Klaver",
                "sounds_like": [
                    "Klaver"
                ]
            },
            {
                "content": "Kröger",
                "sounds_like": [
                    "Kröger"
                ]
            },
            {
                "content": "Lahlah",
                "sounds_like": [
                    "Lahlah"
                ]
            },
            {
                "content": "van der Lee",
                "sounds_like": [
                    "van der Lee"
                ]
            },
            {
                "content": "Mohandis",
                "sounds_like": [
                    "Mohandis"
                ]
            },
            {
                "content": "Mutluer",
                "sounds_like": [
                    "Mutluer"
                ]
            },
            {
                "content": "Nordkamp",
                "sounds_like": [
                    "Nordkamp"
                ]
            },
            {
                "content": "Patijn",
                "sounds_like": [
                    "Patijn"
                ]
            },
            {
                "content": "Pijpelink",
                "sounds_like": [
                    "Pijpelink"
                ]
            },
            {
                "content": "Piri",
                "sounds_like": [
                    "Piri"
                ]
            },
            {
                "content": "Slagt-Tichelman",
                "sounds_like": [
                    "Slagt-Tichelman"
                ]
            },
            {
                "content": "Stultiens",
                "sounds_like": [
                    "Stultiens"
                ]
            },
            {
                "content": "Thijssen",
                "sounds_like": [
                    "Thijssen"
                ]
            },
            {
                "content": "Timmermans",
                "sounds_like": [
                    "Timmermans"
                ]
            },
            {
                "content": "Tseggai",
                "sounds_like": [
                    "Tseggai"
                ]
            },
            {
                "content": "Westerveld",
                "sounds_like": [
                    "Westerveld"
                ]
            },
            {
                "content": "White",
                "sounds_like": [
                    "White"
                ]
            },
            {
                "content": "Aartsen",
                "sounds_like": [
                    "Aartsen"
                ]
            },
            {
                "content": "Becker",
                "sounds_like": [
                    "Becker"
                ]
            },
            {
                "content": "Bevers",
                "sounds_like": [
                    "Bevers"
                ]
            },
            {
                "content": "Bikkers",
                "sounds_like": [
                    "Bikkers"
                ]
            },
            {
                "content": "Buijsse",
                "sounds_like": [
                    "Buijsse"
                ]
            },
            {
                "content": "van der Burg",
                "sounds_like": [
                    "van der Burg"
                ]
            },
            {
                "content": "van Campen",
                "sounds_like": [
                    "van Campen"
                ]
            },
            {
                "content": "Dral",
                "sounds_like": [
                    "Dral"
                ]
            },
            {
                "content": "van Eijk",
                "sounds_like": [
                    "van Eijk"
                ]
            },
            {
                "content": "Ellian",
                "sounds_like": [
                    "Ellian"
                ]
            },
            {
                "content": "Erkens",
                "sounds_like": [
                    "Erkens"
                ]
            },
            {
                "content": "de Groot",
                "sounds_like": [
                    "de Groot"
                ]
            },
            {
                "content": "Kisteman",
                "sounds_like": [
                    "Kisteman"
                ]
            },
            {
                "content": "de Kort",
                "sounds_like": [
                    "de Kort"
                ]
            },
            {
                "content": "Martens-America",
                "sounds_like": [
                    "Martens-America"
                ]
            },
            {
                "content": "Meulenkamp",
                "sounds_like": [
                    "Meulenkamp"
                ]
            },
            {
                "content": "Michon-Derkzen",
                "sounds_like": [
                    "Michon-Derkzen"
                ]
            },
            {
                "content": "Rajkowski",
                "sounds_like": [
                    "Rajkowski"
                ]
            },
            {
                "content": "Richardson",
                "sounds_like": [
                    "Richardson"
                ]
            },
            {
                "content": "Tielen",
                "sounds_like": [
                    "Tielen"
                ]
            },
            {
                "content": "Veltman",
                "sounds_like": [
                    "Veltman"
                ]
            },
            {
                "content": "Verkuijlen",
                "sounds_like": [
                    "Verkuijlen"
                ]
            },
            {
                "content": "de Vries",
                "sounds_like": [
                    "de Vries"
                ]
            },
            {
                "content": "Yeşilgöz-Zegerius",
                "sounds_like": [
                    "Yeşilgöz-Zegerius"
                ]
            },
            {
                "content": "Boomsma",
                "sounds_like": [
                    "Boomsma"
                ]
            },
            {
                "content": "Bruyning",
                "sounds_like": [
                    "Bruyning"
                ]
            },
            {
                "content": "van Dijk",
                "sounds_like": [
                    "van Dijk"
                ]
            },
            {
                "content": "Heite",
                "sounds_like": [
                    "Heite"
                ]
            },
            {
                "content": "Holman",
                "sounds_like": [
                    "Holman"
                ]
            },
            {
                "content": "Idsinga",
                "sounds_like": [
                    "Idsinga"
                ]
            },
            {
                "content": "Jansen",
                "sounds_like": [
                    "Jansen"
                ]
            },
            {
                "content": "Joseph",
                "sounds_like": [
                    "Joseph"
                ]
            },
            {
                "content": "Kahraman",
                "sounds_like": [
                    "Kahraman"
                ]
            },
            {
                "content": "Koops",
                "sounds_like": [
                    "Koops"
                ]
            },
            {
                "content": "de Korte",
                "sounds_like": [
                    "de Korte"
                ]
            },
            {
                "content": "Kouwenhoven",
                "sounds_like": [
                    "Kouwenhoven"
                ]
            },
            {
                "content": "Postma",
                "sounds_like": [
                    "Postma"
                ]
            },
            {
                "content": "Saris",
                "sounds_like": [
                    "Saris"
                ]
            },
            {
                "content": "Dijkstra",
                "sounds_like": [
                    "Dijkstra"
                ]
            },
            {
                "content": "Soepboer",
                "sounds_like": [
                    "Soepboer"
                ]
            },
            {
                "content": "van Vroonhoven",
                "sounds_like": [
                    "van Vroonhoven"
                ]
            },
            {
                "content": "van Waveren",
                "sounds_like": [
                    "van Waveren"
                ]
            },
            {
                "content": "Welzijn",
                "sounds_like": [
                    "Welzijn"
                ]
            },
            {
                "content": "Wingelaar",
                "sounds_like": [
                    "Wingelaar"
                ]
            },
            {
                "content": "Bamenga",
                "sounds_like": [
                    "Bamenga"
                ]
            },
            {
                "content": "Jetten",
                "sounds_like": [
                    "Jetten"
                ]
            },
            {
                "content": "Paternotte",
                "sounds_like": [
                    "Paternotte"
                ]
            },
            {
                "content": "Paulusma",
                "sounds_like": [
                    "Paulusma"
                ]
            },
            {
                "content": "Podt",
                "sounds_like": [
                    "Podt"
                ]
            },
            {
                "content": "Rooderkerk",
                "sounds_like": [
                    "Rooderkerk"
                ]
            },
            {
                "content": "Sneller",
                "sounds_like": [
                    "Sneller"
                ]
            },
            {
                "content": "Vijlbrief",
                "sounds_like": [
                    "Vijlbrief"
                ]
            },
            {
                "content": "van der Werf",
                "sounds_like": [
                    "van der Werf"
                ]
            },
            {
                "content": "Oostenbrink",
                "sounds_like": [
                    "Oostenbrink"
                ]
            },
            {
                "content": "Pierik",
                "sounds_like": [
                    "Pierik"
                ]
            },
            {
                "content": "van der Plas",
                "sounds_like": [
                    "van der Plas"
                ]
            },
            {
                "content": "Rikkers-Oosterkamp",
                "sounds_like": [
                    "Rikkers-Oosterkamp"
                ]
            },
            {
                "content": "Vermeer",
                "sounds_like": [
                    "Vermeer"
                ]
            },
            {
                "content": "Wijen-Nass",
                "sounds_like": [
                    "Wijen-Nass"
                ]
            },
            {
                "content": "van Zanten",
                "sounds_like": [
                    "van Zanten"
                ]
            },
            {
                "content": "Bontenbal",
                "sounds_like": [
                    "Bontenbal"
                ]
            },
            {
                "content": "Boswijk",
                "sounds_like": [
                    "Boswijk"
                ]
            },
            {
                "content": "van Dijk",
                "sounds_like": [
                    "van Dijk"
                ]
            },
            {
                "content": "Krul",
                "sounds_like": [
                    "Krul"
                ]
            },
            {
                "content": "Vedder",
                "sounds_like": [
                    "Vedder"
                ]
            },
            {
                "content": "Beckerman",
                "sounds_like": [
                    "Beckerman"
                ]
            },
            {
                "content": "Dijk",
                "sounds_like": [
                    "Dijk"
                ]
            },
            {
                "content": "Dobbe",
                "sounds_like": [
                    "Dobbe"
                ]
            },
            {
                "content": "van Kent",
                "sounds_like": [
                    "van Kent"
                ]
            },
            {
                "content": "van Nispen",
                "sounds_like": [
                    "van Nispen"
                ]
            },
            {
                "content": "el Abassi",
                "sounds_like": [
                    "el Abassi"
                ]
            },
            {
                "content": "van Baarle",
                "sounds_like": [
                    "van Baarle"
                ]
            },
            {
                "content": "Ergin",
                "sounds_like": [
                    "Ergin"
                ]
            },
            {
                "content": "Kostić",
                "sounds_like": [
                    "Kostić"
                ]
            },
            {
                "content": "Ouwehand",
                "sounds_like": [
                    "Ouwehand"
                ]
            },
            {
                "content": "Teunissen",
                "sounds_like": [
                    "Teunissen"
                ]
            },
            {
                "content": "Baudet",
                "sounds_like": [
                    "Baudet"
                ]
            },
            {
                "content": "van Houwelingen",
                "sounds_like": [
                    "van Houwelingen"
                ]
            },
            {
                "content": "van Meijeren",
                "sounds_like": [
                    "van Meijeren"
                ]
            },
            {
                "content": "van Dijk",
                "sounds_like": [
                    "van Dijk"
                ]
            },
            {
                "content": "Flach",
                "sounds_like": [
                    "Flach"
                ]
            },
            {
                "content": "Stoffer",
                "sounds_like": [
                    "Stoffer"
                ]
            },
            {
                "content": "Bikker",
                "sounds_like": [
                    "Bikker"
                ]
            },
            {
                "content": "Ceder",
                "sounds_like": [
                    "Ceder"
                ]
            },
            {
                "content": "Grinwis",
                "sounds_like": [
                    "Grinwis"
                ]
            },
            {
                "content": "Dassen",
                "sounds_like": [
                    "Dassen"
                ]
            },
            {
                "content": "Koekkoek",
                "sounds_like": [
                    "Koekkoek"
                ]
            },
            {
                "content": "Eerdmans",
                "sounds_like": [
                    "Eerdmans"
                ]
            }
        ]
    )
        speechmatics_settings = speechmatics.models.AudioSettings()

    chunk_index = 0
    processed_bytes = 0
    chunk_bytes = CHUNK_DURATION_SEC * 16000 * 2  # 40s * 16kHz * 2 bytes/sample. 32k bytes per second
    overlap_bytes = OVERLAP_SEC * 16000 * 2  # 1s * 16kHz * 2 bytes/sample. 32k bytes per second

    global audio_buffer

    while True:
        try:
            time.sleep(1)  # Give time to fill. Prevents busy-looping (CPU waste checking empty buffer)
            with buffer_lock:
                # print("DEBUG: audio_buffer.tell(): ", audio_buffer.tell())
                # first_size = audio_buffer.tell()

                # print("DEBUG: audio_buffer.read(): ", audio_buffer.read())

                # The `if` statement checks if the buffer has accumulated enough data to form a full chunk.
                # If the current buffer size is less than the required chunk size, it continues the loop,
                # waiting for more data to be written by the FFmpeg thread.
                if audio_buffer.tell() < chunk_bytes:
                    continue

                # Move the cursor to the beginning of the buffer.
                # This is crucial for starting the read operation from the start of the accumulated data.
                audio_buffer.seek(0)

                # Read the full chunk of data for processing. This reads `chunk_bytes` from the beginning of the buffer.
                raw_data = audio_buffer.read(chunk_bytes)

                # Read the rest of the data in the buffer. The cursor is now at the end of the first chunk,
                # so this reads all the remaining, unprocessed data from that point to the end.
                remaining_data = audio_buffer.read()

                # Create a new, clean `io.BytesIO` buffer to hold the data for the next loop iteration.
                audio_buffer = io.BytesIO()

                # Write the `remaining_data` to the newly created buffer. This effectively "shifts" the unprocessed data
                # to the beginning of the new buffer, ready to be added to in the next loop.
                audio_buffer.write(remaining_data)

                # Create an `AudioSegment` object from the `raw_data`. This object is used to
                # prepare the audio chunk for transcription
                chunk = AudioSegment(
                    data=raw_data,
                    sample_width=2,  # 16-bit PCM
                    frame_rate=16000,
                    channels=1
                )

                # Update the count of processed bytes. This is used to track the total amount of audio
                # that has been successfully transcribed, accounting for the overlap.
                processed_bytes += chunk_bytes - overlap_bytes
                # Increment the chunk index to keep track of the current chunk number.
                chunk_index += 1

            logger.info(f"Processing chunk {chunk_index} ({len(chunk)}ms)")

            if not process_chunk_with_retry(chunk, chunk_index, conf, speechmatics_settings, use_rag=use_rag):
                logger.warning("Skipping to next chunk...")
                continue

            # Add cooldown between chunks to prevent rate limiting
            logger.debug(f"Waiting {CHUNK_COOLDOWN}s before next chunk...")
            time.sleep(CHUNK_COOLDOWN)

        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt. Shutting down...")
            break
        except Exception as e:
            logger.error(f"Main loop error: {str(e)}")
            time.sleep(5)


if __name__ == "__main__":
    try:
        threading.Thread(target=run_ffmpeg_stream, daemon=True).start()  # Starts the FFmpeg streaming thread as a daemon
        stream_and_transcribe(use_rag=True)  # Main transcription loop
    except KeyboardInterrupt:
        logger.info("Shutting down...")

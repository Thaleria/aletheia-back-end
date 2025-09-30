from pathlib import Path
from lfc.app_settings import settings

# === CONFIGURATION ===
M3U8_STREAM_URL = "http://as-hls-ww-live.akamaized.net/pool_01505109/live/ww/bbc_radio_one/bbc_radio_one.isml/bbc_radio_one-audio%3d96000.norewind.m3u8"
CHUNK_DURATION_SEC = 40
OVERLAP_SEC = 1
LANGUAGE = "en"
CONNECTION_URL = "wss://eu2.rt.speechmatics.com/v2"
RAG_API_URL = "http://127.0.0.1:8002/v1/chat"

# Retry configuration
MAX_RETRIES = 5
INITIAL_RETRY_DELAY = 2  # seconds
MAX_RETRY_DELAY = 32  # seconds
CHUNK_COOLDOWN = 3  # seconds

# Derived values
CHUNK_BYTES = CHUNK_DURATION_SEC * 16000 * 2  # 40s * 16kHz * 2 bytes/sample
OVERLAP_BYTES = OVERLAP_SEC * 16000 * 2  # 1s * 16kHz * 2 bytes/sample

# Speechmatics API key
SPEECHMATICS_API_KEY = settings.speechmatics_api_key
if not SPEECHMATICS_API_KEY:
    raise EnvironmentError("SPEECHMATICS_API_KEY environment variable not set")

# Directories
OUTPUT_DIR = Path("lfc/transcriptions")
FAILED_CHUNKS_DIR = OUTPUT_DIR / "failed_chunks"  # Directory to save chunks that failed transcription

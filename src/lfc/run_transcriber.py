import sys
from lfc.live_transcriber_class.speechmatics_class import SpeechmaticsTranscriberSpeakers


def main():
    """Starts the live transcription process.

    This function serves as the entry point for the application. It initializes
    the `SpeechmaticsTranscriber` and starts the audio streaming and
    transcription loops. It includes a `try-except` block to gracefully handle
    common application errors.

    Raises:
        EnvironmentError: If required environment variables or configurations are
            missing.
        KeyboardInterrupt: If the user manually stops the process (e.g., with Ctrl+C).
        Exception: For any other unexpected errors that occur during runtime.
    """

    try:
        transcriber = SpeechmaticsTranscriberSpeakers(use_rag=False)
        transcriber.start()
    except EnvironmentError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("Shutting down...", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()

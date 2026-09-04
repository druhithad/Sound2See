import sys
import os
import whisper


# ============================================================
# SETTINGS
# ============================================================

MODEL_NAME = "tiny"


# ============================================================
# LOAD WHISPER MODEL
# ============================================================

print("=" * 55)
print("SOUND2SEE - SPEECH TO TEXT")
print("=" * 55)

print()
print(f"Loading Whisper model: {MODEL_NAME}")

model = whisper.load_model(MODEL_NAME)

print("Whisper model loaded successfully.")


# ============================================================
# SPEECH TO TEXT
# ============================================================

def speech_to_text(audio_file):

    if not os.path.exists(audio_file):

        print()
        print("ERROR: Audio file not found.")
        print(audio_file)

        return None

    print()
    print("Processing audio:")
    print(audio_file)

    result = model.transcribe(
        audio_file,
        fp16=False
    )

    text = result["text"].strip()

    print()
    print("=" * 55)
    print("           SPEECH TO TEXT RESULT")
    print("=" * 55)

    if text:

        print()
        print("Recognized text:")
        print(text)

    else:

        print()
        print("No speech recognized.")

    print()
    print("=" * 55)

    return text


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) != 2:

        print()
        print("Usage:")
        print(
            "python src\\speech_to_text.py <audio.wav>"
        )
        print()

        sys.exit(1)

    audio_file = sys.argv[1]

    speech_to_text(audio_file)

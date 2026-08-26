import os
import sys

from record_live import main as record_audio
from speech_detector import detect_speech
from speech_to_text import speech_to_text
from predict_audio import main as predict_audio


# ============================================================
# SETTINGS
# ============================================================

AUDIO_FILE = r"data\inference\live_test.wav"


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 65)
print("                         SOUND2SEE")
print("                  UNIFIED AUDIO PIPELINE")
print("=" * 65)


# ============================================================
# STAGE 1 - AUDIO CAPTURE
# ============================================================

print()
print("=" * 65)
print("                         AUDIO CAPTURE")
print("=" * 65)

print()
print("Starting microphone recording...")

record_audio()

if not os.path.exists(AUDIO_FILE):

    print()
    print("ERROR: Audio recording was not created.")
    print(AUDIO_FILE)

    sys.exit(1)

print()
print("Audio recording successful.")


# ============================================================
# STAGE 2 - SPEECH DETECTION
# ============================================================

print()
print("Analyzing recorded audio...")

speech_detected = detect_speech(AUDIO_FILE)


# ============================================================
# STAGE 3 - AUTOMATIC DECISION
# ============================================================

print()
print("=" * 65)
print("                    AUTOMATIC DECISION")
print("=" * 65)

if speech_detected:

    # ========================================================
    # SPEECH BRANCH
    # ========================================================

    print()
    print("Speech detected.")
    print("Automatically selecting Speech-to-Text.")

    print()
    print("=" * 65)
    print("                      SPEECH TO TEXT")
    print("=" * 65)

    text = speech_to_text(AUDIO_FILE)

    print()
    print("=" * 65)
    print("                    SOUND2SEE RESULT")
    print("=" * 65)

    print()
    print("Input Type : Human Speech")

    print()
    print("Recognized Text:")

    if text:

        print(text)

    else:

        print("No speech recognized.")

else:

    # ========================================================
    # SOUND CLASSIFICATION BRANCH
    # ========================================================

    print()
    print("No speech detected.")
    print("Automatically selecting Sound Classification.")

    print()
    print("=" * 65)
    print("                   SOUND CLASSIFICATION")
    print("=" * 65)

    # predict_audio.main() expects the audio path
    # through sys.argv.

    original_argv = sys.argv.copy()

    try:

        sys.argv = [
            "predict_audio.py",
            AUDIO_FILE
        ]

        predict_audio()

    finally:

        sys.argv = original_argv


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 65)
print("                  SOUND2SEE PIPELINE COMPLETE")
print("=" * 65)
print()
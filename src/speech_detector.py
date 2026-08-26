import sys
import os

import numpy as np
import librosa


# ============================================================
# SETTINGS
# ============================================================

SAMPLE_RATE = 16000

FRAME_LENGTH = 1024
HOP_LENGTH = 512

# These are intentionally conservative.
# The detector uses several measurements instead of RMS alone.
MIN_SPEECH_RMS = 0.008
MAX_SPEECH_ZCR = 0.25

MIN_SPEECH_FRAMES_RATIO = 0.20
MAX_SILENCE_RATIO = 0.80


# ============================================================
# FEATURE MEASUREMENTS
# ============================================================

def calculate_audio_features(audio):

    rms = librosa.feature.rms(
        y=audio,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH
    )[0]

    zcr = librosa.feature.zero_crossing_rate(
        y=audio,
        frame_length=FRAME_LENGTH,
        hop_length=HOP_LENGTH
    )[0]

    spectral_centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=FRAME_LENGTH,
        hop_length=HOP_LENGTH
    )[0]

    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=FRAME_LENGTH,
        hop_length=HOP_LENGTH
    )[0]

    return (
        rms,
        zcr,
        spectral_centroid,
        spectral_bandwidth
    )


# ============================================================
# SPEECH DETECTION
# ============================================================

def detect_speech(audio_file):

    print("=" * 55)
    print("SOUND2SEE - SPEECH DETECTION")
    print("=" * 55)

    # --------------------------------------------------------
    # Check file
    # --------------------------------------------------------

    if not os.path.exists(audio_file):

        print()
        print("ERROR: Audio file not found:")
        print(audio_file)

        return False

    print()
    print("Loading audio:")
    print(audio_file)

    # --------------------------------------------------------
    # Load audio
    # --------------------------------------------------------

    audio, sr = librosa.load(
        audio_file,
        sr=SAMPLE_RATE,
        mono=True
    )

    print()
    print("Sample rate :", sr)
    print("Samples     :", len(audio))
    print(
        "Duration    :",
        round(len(audio) / sr, 2),
        "seconds"
    )

    # --------------------------------------------------------
    # Calculate features
    # --------------------------------------------------------

    (
        rms,
        zcr,
        spectral_centroid,
        spectral_bandwidth
    ) = calculate_audio_features(audio)

    mean_rms = float(np.mean(rms))
    mean_zcr = float(np.mean(zcr))
    mean_centroid = float(np.mean(spectral_centroid))
    mean_bandwidth = float(np.mean(spectral_bandwidth))

    # --------------------------------------------------------
    # Frame-level speech candidates
    # --------------------------------------------------------

    speech_candidate = (
        (rms >= MIN_SPEECH_RMS)
        &
        (zcr <= MAX_SPEECH_ZCR)
    )

    speech_frames = int(
        np.sum(speech_candidate)
    )

    total_frames = len(rms)

    if total_frames > 0:

        speech_ratio = (
            speech_frames /
            total_frames
        )

    else:

        speech_ratio = 0.0

    silence_ratio = 1.0 - speech_ratio

    # --------------------------------------------------------
    # Print measurements
    # --------------------------------------------------------

    print()
    print("Audio measurements")
    print("----------------------------")

    print(
        "Mean RMS              :",
        round(mean_rms, 6)
    )

    print(
        "Mean ZCR              :",
        round(mean_zcr, 6)
    )

    print(
        "Mean spectral centroid:",
        round(mean_centroid, 2),
        "Hz"
    )

    print(
        "Mean spectral bandwidth:",
        round(mean_bandwidth, 2),
        "Hz"
    )

    print(
        "Speech-like frames    :",
        speech_frames,
        "/",
        total_frames
    )

    print(
        "Speech frame ratio    :",
        round(speech_ratio, 3)
    )

    # --------------------------------------------------------
    # Decision
    # --------------------------------------------------------

    speech_detected = (
        mean_rms >= MIN_SPEECH_RMS
        and
        speech_ratio >= MIN_SPEECH_FRAMES_RATIO
        and
        silence_ratio <= MAX_SILENCE_RATIO
    )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    print()
    print("=" * 55)

    if speech_detected:

        print("RESULT: SPEECH DETECTED")

    else:

        print("RESULT: NO SPEECH DETECTED")

    print("=" * 55)

    return speech_detected


# ============================================================
# PROGRAM ENTRY
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) != 2:

        print()
        print("Usage:")
        print(
            "python src\\speech_detector.py <audio.wav>"
        )
        print()

        sys.exit(1)

    audio_file = sys.argv[1]

    detect_speech(audio_file)
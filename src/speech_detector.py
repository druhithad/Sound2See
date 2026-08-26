import sys
import numpy as np
import librosa


SAMPLE_RATE = 16000


def detect_speech(audio_file):

    print("=" * 55)
    print("SOUND2SEE - SPEECH DETECTION")
    print("=" * 55)

    print()
    print("Loading audio:")
    print(audio_file)

    audio, sr = librosa.load(
        audio_file,
        sr=SAMPLE_RATE,
        mono=True
    )

    print()
    print("Sample rate :", sr)
    print("Samples     :", len(audio))
    print("Duration    :", round(len(audio) / sr, 2), "seconds")

    # ---------------------------------------------
    # Basic audio measurements
    # ---------------------------------------------

    rms = librosa.feature.rms(y=audio)[0]

    zero_crossing_rate = librosa.feature.zero_crossing_rate(
        audio
    )[0]

    mean_rms = float(np.mean(rms))
    mean_zcr = float(np.mean(zero_crossing_rate))

    print()
    print("Audio measurements")
    print("----------------------------")
    print("Mean RMS :", round(mean_rms, 6))
    print("Mean ZCR :", round(mean_zcr, 6))

    # ---------------------------------------------
    # Initial speech decision
    # ---------------------------------------------

    if mean_rms < 0.01:

        speech_detected = False

    else:

        speech_detected = True

    print()
    print("=" * 55)

    if speech_detected:

        print("RESULT: SPEECH DETECTED")

    else:

        print("RESULT: NO SPEECH DETECTED")

    print("=" * 55)

    return speech_detected


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
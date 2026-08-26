import os
import sys
import joblib
import numpy as np
import pandas as pd
import librosa


# ============================================================
# SETTINGS
# ============================================================

MODEL_FILE = (
    r"outputs\final_baseline\sound2see_svm_baseline.joblib"
)

SAMPLE_RATE = 16000

N_MFCC = 13
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 40


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(audio_file):

    print("\nLoading audio...")

    audio, sr = librosa.load(
        audio_file,
        sr=SAMPLE_RATE,
        mono=True
    )

    print(
        f"Sample rate : {sr} Hz"
    )

    print(
        f"Samples     : {len(audio)}"
    )

    print(
        f"Duration    : {len(audio) / sr:.2f} seconds"
    )

    # --------------------------------------------------------
    # MFCC
    # --------------------------------------------------------

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )

    # --------------------------------------------------------
    # Delta
    # --------------------------------------------------------

    delta = librosa.feature.delta(
        mfcc
    )

    # --------------------------------------------------------
    # Delta-Delta
    # --------------------------------------------------------

    delta2 = librosa.feature.delta(
        mfcc,
        order=2
    )

    features = []

    # ========================================================
    # MFCC FEATURES
    # ========================================================

    for i in range(N_MFCC):

        features.append(
            np.mean(mfcc[i])
        )

        features.append(
            np.std(mfcc[i])
        )

    # ========================================================
    # DELTA FEATURES
    # ========================================================

    for i in range(N_MFCC):

        features.append(
            np.mean(delta[i])
        )

        features.append(
            np.std(delta[i])
        )

    # ========================================================
    # DELTA-DELTA FEATURES
    # ========================================================

    for i in range(N_MFCC):

        features.append(
            np.mean(delta2[i])
        )

        features.append(
            np.std(delta2[i])
        )

    # ========================================================
    # SPECTRAL FEATURES
    # ========================================================

    spectral_centroid = (
        librosa.feature.spectral_centroid(
            y=audio,
            sr=sr,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
        )
    )

    spectral_bandwidth = (
        librosa.feature.spectral_bandwidth(
            y=audio,
            sr=sr,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
        )
    )

    spectral_rolloff = (
        librosa.feature.spectral_rolloff(
            y=audio,
            sr=sr,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH
        )
    )

    zero_crossing_rate = (
        librosa.feature.zero_crossing_rate(
            audio,
            frame_length=N_FFT,
            hop_length=HOP_LENGTH
        )
    )

    rms_energy = (
        librosa.feature.rms(
            y=audio,
            frame_length=N_FFT,
            hop_length=HOP_LENGTH
        )
    )

    spectral_features = [
        spectral_centroid,
        spectral_bandwidth,
        spectral_rolloff,
        zero_crossing_rate,
        rms_energy
    ]

    for feature in spectral_features:

        features.append(
            np.mean(feature)
        )

        features.append(
            np.std(feature)
        )

    # --------------------------------------------------------
    # Convert to array
    # --------------------------------------------------------

    features = np.array(
        features,
        dtype=np.float64
    )

    return features


# ============================================================
# MAIN
# ============================================================

def main():

    print("========================================")
    print("       SOUND2SEE AUDIO PREDICTOR")
    print("========================================")

    # --------------------------------------------------------
    # Check command-line argument
    # --------------------------------------------------------

    if len(sys.argv) < 2:

        print(
            "\nUsage:"
        )

        print(
            "python src\\predict_audio.py "
            "<audio_file.wav>"
        )

        print(
            "\nExample:"
        )

        print(
            "python src\\predict_audio.py "
            "data\\inference\\sample.wav"
        )

        return

    audio_file = sys.argv[1]

    # --------------------------------------------------------
    # Check audio file
    # --------------------------------------------------------

    if not os.path.exists(audio_file):

        print(
            "\nERROR: Audio file not found."
        )

        print(
            audio_file
        )

        return

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if not os.path.exists(MODEL_FILE):

        print(
            "\nERROR: Trained model not found."
        )

        print(
            MODEL_FILE
        )

        return

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print(
        "\nLoading trained model..."
    )

    model = joblib.load(
        MODEL_FILE
    )

    # --------------------------------------------------------
    # Extract features
    # --------------------------------------------------------

    features = extract_features(
        audio_file
    )

    print(
        f"\nExtracted features: "
        f"{len(features)}"
    )

    # --------------------------------------------------------
    # Verify feature count
    # --------------------------------------------------------

    if len(features) != 88:

        print(
            "\nERROR: Expected 88 features."
        )

        print(
            f"Received: {len(features)}"
        )

        return

    # --------------------------------------------------------
    # Reshape for model
    # --------------------------------------------------------

    feature_file = r"data\processed\features.csv"
    feature_columns = pd.read_csv(
        feature_file,
        nrows=0
        ).columns.tolist()[2:]
    X = pd.DataFrame(
        [features],
        columns=feature_columns
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = model.predict(
        X
    )[0]

    # --------------------------------------------------------
    # Decision score
    # --------------------------------------------------------

    decision = model.decision_function(
        X
    )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    print("\n========================================")
    print("           PREDICTION RESULT")
    print("========================================")

    print(
        f"Predicted sound : {prediction}"
    )

    print(
        "\nKnown classes:"
    )

    for sound_class in model.classes_:

        print(
            f"  - {sound_class}"
        )

    # --------------------------------------------------------
    # Decision scores
    # --------------------------------------------------------

    print(
        "\nDecision scores:"
    )

    if decision.ndim == 1:

        for sound_class, score in zip(
            model.classes_,
            decision
        ):

            print(
                f"{sound_class:20s}: "
                f"{score:.4f}"
            )

    else:

        for sound_class, score in zip(
            model.classes_,
            decision[0]
        ):

            print(
                f"{sound_class:20s}: "
                f"{score:.4f}"
            )

    print("\n========================================")
    print("        PREDICTION COMPLETE")
    print("========================================")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
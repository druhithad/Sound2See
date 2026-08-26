import os
import joblib
import numpy as np
import pandas as pd
import librosa
import sounddevice as sd


# ============================================================
# SETTINGS
# ============================================================

MODEL_FILE = r"outputs\final_baseline\sound2see_svm_baseline.joblib"
FEATURE_FILE = r"data\processed\features.csv"

MIC_SAMPLE_RATE = 48000
MODEL_SAMPLE_RATE = 16000

DURATION = 5

MICROPHONE_CHANNELS = 2
MICROPHONE_DEVICE = 11


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(audio):

    sr = MODEL_SAMPLE_RATE

    n_mfcc = 13
    n_fft = 1024
    hop_length = 512
    n_mels = 40

    # --------------------------------------------------------
    # MFCC
    # --------------------------------------------------------

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )

    # --------------------------------------------------------
    # Delta
    # --------------------------------------------------------

    delta = librosa.feature.delta(mfcc)

    # --------------------------------------------------------
    # Delta-Delta
    # --------------------------------------------------------

    delta2 = librosa.feature.delta(
        mfcc,
        order=2
    )

    features = []

    # --------------------------------------------------------
    # MFCC
    # --------------------------------------------------------

    for i in range(n_mfcc):

        features.append(np.mean(mfcc[i]))
        features.append(np.std(mfcc[i]))

    # --------------------------------------------------------
    # Delta
    # --------------------------------------------------------

    for i in range(n_mfcc):

        features.append(np.mean(delta[i]))
        features.append(np.std(delta[i]))

    # --------------------------------------------------------
    # Delta-Delta
    # --------------------------------------------------------

    for i in range(n_mfcc):

        features.append(np.mean(delta2[i]))
        features.append(np.std(delta2[i]))

    # --------------------------------------------------------
    # Spectral features
    # --------------------------------------------------------

    spectral_centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length
    )

    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length
    )

    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length
    )

    zero_crossing_rate = librosa.feature.zero_crossing_rate(
        audio,
        frame_length=n_fft,
        hop_length=hop_length
    )

    rms_energy = librosa.feature.rms(
        y=audio,
        frame_length=n_fft,
        hop_length=hop_length
    )

    spectral_features = [
        spectral_centroid,
        spectral_bandwidth,
        spectral_rolloff,
        zero_crossing_rate,
        rms_energy
    ]

    for feature in spectral_features:

        features.append(np.mean(feature))
        features.append(np.std(feature))

    return np.array(
        features,
        dtype=np.float64
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("========================================")
    print("     SOUND2SEE REAL-TIME DETECTOR")
    print("========================================")

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if not os.path.exists(MODEL_FILE):

        print("\nERROR: Model not found.")
        print(MODEL_FILE)
        return

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading trained model...")

    model = joblib.load(
        MODEL_FILE
    )

    print(
        f"Model features : {model.n_features_in_}"
    )

    print(
        f"Classes        : {list(model.classes_)}"
    )

    # --------------------------------------------------------
    # Ready
    # --------------------------------------------------------

    print("\n========================================")
    print("             READY")
    print("========================================")

    input(
        "\nPress ENTER to start listening..."
    )

    # --------------------------------------------------------
    # Recording
    # --------------------------------------------------------

    print("\nRecording for 5 seconds...")
    print(">>> MAKE A SOUND <<<")

    try:

        audio = sd.rec(
            int(DURATION * MIC_SAMPLE_RATE),
            samplerate=MIC_SAMPLE_RATE,
            channels=MICROPHONE_CHANNELS,
            dtype="float32",
            device=MICROPHONE_DEVICE
        )

        sd.wait()

    except Exception as e:

        print("\nERROR while recording:")
        print(e)
        return

    # --------------------------------------------------------
    # Check recording shape
    # --------------------------------------------------------

    print("\nRecording complete.")

    print(
        f"Original shape : {audio.shape}"
    )

    print(
        f"Original rate  : {MIC_SAMPLE_RATE} Hz"
    )

    # --------------------------------------------------------
    # Calculate channel levels
    # --------------------------------------------------------

    channel_rms = []

    for channel in range(
        MICROPHONE_CHANNELS
    ):

        rms = np.sqrt(
            np.mean(
                np.square(
                    audio[:, channel]
                )
            )
        )

        channel_rms.append(rms)

        peak = np.max(
            np.abs(
                audio[:, channel]
            )
        )

        print(
            f"Channel {channel + 1} RMS  : "
            f"{rms:.4f}"
        )

        print(
            f"Channel {channel + 1} Peak : "
            f"{peak:.4f}"
        )

    # --------------------------------------------------------
    # Select strongest channel
    # --------------------------------------------------------

    selected_channel = int(
        np.argmax(channel_rms)
    )

    print(
        f"\nSelected channel : "
        f"{selected_channel + 1}"
    )

    # --------------------------------------------------------
    # Convert selected channel to mono
    # --------------------------------------------------------

    audio = audio[
        :,
        selected_channel
    ]

    

    print(
        f"Selected samples : {len(audio)}"
    )

    print(
        f"Duration         : "
        f"{len(audio) / MIC_SAMPLE_RATE:.2f} seconds"
    )

    # --------------------------------------------------------
    # Audio level check
    # --------------------------------------------------------

    rms = np.sqrt(
        np.mean(
            np.square(audio)
        )
    )

    peak = np.max(
        np.abs(audio)
    )

    print(
        f"Selected RMS     : {rms:.4f}"
    )

    print(
        f"Selected Peak    : {peak:.4f}"
    )

    # --------------------------------------------------------
    # Weak audio check
    # --------------------------------------------------------

    if rms < 0.01:

        print("\n========================================")
        print("          NO CLEAR SOUND")
        print("========================================")

        print(
            f"\nRMS level: {rms:.4f}"
        )

        print(
            "\nThe microphone signal is too weak."
        )

        return

    # --------------------------------------------------------
    # Resample
    # --------------------------------------------------------

    print("\nResampling audio...")

    print(
        f"{MIC_SAMPLE_RATE} Hz -> "
        f"{MODEL_SAMPLE_RATE} Hz"
    )

    audio = librosa.resample(
        audio,
        orig_sr=MIC_SAMPLE_RATE,
        target_sr=MODEL_SAMPLE_RATE
    )

    print(
        f"Resampled samples: "
        f"{len(audio)}"
    )

    print(
        f"Resampled duration: "
        f"{len(audio) / MODEL_SAMPLE_RATE:.2f} seconds"
    )

    # --------------------------------------------------------
    # Feature extraction
    # --------------------------------------------------------

    print("\nExtracting features...")

    features = extract_features(
        audio
    )

    print(
        f"Features extracted: "
        f"{len(features)}"
    )

    # --------------------------------------------------------
    # Verify features
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
    # Load feature names
    # --------------------------------------------------------

    feature_columns = pd.read_csv(
        FEATURE_FILE,
        nrows=0
    ).columns.tolist()[2:]

    if len(feature_columns) != 88:

        print(
            "\nERROR: Expected 88 feature columns."
        )

        print(
            f"Found: {len(feature_columns)}"
        )

        return

    # --------------------------------------------------------
    # Create input dataframe
    # --------------------------------------------------------

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
    # Decision scores
    # --------------------------------------------------------

    decision = model.decision_function(
        X
    )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    print("\n========================================")
    print("          SOUND DETECTED")
    print("========================================")

    print(
        f"\nPredicted sound : "
        f"{prediction}"
    )

    print("\nDecision scores:")

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
    print("       REAL-TIME TEST COMPLETE")
    print("========================================")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
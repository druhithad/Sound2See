import os
import numpy as np
import pandas as pd
import librosa


# ============================================================
# SETTINGS
# ============================================================

DATASET_FOLDER = r"data\processed\esc50_16khz"

OUTPUT_FILE = r"data\processed\features.csv"

SELECTED_CLASSES = [
    "car_horn",
    "siren",
    "door_wood_knock",
    "crying_baby",
    "dog",
    "glass_breaking"
]

SAMPLE_RATE = 16000

N_MFCC = 13

N_FFT = 1024

HOP_LENGTH = 512

N_MELS = 40


# ============================================================
# FEATURE EXTRACTION FUNCTION
# ============================================================

def extract_audio_features(audio_file):

    # Load processed audio
    audio, sample_rate = librosa.load(
        audio_file,
        sr=SAMPLE_RATE,
        mono=True
    )

    # --------------------------------------------------------
    # MFCC
    # --------------------------------------------------------

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )

    # First-order MFCC derivative
    mfcc_delta = librosa.feature.delta(
        mfcc
    )

    # Second-order MFCC derivative
    mfcc_delta2 = librosa.feature.delta(
        mfcc,
        order=2
    )

    # --------------------------------------------------------
    # Spectral features
    # --------------------------------------------------------

    spectral_centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sample_rate,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=audio,
        sr=sample_rate,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=audio,
        sr=sample_rate,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    zero_crossing_rate = librosa.feature.zero_crossing_rate(
        audio,
        frame_length=N_FFT,
        hop_length=HOP_LENGTH
    )

    rms_energy = librosa.feature.rms(
        y=audio,
        frame_length=N_FFT,
        hop_length=HOP_LENGTH
    )

    # --------------------------------------------------------
    # Store features
    # --------------------------------------------------------

    features = {}

    # --------------------------------------------------------
    # MFCC mean and standard deviation
    # --------------------------------------------------------

    for i in range(N_MFCC):

        features[f"mfcc_{i+1}_mean"] = np.mean(
            mfcc[i]
        )

        features[f"mfcc_{i+1}_std"] = np.std(
            mfcc[i]
        )

    # --------------------------------------------------------
    # MFCC delta
    # --------------------------------------------------------

    for i in range(N_MFCC):

        features[f"delta_{i+1}_mean"] = np.mean(
            mfcc_delta[i]
        )

        features[f"delta_{i+1}_std"] = np.std(
            mfcc_delta[i]
        )

    # --------------------------------------------------------
    # MFCC delta-delta
    # --------------------------------------------------------

    for i in range(N_MFCC):

        features[f"delta2_{i+1}_mean"] = np.mean(
            mfcc_delta2[i]
        )

        features[f"delta2_{i+1}_std"] = np.std(
            mfcc_delta2[i]
        )

    # --------------------------------------------------------
    # Spectral feature statistics
    # --------------------------------------------------------

    spectral_features = {
        "spectral_centroid": spectral_centroid,
        "spectral_bandwidth": spectral_bandwidth,
        "spectral_rolloff": spectral_rolloff,
        "zero_crossing_rate": zero_crossing_rate,
        "rms_energy": rms_energy
    }

    for name, values in spectral_features.items():

        features[f"{name}_mean"] = np.mean(
            values
        )

        features[f"{name}_std"] = np.std(
            values
        )

    return features


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print("========================================")
    print("       SOUND2SEE FEATURE EXTRACTION")
    print("========================================")

    all_features = []

    processed_count = 0
    failed_count = 0

    # --------------------------------------------------------
    # Process each class
    # --------------------------------------------------------

    for sound_class in SELECTED_CLASSES:

        class_folder = os.path.join(
            DATASET_FOLDER,
            sound_class
        )

        if not os.path.exists(class_folder):

            print(
                f"\nERROR: Missing class folder: "
                f"{sound_class}"
            )

            failed_count += 1
            continue

        files = sorted([
            file
            for file in os.listdir(class_folder)
            if file.lower().endswith(".wav")
        ])

        print(
            f"\nProcessing {sound_class}: "
            f"{len(files)} files"
        )

        # ----------------------------------------------------
        # Process every audio file
        # ----------------------------------------------------

        for filename in files:

            audio_file = os.path.join(
                class_folder,
                filename
            )

            try:

                features = extract_audio_features(
                    audio_file
                )

                # Add identifying information
                features["filename"] = filename

                features["class"] = sound_class

                all_features.append(
                    features
                )

                processed_count += 1

            except Exception as error:

                print(
                    f"ERROR: {filename}"
                )

                print(error)

                failed_count += 1

    # --------------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------------

    feature_data = pd.DataFrame(
        all_features
    )

    # --------------------------------------------------------
    # Reorder columns
    # --------------------------------------------------------

    identifier_columns = [
        "filename",
        "class"
    ]

    feature_columns = [
        column
        for column in feature_data.columns
        if column not in identifier_columns
    ]

    feature_data = feature_data[
        identifier_columns + feature_columns
    ]

    # --------------------------------------------------------
    # Check for invalid values
    # --------------------------------------------------------

    nan_count = feature_data.isna().sum().sum()

    infinite_count = np.isinf(
        feature_data.select_dtypes(
            include=np.number
        ).values
    ).sum()

    # --------------------------------------------------------
    # Save feature dataset
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    feature_data.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n========================================")
    print("         FEATURE EXTRACTION SUMMARY")
    print("========================================")

    print(
        f"Files processed : {processed_count}"
    )

    print(
        f"Files failed    : {failed_count}"
    )

    print(
        f"Feature rows    : {len(feature_data)}"
    )

    print(
        f"Total columns   : {len(feature_data.columns)}"
    )

    print(
        f"NaN values      : {nan_count}"
    )

    print(
        f"Infinite values : {infinite_count}"
    )

    print(
        f"Output file     : {OUTPUT_FILE}"
    )

    print()

    if (
        processed_count == 240
        and failed_count == 0
        and nan_count == 0
        and infinite_count == 0
    ):

        print(
            "STATUS: FEATURE EXTRACTION SUCCESSFUL"
        )

    else:

        print(
            "STATUS: CHECK FEATURE EXTRACTION"
        )

    print("========================================")


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
import os
import numpy as np
import pandas as pd
import librosa


# ============================================================
# SETTINGS
# ============================================================

DATASET_FOLDER = r"data\processed\esc50_16khz"

METADATA_FILE = (
    r"data\raw\esc50_selected\selected_metadata.csv"
)

OUTPUT_FILE = (
    r"data\processed\temporal_features.csv"
)

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

N_SEGMENTS = 3


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(audio_file):

    audio, sr = librosa.load(
        audio_file,
        sr=SAMPLE_RATE,
        mono=True
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
    # MFCC derivatives
    # --------------------------------------------------------

    delta = librosa.feature.delta(mfcc)

    delta2 = librosa.feature.delta(
        mfcc,
        order=2
    )

    features = {}

    # ========================================================
    # TEMPORAL MFCC FEATURES
    # ========================================================

    total_frames = mfcc.shape[1]

    segment_edges = np.linspace(
        0,
        total_frames,
        N_SEGMENTS + 1,
        dtype=int
    )

    for segment in range(N_SEGMENTS):

        start = segment_edges[segment]
        end = segment_edges[segment + 1]

        mfcc_segment = mfcc[:, start:end]

        for i in range(N_MFCC):

            features[
                f"mfcc_s{segment+1}_{i+1}_mean"
            ] = np.mean(
                mfcc_segment[i]
            )

            features[
                f"mfcc_s{segment+1}_{i+1}_std"
            ] = np.std(
                mfcc_segment[i]
            )

    # ========================================================
    # GLOBAL DELTA FEATURES
    # ========================================================

    for i in range(N_MFCC):

        features[
            f"delta_{i+1}_mean"
        ] = np.mean(delta[i])

        features[
            f"delta_{i+1}_std"
        ] = np.std(delta[i])

    # ========================================================
    # GLOBAL DELTA-DELTA FEATURES
    # ========================================================

    for i in range(N_MFCC):

        features[
            f"delta2_{i+1}_mean"
        ] = np.mean(delta2[i])

        features[
            f"delta2_{i+1}_std"
        ] = np.std(delta2[i])

    # ========================================================
    # SPECTRAL FEATURES
    # ========================================================

    spectral_centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=audio,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    spectral_rolloff = librosa.feature.spectral_rolloff(
        y=audio,
        sr=sr,
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

    spectral_features = {
        "spectral_centroid": spectral_centroid,
        "spectral_bandwidth": spectral_bandwidth,
        "spectral_rolloff": spectral_rolloff,
        "zero_crossing_rate": zero_crossing_rate,
        "rms_energy": rms_energy
    }

    for name, values in spectral_features.items():

        features[f"{name}_mean"] = np.mean(values)

        features[f"{name}_std"] = np.std(values)

    return features


# ============================================================
# MAIN
# ============================================================

def main():

    print("========================================")
    print("   SOUND2SEE TEMPORAL FEATURE EXTRACTION")
    print("========================================")

    metadata = pd.read_csv(
        METADATA_FILE
    )

    all_features = []

    processed = 0
    failed = 0

    for sound_class in SELECTED_CLASSES:

        class_folder = os.path.join(
            DATASET_FOLDER,
            sound_class
        )

        files = sorted([
            file
            for file in os.listdir(class_folder)
            if file.lower().endswith(".wav")
        ])

        print(
            f"\nProcessing {sound_class}: "
            f"{len(files)} files"
        )

        for filename in files:

            audio_file = os.path.join(
                class_folder,
                filename
            )

            try:

                features = extract_features(
                    audio_file
                )

                features["filename"] = filename
                features["class"] = sound_class

                all_features.append(features)

                processed += 1

            except Exception as error:

                print(
                    f"ERROR: {filename}"
                )

                print(error)

                failed += 1

    feature_data = pd.DataFrame(
        all_features
    )

    # --------------------------------------------------------
    # Add fold information
    # --------------------------------------------------------

    feature_data = feature_data.merge(
        metadata[["filename", "fold"]],
        on="filename",
        how="left"
    )

    # --------------------------------------------------------
    # Check invalid values
    # --------------------------------------------------------

    numeric_data = feature_data.select_dtypes(
        include=np.number
    )

    nan_count = feature_data.isna().sum().sum()

    infinite_count = np.isinf(
        numeric_data.values
    ).sum()

    # --------------------------------------------------------
    # Save
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
    print("       TEMPORAL FEATURE SUMMARY")
    print("========================================")

    print(
        f"Files processed : {processed}"
    )

    print(
        f"Files failed    : {failed}"
    )

    print(
        f"Rows            : {len(feature_data)}"
    )

    print(
        f"Columns         : {len(feature_data.columns)}"
    )

    print(
        f"NaN values      : {nan_count}"
    )

    print(
        f"Infinite values : {infinite_count}"
    )

    print(
        f"Output          : {OUTPUT_FILE}"
    )

    if (
        processed == 240
        and failed == 0
        and nan_count == 0
        and infinite_count == 0
    ):

        print(
            "\nSTATUS: TEMPORAL FEATURES SUCCESSFUL"
        )

    else:

        print(
            "\nSTATUS: CHECK FEATURE EXTRACTION"
        )

    print("========================================")


if __name__ == "__main__":
    main()
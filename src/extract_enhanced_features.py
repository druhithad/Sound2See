import os
import numpy as np
import pandas as pd
import librosa


# ============================================================
# SETTINGS
# ============================================================

METADATA_FILE = (
    r"data\processed\combined\combined_metadata.csv"
)

OUTPUT_FILE = (
    r"data\processed\combined\enhanced_features.csv"
)

SAMPLE_RATE = 16000

N_MFCC = 13
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 40

N_SEGMENTS = 3

N_CHROMA = 12
N_CONTRAST = 7


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(audio_file):

    audio, sr = librosa.load(
        audio_file,
        sr=SAMPLE_RATE,
        mono=True
    )

    features = {}

    # ========================================================
    # MFCC
    # ========================================================

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )

    delta = librosa.feature.delta(
        mfcc
    )

    delta2 = librosa.feature.delta(
        mfcc,
        order=2
    )

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

            if mfcc_segment.shape[1] == 0:

                features[
                    f"mfcc_s{segment + 1}_{i + 1}_mean"
                ] = 0.0

                features[
                    f"mfcc_s{segment + 1}_{i + 1}_std"
                ] = 0.0

            else:

                features[
                    f"mfcc_s{segment + 1}_{i + 1}_mean"
                ] = np.mean(
                    mfcc_segment[i]
                )

                features[
                    f"mfcc_s{segment + 1}_{i + 1}_std"
                ] = np.std(
                    mfcc_segment[i]
                )

    # ========================================================
    # GLOBAL MFCC DELTA
    # ========================================================

    for i in range(N_MFCC):

        features[
            f"delta_{i + 1}_mean"
        ] = np.mean(delta[i])

        features[
            f"delta_{i + 1}_std"
        ] = np.std(delta[i])

    # ========================================================
    # GLOBAL MFCC DELTA-DELTA
    # ========================================================

    for i in range(N_MFCC):

        features[
            f"delta2_{i + 1}_mean"
        ] = np.mean(delta2[i])

        features[
            f"delta2_{i + 1}_std"
        ] = np.std(delta2[i])

    # ========================================================
    # BASIC SPECTRAL FEATURES
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

    basic_spectral_features = {

        "spectral_centroid":
            spectral_centroid,

        "spectral_bandwidth":
            spectral_bandwidth,

        "spectral_rolloff":
            spectral_rolloff,

        "zero_crossing_rate":
            zero_crossing_rate,

        "rms_energy":
            rms_energy
    }

    for name, values in basic_spectral_features.items():

        features[
            f"{name}_mean"
        ] = np.mean(values)

        features[
            f"{name}_std"
        ] = np.std(values)

    # ========================================================
    # NEW FEATURE 1 — CHROMA
    # ========================================================

    chroma = librosa.feature.chroma_stft(
        y=audio,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_chroma=N_CHROMA
    )

    for i in range(N_CHROMA):

        features[
            f"chroma_{i + 1}_mean"
        ] = np.mean(chroma[i])

        features[
            f"chroma_{i + 1}_std"
        ] = np.std(chroma[i])

    # ========================================================
    # NEW FEATURE 2 — SPECTRAL CONTRAST
    # ========================================================

    nyquist = sr / 2

    safe_fmin = 200.0

    if safe_fmin >= nyquist:

        safe_fmin = max(
            50.0,
            nyquist / 4
        )

    max_bands = int(
        np.floor(
            np.log2(
                nyquist / safe_fmin
            )
        )
    )

    safe_n_bands = min(
        N_CONTRAST,
        max_bands
    )

    safe_n_bands = max(
        1,
        safe_n_bands
    )

    spectral_contrast = librosa.feature.spectral_contrast(
        y=audio,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_bands=safe_n_bands,
        fmin=safe_fmin
    )

    for i in range(
        spectral_contrast.shape[0]
    ):

        features[
            f"spectral_contrast_{i + 1}_mean"
        ] = np.mean(
            spectral_contrast[i]
        )

        features[
            f"spectral_contrast_{i + 1}_std"
        ] = np.std(
            spectral_contrast[i]
        )

    # ========================================================
    # NEW FEATURE 3 — TONNETZ
    # ========================================================

    tonnetz = librosa.feature.tonnetz(
        y=audio,
        sr=sr
    )

    for i in range(
        tonnetz.shape[0]
    ):

        features[
            f"tonnetz_{i + 1}_mean"
        ] = np.mean(
            tonnetz[i]
        )

        features[
            f"tonnetz_{i + 1}_std"
        ] = np.std(
            tonnetz[i]
        )

    return features


# ============================================================
# MAIN
# ============================================================

def main():

    print("========================================")
    print(" SOUND2SEE ENHANCED FEATURE EXTRACTION")
    print("========================================")

    metadata = pd.read_csv(
        METADATA_FILE
    )

    expected_files = len(metadata)

    print(
        "\nInput samples:",
        expected_files
    )

    all_features = []

    processed = 0
    failed = 0

    # --------------------------------------------------------
    # Process recordings
    # --------------------------------------------------------

    for index, row in metadata.iterrows():

        audio_file = row["file_path"]

        try:

            extracted = extract_features(
                audio_file
            )

            extracted["dataset"] = row["dataset"]

            extracted["filename"] = os.path.basename(
                audio_file
            )

            extracted["file_path"] = audio_file

            extracted["class"] = row["class"]

            extracted["split"] = row["split"]

            all_features.append(
                extracted
            )

            processed += 1

        except Exception as error:

            failed += 1

            print(
                f"\nERROR: {audio_file}"
            )

            print(
                "Reason:",
                error
            )

        if processed > 0 and processed % 100 == 0:

            print(
                f"Processed {processed}/{expected_files}"
            )

    # ========================================================
    # DATAFRAME
    # ========================================================

    feature_data = pd.DataFrame(
        all_features
    )

    # --------------------------------------------------------
    # Prevent empty DataFrame failure
    # --------------------------------------------------------

    if feature_data.empty:

        print(
            "\nERROR: No audio files were successfully processed."
        )

        print(
            "Enhanced feature extraction stopped."
        )

        return

    identifier_columns = [
        "dataset",
        "filename",
        "file_path",
        "class",
        "split"
    ]

    feature_columns = [
        column
        for column in feature_data.columns
        if column not in identifier_columns
    ]

    feature_data = feature_data[
        identifier_columns + feature_columns
    ]

    # ========================================================
    # VALIDATION
    # ========================================================

    numeric_data = feature_data[
        feature_columns
    ]

    nan_count = numeric_data.isna().sum().sum()

    infinite_count = np.isinf(
        numeric_data.values
    ).sum()

    # ========================================================
    # SAVE
    # ========================================================

    output_directory = os.path.dirname(
        OUTPUT_FILE
    )

    if output_directory:

        os.makedirs(
            output_directory,
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
    print(" ENHANCED FEATURE SUMMARY")
    print("========================================")

    print(
        "Files processed :",
        processed
    )

    print(
        "Files failed    :",
        failed
    )

    print(
        "Rows            :",
        len(feature_data)
    )

    print(
        "Feature columns :",
        len(feature_columns)
    )

    print(
        "Total columns   :",
        len(feature_data.columns)
    )

    print(
        "NaN values      :",
        nan_count
    )

    print(
        "Infinite values :",
        infinite_count
    )

    print(
        "\nDataset distribution:"
    )

    print(
        feature_data["dataset"].value_counts()
    )

    print(
        "\nSplit distribution:"
    )

    print(
        feature_data["split"].value_counts()
    )

    print(
        "\nClass distribution:"
    )

    print(
        feature_data["class"]
        .value_counts()
        .sort_index()
    )

    print(
        "\nOutput:",
        OUTPUT_FILE
    )

    # ========================================================
    # FEATURE VALIDATION
    # ========================================================

    print("\n========================================")
    print(" FEATURE VALIDATION")
    print("========================================")

    success = True

    if processed != expected_files:

        print(
            f"ERROR: Expected {expected_files} processed files, "
            f"got {processed}"
        )

        success = False

    else:

        print(
            f"All {expected_files} recordings processed."
        )

    if failed != 0:

        print(
            f"ERROR: {failed} files failed."
        )

        success = False

    else:

        print(
            "No extraction failures."
        )

    if nan_count != 0:

        print(
            f"ERROR: Found {nan_count} NaN values."
        )

        success = False

    else:

        print(
            "No NaN values."
        )

    if infinite_count != 0:

        print(
            f"ERROR: Found {infinite_count} "
            f"infinite values."
        )

        success = False

    else:

        print(
            "No infinite values."
        )

    if success:

        print(
            "\nSTATUS: ENHANCED FEATURE EXTRACTION SUCCESSFUL"
        )

    else:

        print(
            "\nSTATUS: CHECK FEATURE EXTRACTION"
        )

    print("========================================")


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
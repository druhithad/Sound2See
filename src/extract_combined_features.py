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
    r"data\processed\combined\combined_features.csv"
)

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

    delta = librosa.feature.delta(
        mfcc
    )

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
        ] = np.mean(
            delta[i]
        )

        features[
            f"delta_{i + 1}_std"
        ] = np.std(
            delta[i]
        )

    # ========================================================
    # GLOBAL MFCC DELTA-DELTA
    # ========================================================

    for i in range(N_MFCC):

        features[
            f"delta2_{i + 1}_mean"
        ] = np.mean(
            delta2[i]
        )

        features[
            f"delta2_{i + 1}_std"
        ] = np.std(
            delta2[i]
        )

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

    for name, values in spectral_features.items():

        features[
            f"{name}_mean"
        ] = np.mean(
            values
        )

        features[
            f"{name}_std"
        ] = np.std(
            values
        )

    return features


# ============================================================
# MAIN
# ============================================================

def main():

    print("========================================")
    print(" SOUND2SEE COMBINED FEATURE EXTRACTION")
    print("========================================")

    metadata = pd.read_csv(
        METADATA_FILE
    )

    print(
        "\nInput samples:",
        len(metadata)
    )

    all_features = []

    processed = 0

    failed = 0

    # --------------------------------------------------------
    # Process every recording
    # --------------------------------------------------------

    for index, row in metadata.iterrows():

        audio_file = row["file_path"]

        try:

            extracted = extract_features(
                audio_file
            )

            # ------------------------------------------------
            # Add metadata
            # ------------------------------------------------

            extracted["dataset"] = row[
                "dataset"
            ]

            extracted["filename"] = os.path.basename(
                audio_file
            )

            extracted["file_path"] = audio_file

            extracted["class"] = row[
                "class"
            ]

            extracted["split"] = row[
                "split"
            ]

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

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if processed > 0 and processed % 100 == 0:

            print(
                f"Processed {processed}/{len(metadata)}"
            )

    # --------------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------------

    feature_data = pd.DataFrame(
        all_features
    )

    # --------------------------------------------------------
    # Identify columns
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Reorder
    # --------------------------------------------------------

    feature_data = feature_data[
        identifier_columns + feature_columns
    ]

    # --------------------------------------------------------
    # Check numerical values
    # --------------------------------------------------------

    numeric_data = feature_data[
        feature_columns
    ]

    nan_count = numeric_data.isna().sum().sum()

    infinite_count = np.isinf(
        numeric_data.values
    ).sum()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(
            OUTPUT_FILE
        ),
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
    print(" COMBINED FEATURE SUMMARY")
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
        "Feature columns  :",
        len(feature_columns)
    )

    print(
        "Total columns    :",
        len(feature_data.columns)
    )

    print(
        "NaN values       :",
        nan_count
    )

    print(
        "Infinite values  :",
        infinite_count
    )

    print(
        "\nDataset distribution:"
    )

    print(
        feature_data[
            "dataset"
        ].value_counts()
    )

    print(
        "\nSplit distribution:"
    )

    print(
        feature_data[
            "split"
        ].value_counts()
    )

    print(
        "\nClass distribution:"
    )

    print(
        feature_data[
            "class"
        ].value_counts().sort_index()
    )

    print(
        "\nOutput:",
        OUTPUT_FILE
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    print("\n========================================")
    print(" FEATURE VALIDATION")
    print("========================================")

    success = True

    if processed != 1489:

        print(
            f"ERROR: Expected 1489 processed files, "
            f"got {processed}"
        )

        success = False

    else:

        print(
            "All 1489 recordings processed."
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

    if len(feature_columns) != 140:

        print(
            f"ERROR: Expected 140 features, "
            f"got {len(feature_columns)}"
        )

        success = False

    else:

        print(
            "140 features verified."
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
            "\nSTATUS: COMBINED FEATURE EXTRACTION SUCCESSFUL"
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
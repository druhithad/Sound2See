from pathlib import Path

import librosa
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

SPLIT_DIR = Path("data/processed/splits")
FEATURE_DIR = Path("data/processed/features")

FEATURE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# AUDIO / FEATURE SETTINGS
# ============================================================

SAMPLE_RATE = 16000

N_MFCC = 40

N_FFT = 2048

HOP_LENGTH = 512


# ============================================================
# FEATURE EXTRACTION FUNCTION
# ============================================================

def extract_mfcc(file_path):
    """
    Load an audio file and extract MFCC features.

    The mean and standard deviation are calculated across
    time so that every audio file produces a fixed-size
    feature vector.
    """

    audio, sr = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    mfcc_mean = np.mean(mfcc, axis=1)

    mfcc_std = np.std(mfcc, axis=1)

    features = np.concatenate(
        [mfcc_mean, mfcc_std]
    )

    return features


# ============================================================
# PROCESS ONE DATASET SPLIT
# ============================================================

def process_split(split_name):

    input_file = SPLIT_DIR / f"{split_name}.csv"

    output_file = FEATURE_DIR / f"{split_name}_features.npz"

    print("\n" + "=" * 50)
    print(f"Processing {split_name.upper()} dataset")
    print("=" * 50)

    df = pd.read_csv(input_file)

    features = []
    labels = []
    file_paths = []

    failed = 0

    for index, row in df.iterrows():

        file_path = Path(row["file_path"])
        label = row["label"]

        try:

            feature_vector = extract_mfcc(file_path)

            features.append(feature_vector)
            labels.append(label)
            file_paths.append(str(file_path))

        except Exception as error:

            failed += 1

            print(
                f"Failed: {file_path}"
            )

            print(
                f"Reason: {error}"
            )

        if (index + 1) % 100 == 0:

            print(
                f"Processed {index + 1}/{len(df)}"
            )

    X = np.array(features)

    y = np.array(labels)

    paths = np.array(file_paths)

    np.savez_compressed(
        output_file,
        X=X,
        y=y,
        paths=paths
    )

    print("\nCompleted:", split_name)

    print("Samples :", len(X))
    print("Features:", X.shape[1])
    print("Failed  :", failed)

    print("Saved to:", output_file)


# ============================================================
# MAIN
# ============================================================

print("\n======================================")
print("SOUND2SEE FEATURE EXTRACTION")
print("======================================")

print(f"Sample rate : {SAMPLE_RATE}")
print(f"MFCC count  : {N_MFCC}")
print(f"FFT size    : {N_FFT}")
print(f"Hop length  : {HOP_LENGTH}")

process_split("train")

process_split("validation")

process_split("test")


print("\n======================================")
print("FEATURE EXTRACTION COMPLETE")
print("======================================")
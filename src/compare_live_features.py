import pandas as pd
import numpy as np
import joblib
import librosa


FEATURE_FILE = r"data\processed\features.csv"
LIVE_FILE = r"data\inference\live_test.wav"


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(audio, sr=16000):

    n_mfcc = 13
    n_fft = 1024
    hop_length = 512
    n_mels = 40

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )

    delta = librosa.feature.delta(mfcc)

    delta2 = librosa.feature.delta(
        mfcc,
        order=2
    )

    features = []

    for i in range(n_mfcc):
        features.append(np.mean(mfcc[i]))
        features.append(np.std(mfcc[i]))

    for i in range(n_mfcc):
        features.append(np.mean(delta[i]))
        features.append(np.std(delta[i]))

    for i in range(n_mfcc):
        features.append(np.mean(delta2[i]))
        features.append(np.std(delta2[i]))

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

    rms = librosa.feature.rms(
        y=audio,
        frame_length=n_fft,
        hop_length=hop_length
    )

    for feature in [
        spectral_centroid,
        spectral_bandwidth,
        spectral_rolloff,
        zero_crossing_rate,
        rms
    ]:
        features.append(np.mean(feature))
        features.append(np.std(feature))

    return np.array(features)


# ============================================================
# LOAD DATA
# ============================================================

print("========================================")
print(" SOUND2SEE LIVE FEATURE COMPARISON")
print("========================================")

df = pd.read_csv(FEATURE_FILE)

feature_columns = df.columns[2:]

print("\nTraining rows :", len(df))
print("Features      :", len(feature_columns))
print("Classes       :", df["class"].unique().tolist())


# ============================================================
# LOAD LIVE AUDIO
# ============================================================

print("\nLoading live recording...")

audio, sr = librosa.load(
    LIVE_FILE,
    sr=16000,
    mono=True
)

print("Sample rate :", sr)
print("Samples     :", len(audio))
print("Duration    :", len(audio) / sr)


# ============================================================
# EXTRACT LIVE FEATURES
# ============================================================

live_features = extract_features(
    audio,
    sr
)

print(
    "\nLive features extracted:",
    len(live_features)
)


# ============================================================
# CHECK FEATURE COUNT
# ============================================================

if len(live_features) != len(feature_columns):

    print("\nERROR: Feature count mismatch!")

    print(
        "Live    :",
        len(live_features)
    )

    print(
        "Training:",
        len(feature_columns)
    )

    raise SystemExit


# ============================================================
# CREATE LIVE DATAFRAME
# ============================================================

live_df = pd.DataFrame(
    [live_features],
    columns=feature_columns
)


# ============================================================
# CLASS CENTROIDS
# ============================================================

centroids = (
    df.groupby("class")[feature_columns]
    .mean()
)


# ============================================================
# LOAD MODEL SCALER
# ============================================================

model = joblib.load(
    r"outputs\final_baseline\sound2see_svm_baseline.joblib"
)

scaler = model.named_steps.get(
    "scaler"
)

print("\nCalculating distances...")


# ============================================================
# SCALE FEATURES
# ============================================================

if scaler is not None:

    training_scaled = scaler.transform(
        centroids
    )

    live_scaled = scaler.transform(
        live_df
    )[0]

else:

    training_scaled = centroids.values
    live_scaled = live_features


# ============================================================
# EUCLIDEAN DISTANCE
# ============================================================

results = []

for sound_class, vector in zip(
    centroids.index,
    training_scaled
):

    distance = np.linalg.norm(
        live_scaled - vector
    )

    results.append(
        (
            sound_class,
            distance
        )
    )


results.sort(
    key=lambda x: x[1]
)


# ============================================================
# DISPLAY
# ============================================================

print("\n========================================")
print(" DISTANCE TO CLASS CENTROIDS")
print("========================================")

for sound_class, distance in results:

    print(
        f"{sound_class:20s} : "
        f"{distance:.4f}"
    )


print("\n========================================")
print(" CLOSEST CLASS")
print("========================================")

print(
    "Closest class:",
    results[0][0]
)

print(
    "Distance:",
    round(results[0][1], 4)
)

print("\n========================================")
print(" COMPARISON COMPLETE")
print("========================================")
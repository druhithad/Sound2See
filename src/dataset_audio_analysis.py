import os
import pandas as pd
import soundfile as sf
import numpy as np


# ============================================================
# ESC-50 DATASET LOCATION
# ============================================================

ESC50_PATH = r"C:\Users\D LAHARI\Downloads\ESC-50-master\ESC-50-master"

CSV_FILE = os.path.join(
    ESC50_PATH,
    "meta",
    "esc50.csv"
)

AUDIO_FOLDER = os.path.join(
    ESC50_PATH,
    "audio"
)


# ============================================================
# CLASSES TO INSPECT
# ============================================================

selected_classes = [
    "car_horn",
    "siren",
    "door_wood_knock",
    "crying_baby",
    "dog",
    "glass_breaking"
]


# ============================================================
# LOAD DATASET
# ============================================================

data = pd.read_csv(CSV_FILE)


print("========================================")
print("     SOUND2SEE AUDIO PROPERTY ANALYSIS")
print("========================================")


# ============================================================
# INSPECT ONE FILE FROM EACH CLASS
# ============================================================

for sound_class in selected_classes:

    class_files = data[
        data["category"] == sound_class
    ]

    if len(class_files) == 0:
        print(f"\nClass not found: {sound_class}")
        continue

    filename = class_files.iloc[0]["filename"]

    file_path = os.path.join(
        AUDIO_FOLDER,
        filename
    )

    audio, sample_rate = sf.read(file_path)

    number_of_samples = len(audio)

    duration = number_of_samples / sample_rate

    minimum_amplitude = np.min(audio)

    maximum_amplitude = np.max(audio)

    average_amplitude = np.mean(
        np.abs(audio)
    )

    # Determine channels
    if audio.ndim == 1:
        channels = 1
    else:
        channels = audio.shape[1]

    print()
    print("----------------------------------------")
    print(f"Class            : {sound_class}")
    print(f"File             : {filename}")
    print(f"Sample rate      : {sample_rate} Hz")
    print(f"Channels         : {channels}")
    print(f"Samples          : {number_of_samples}")
    print(f"Duration         : {duration:.2f} seconds")
    print(f"Minimum amplitude: {minimum_amplitude:.4f}")
    print(f"Maximum amplitude: {maximum_amplitude:.4f}")
    print(f"Average amplitude: {average_amplitude:.4f}")


print()
print("========================================")
print("          ANALYSIS COMPLETE")
print("========================================")
import os
import pandas as pd
import numpy as np
import soundfile as sf
import librosa
import librosa.display
import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================

DATASET_FOLDER = r"data\raw\esc50_selected"

OUTPUT_FOLDER = r"outputs\dataset_visualization"

SELECTED_CLASSES = [
    "car_horn",
    "siren",
    "door_wood_knock",
    "crying_baby",
    "dog",
    "glass_breaking"
]


# ============================================================
# CREATE OUTPUT FOLDER
# ============================================================

os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


# ============================================================
# PRINT HEADER
# ============================================================

print("========================================")
print("      SOUND2SEE DATASET VISUALIZATION")
print("========================================")


# ============================================================
# CHECK DATASET
# ============================================================

if not os.path.exists(DATASET_FOLDER):

    print("\nERROR: Dataset folder not found.")
    print(DATASET_FOLDER)
    exit()


# ============================================================
# COUNT FILES
# ============================================================

class_counts = {}

total_files = 0

for sound_class in SELECTED_CLASSES:

    class_folder = os.path.join(
        DATASET_FOLDER,
        sound_class
    )

    if not os.path.exists(class_folder):

        print(
            f"ERROR: Missing class folder: {sound_class}"
        )

        exit()

    files = [
        file
        for file in os.listdir(class_folder)
        if file.lower().endswith(".wav")
    ]

    class_counts[sound_class] = len(files)

    total_files += len(files)


# ============================================================
# PRINT CLASS DISTRIBUTION
# ============================================================

print("\n========================================")
print("          CLASS DISTRIBUTION")
print("========================================")

for sound_class, count in class_counts.items():

    print(
        f"{sound_class:<20} : {count} recordings"
    )

print(
    f"\nTotal recordings: {total_files}"
)


# ============================================================
# CLASS DISTRIBUTION GRAPH
# ============================================================

plt.figure(figsize=(10, 5))

plt.bar(
    list(class_counts.keys()),
    list(class_counts.values())
)

plt.title("Sound2See Environmental Sound Dataset")
plt.xlabel("Sound Class")
plt.ylabel("Number of Recordings")

plt.xticks(
    rotation=30,
    ha="right"
)

plt.tight_layout()

distribution_file = os.path.join(
    OUTPUT_FOLDER,
    "class_distribution.png"
)

plt.savefig(
    distribution_file,
    dpi=150
)

plt.close()

print(
    f"\nClass distribution saved to:"
    f"\n{distribution_file}"
)


# ============================================================
# ANALYZE ONE SAMPLE FROM EACH CLASS
# ============================================================

print("\n========================================")
print("       SAMPLE AUDIO ANALYSIS")
print("========================================")


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

    if len(files) == 0:

        print(
            f"No WAV files found for {sound_class}"
        )

        continue

    # Use first sample for visualization
    filename = files[0]

    audio_file = os.path.join(
        class_folder,
        filename
    )

    # --------------------------------------------------------
    # Load audio
    # --------------------------------------------------------

    audio, sample_rate = sf.read(
        audio_file
    )

    # Convert stereo to mono if necessary
    if audio.ndim > 1:

        audio = np.mean(
            audio,
            axis=1
        )

    duration = len(audio) / sample_rate

    print(
        f"{sound_class:<20} | "
        f"{filename:<25} | "
        f"{sample_rate} Hz | "
        f"{duration:.2f} sec"
    )

    # ========================================================
    # WAVEFORM
    # ========================================================

    time = np.arange(
        len(audio)
    ) / sample_rate

    plt.figure(figsize=(10, 4))

    plt.plot(
        time,
        audio
    )

    plt.title(
        f"Sound2See Waveform - {sound_class}"
    )

    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "Amplitude"
    )

    plt.tight_layout()

    waveform_file = os.path.join(
        OUTPUT_FOLDER,
        f"{sound_class}_waveform.png"
    )

    plt.savefig(
        waveform_file,
        dpi=150
    )

    plt.close()

    # ========================================================
    # SPECTROGRAM
    # ========================================================

    spectrogram = librosa.stft(
        audio
    )

    spectrogram_db = librosa.amplitude_to_db(
        np.abs(spectrogram),
        ref=np.max
    )

    plt.figure(figsize=(10, 4))

    librosa.display.specshow(
        spectrogram_db,
        sr=sample_rate,
        x_axis="time",
        y_axis="hz"
    )

    plt.colorbar(
        format="%+2.0f dB"
    )

    plt.title(
        f"Sound2See Spectrogram - {sound_class}"
    )

    plt.tight_layout()

    spectrogram_file = os.path.join(
        OUTPUT_FOLDER,
        f"{sound_class}_spectrogram.png"
    )

    plt.savefig(
        spectrogram_file,
        dpi=150
    )

    plt.close()


# ============================================================
# FINAL VERIFICATION
# ============================================================

print("\n========================================")
print("        DATASET VERIFICATION")
print("========================================")

verification_passed = True

for sound_class in SELECTED_CLASSES:

    count = class_counts[sound_class]

    if count == 40:

        print(
            f"[PASS] {sound_class:<20} "
            f"40 recordings"
        )

    else:

        print(
            f"[FAIL] {sound_class:<20} "
            f"{count} recordings"
        )

        verification_passed = False


print("\n========================================")

if verification_passed and total_files == 240:

    print("DATASET VERIFICATION: PASSED")

else:

    print("DATASET VERIFICATION: FAILED")

print("========================================")

print(
    "\nVisualization files saved in:"
)

print(
    OUTPUT_FOLDER
)
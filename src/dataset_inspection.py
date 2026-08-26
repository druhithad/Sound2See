import os
import pandas as pd


# ============================================================
# ESC-50 DATASET LOCATION
# ============================================================

# CHANGE THIS PATH if your ESC-50 folder is somewhere else.
ESC50_PATH = r"C:\Users\D LAHARI\Downloads\ESC-50-master\ESC-50-master"


# ============================================================
# FILE PATHS
# ============================================================

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
# CHECK DATASET
# ============================================================

print("========================================")
print("       SOUND2SEE DATASET INSPECTION")
print("========================================")

print()
print("Dataset location:")
print(ESC50_PATH)

print()


if not os.path.exists(ESC50_PATH):
    print("ERROR: ESC-50 folder was not found.")
    print("Please check the ESC50_PATH in this program.")
    exit()


if not os.path.exists(CSV_FILE):
    print("ERROR: esc50.csv was not found.")
    print("Expected location:")
    print(CSV_FILE)
    exit()


if not os.path.exists(AUDIO_FOLDER):
    print("ERROR: audio folder was not found.")
    print("Expected location:")
    print(AUDIO_FOLDER)
    exit()


print("ESC-50 folder       : FOUND")
print("Metadata file       : FOUND")
print("Audio folder        : FOUND")


# ============================================================
# LOAD METADATA
# ============================================================

data = pd.read_csv(CSV_FILE)


print()
print("========================================")
print("           DATASET INFORMATION")
print("========================================")

print(f"Total recordings : {len(data)}")
print(f"Total classes    : {data['category'].nunique()}")
print()


# ============================================================
# LIST ALL CLASSES
# ============================================================

print("========================================")
print("             SOUND CLASSES")
print("========================================")

classes = sorted(data["category"].unique())

for number, sound_class in enumerate(classes, start=1):
    count = (data["category"] == sound_class).sum()

    print(
        f"{number:02d}. {sound_class:<25} "
        f"{count} recordings"
    )


# ============================================================
# SOUND2SEE TARGET CLASSES
# ============================================================

target_classes = [
    "car_horn",
    "siren",
    "door_knock",
    "crying_baby",
    "dog",
    "glass_breaking",
    "clock_alarm"
]


print()
print("========================================")
print("       SOUND2SEE TARGET CLASSES")
print("========================================")


for target in target_classes:

    matches = data[
        data["category"].str.lower() == target.lower()
    ]

    if len(matches) > 0:
        print(
            f"[FOUND]   {target:<20} "
            f"{len(matches)} recordings"
        )
    else:
        print(
            f"[MISSING] {target:<20}"
        )


# ============================================================
# DATASET FOLD INFORMATION
# ============================================================

print()
print("========================================")
print("             FOLD INFORMATION")
print("========================================")

fold_counts = data["fold"].value_counts().sort_index()

for fold, count in fold_counts.items():
    print(
        f"Fold {fold}: {count} recordings"
    )


# ============================================================
# BASIC FILE CHECK
# ============================================================

print()
print("========================================")
print("          AUDIO FILE CHECK")
print("========================================")

available_files = 0
missing_files = 0

for filename in data["filename"]:

    file_path = os.path.join(
        AUDIO_FOLDER,
        filename
    )

    if os.path.exists(file_path):
        available_files += 1
    else:
        missing_files += 1


print(f"Audio files found : {available_files}")
print(f"Audio files missing: {missing_files}")


print()
print("========================================")
print("       INSPECTION COMPLETE")
print("========================================")
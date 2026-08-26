import os
import shutil
import pandas as pd


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
# SOUND2SEE OUTPUT LOCATION
# ============================================================

OUTPUT_FOLDER = os.path.join(
    "data",
    "raw",
    "esc50_selected"
)


# ============================================================
# INITIAL SOUND CLASSES
# ============================================================

SELECTED_CLASSES = [
    "car_horn",
    "siren",
    "door_wood_knock",
    "crying_baby",
    "dog",
    "glass_breaking"
]


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    print("========================================")
    print("       SOUND2SEE DATASET PREPARATION")
    print("========================================")

    # --------------------------------------------------------
    # Check ESC-50 dataset
    # --------------------------------------------------------

    if not os.path.exists(CSV_FILE):
        print("\nERROR: esc50.csv was not found.")
        print(CSV_FILE)
        return

    if not os.path.exists(AUDIO_FOLDER):
        print("\nERROR: ESC-50 audio folder was not found.")
        print(AUDIO_FOLDER)
        return

    # --------------------------------------------------------
    # Read metadata
    # --------------------------------------------------------

    data = pd.read_csv(CSV_FILE)

    print("\nESC-50 metadata loaded successfully.")
    print(f"Total ESC-50 recordings: {len(data)}")

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # --------------------------------------------------------
    # Select required classes
    # --------------------------------------------------------

    selected_data = data[
        data["category"].isin(SELECTED_CLASSES)
    ].copy()

    print(
        f"\nSelected recordings: {len(selected_data)}"
    )

    # --------------------------------------------------------
    # Create class folders
    # --------------------------------------------------------

    for sound_class in SELECTED_CLASSES:

        class_folder = os.path.join(
            OUTPUT_FOLDER,
            sound_class
        )

        os.makedirs(
            class_folder,
            exist_ok=True
        )

    # --------------------------------------------------------
    # Copy audio files
    # --------------------------------------------------------

    copied_count = 0
    missing_count = 0

    print("\nCopying selected audio files...")

    for _, row in selected_data.iterrows():

        filename = row["filename"]
        sound_class = row["category"]

        source_file = os.path.join(
            AUDIO_FOLDER,
            filename
        )

        destination_folder = os.path.join(
            OUTPUT_FOLDER,
            sound_class
        )

        destination_file = os.path.join(
            destination_folder,
            filename
        )

        if os.path.exists(source_file):

            shutil.copy2(
                source_file,
                destination_file
            )

            copied_count += 1

        else:

            print(
                f"Missing file: {filename}"
            )

            missing_count += 1

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    metadata_file = os.path.join(
        OUTPUT_FOLDER,
        "selected_metadata.csv"
    )

    selected_data.to_csv(
        metadata_file,
        index=False
    )

    # --------------------------------------------------------
    # Verify each class
    # --------------------------------------------------------

    print("\n========================================")
    print("           DATASET VERIFICATION")
    print("========================================")

    verification_success = True

    for sound_class in SELECTED_CLASSES:

        class_folder = os.path.join(
            OUTPUT_FOLDER,
            sound_class
        )

        files = [
            file
            for file in os.listdir(class_folder)
            if file.lower().endswith(".wav")
        ]

        count = len(files)

        print(
            f"{sound_class:<20} : {count} recordings"
        )

        if count != 40:
            verification_success = False

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n========================================")
    print("             FINAL SUMMARY")
    print("========================================")

    print(f"Selected recordings : {len(selected_data)}")
    print(f"Files copied        : {copied_count}")
    print(f"Files missing       : {missing_count}")

    print(
        f"Metadata saved      : {metadata_file}"
    )

    if verification_success and missing_count == 0:

        print("\nSTATUS: DATASET PREPARATION SUCCESSFUL")

    else:

        print("\nSTATUS: VERIFICATION FAILED")

    print("========================================")


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
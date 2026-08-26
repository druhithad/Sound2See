import os
import librosa
import soundfile as sf


# ============================================================
# SETTINGS
# ============================================================

INPUT_FOLDER = r"data\raw\esc50_selected"

OUTPUT_FOLDER = r"data\processed\esc50_16khz"

TARGET_SAMPLE_RATE = 16000

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
    print("       SOUND2SEE AUDIO PREPROCESSING")
    print("========================================")

    total_processed = 0
    total_failed = 0

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_FOLDER,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Process each class
    # --------------------------------------------------------

    for sound_class in SELECTED_CLASSES:

        input_class_folder = os.path.join(
            INPUT_FOLDER,
            sound_class
        )

        output_class_folder = os.path.join(
            OUTPUT_FOLDER,
            sound_class
        )

        os.makedirs(
            output_class_folder,
            exist_ok=True
        )

        if not os.path.exists(input_class_folder):

            print(
                f"\nERROR: Missing class folder: "
                f"{sound_class}"
            )

            total_failed += 1
            continue

        files = [
            file
            for file in os.listdir(
                input_class_folder
            )
            if file.lower().endswith(".wav")
        ]

        print(
            f"\nProcessing {sound_class}: "
            f"{len(files)} files"
        )

        # ----------------------------------------------------
        # Process every WAV file
        # ----------------------------------------------------

        for filename in files:

            input_file = os.path.join(
                input_class_folder,
                filename
            )

            output_file = os.path.join(
                output_class_folder,
                filename
            )

            try:

                # Load audio as mono and resample
                audio, sample_rate = librosa.load(
                    input_file,
                    sr=TARGET_SAMPLE_RATE,
                    mono=True
                )

                # Save processed audio
                sf.write(
                    output_file,
                    audio,
                    TARGET_SAMPLE_RATE
                )

                total_processed += 1

            except Exception as error:

                print(
                    f"ERROR processing {filename}: "
                    f"{error}"
                )

                total_failed += 1

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n========================================")
    print("          PREPROCESSING SUMMARY")
    print("========================================")

    print(
        f"Files processed : {total_processed}"
    )

    print(
        f"Files failed    : {total_failed}"
    )

    print(
        f"Output folder   : {OUTPUT_FOLDER}"
    )

    if total_processed == 240 and total_failed == 0:

        print(
            "\nSTATUS: PREPROCESSING SUCCESSFUL"
        )

    else:

        print(
            "\nSTATUS: CHECK PROCESSING RESULTS"
        )

    print("========================================")


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
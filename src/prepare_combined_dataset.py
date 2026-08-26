import os
import pandas as pd


# ============================================================
# INPUT FILES
# ============================================================

ANIMAL_TRAIN = r"data\processed\splits\train.csv"
ANIMAL_VALIDATION = r"data\processed\splits\validation.csv"
ANIMAL_TEST = r"data\processed\splits\test.csv"

ESC_METADATA = (
    r"data\raw\esc50_selected\selected_metadata.csv"
)


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_FOLDER = r"data\processed\combined"

OUTPUT_FILE = os.path.join(
    OUTPUT_FOLDER,
    "combined_metadata.csv"
)


# ============================================================
# ESC-50 SELECTED CLASSES
# ============================================================

ESC_CLASSES = [
    "car_horn",
    "siren",
    "door_wood_knock",
    "crying_baby",
    "dog",
    "glass_breaking"
]


# ============================================================
# ANIMAL CLASSES
# ============================================================

ANIMAL_CLASSES = [
    "Bird",
    "Cat",
    "Chicken",
    "Cow",
    "Dog",
    "Donkey",
    "Frog",
    "Lion",
    "Monkey",
    "Sheep"
]


# ============================================================
# LABEL NORMALIZATION
# ============================================================

def normalize_label(label):

    label = str(label).strip()

    # Make Dog consistent between datasets
    if label.lower() == "dog":
        return "Dog"

    return label


# ============================================================
# LOAD ANIMAL DATASET
# ============================================================

def load_animal_dataset():

    print("\nLoading Animal Sounds dataset...")

    train = pd.read_csv(ANIMAL_TRAIN)
    validation = pd.read_csv(ANIMAL_VALIDATION)
    test = pd.read_csv(ANIMAL_TEST)

    train["split"] = "train"
    validation["split"] = "validation"
    test["split"] = "test"

    data = pd.concat(
        [
            train,
            validation,
            test
        ],
        ignore_index=True
    )

    data["dataset"] = "animal"

    data["class"] = data["label"].apply(
        normalize_label
    )

    data = data[
        [
            "dataset",
            "file_path",
            "class",
            "split"
        ]
    ]

    return data


# ============================================================
# LOAD ESC-50 DATASET
# ============================================================

def load_esc50_dataset():

    print("\nLoading ESC-50 dataset...")

    metadata = pd.read_csv(
        ESC_METADATA
    )

    metadata = metadata[
        metadata["category"].isin(
            ESC_CLASSES
        )
    ].copy()

    # --------------------------------------------------------
    # ESC-50 fold mapping
    #
    # Folds 1-3 -> training
    # Fold 4    -> validation
    # Fold 5    -> test
    # --------------------------------------------------------

    def assign_split(fold):

        if fold in [1, 2, 3]:
            return "train"

        if fold == 4:
            return "validation"

        if fold == 5:
            return "test"

        return None

    metadata["split"] = metadata[
        "fold"
    ].apply(assign_split)

    metadata["dataset"] = "esc50"

    metadata["class"] = metadata[
        "category"
    ].apply(normalize_label)

    # --------------------------------------------------------
    # Actual processed ESC-50 audio location
    # --------------------------------------------------------

    metadata["file_path"] = metadata.apply(
        lambda row:
            os.path.join(
                "data",
                "processed",
                "esc50_16khz",
                row["category"],
                row["filename"]
            ),
        axis=1
    )

    data = metadata[
        [
            "dataset",
            "file_path",
            "class",
            "split"
        ]
    ].copy()

    return data


# ============================================================
# MAIN
# ============================================================

def main():

    print("========================================")
    print(" SOUND2SEE COMBINED DATASET PREPARATION")
    print("========================================")

    os.makedirs(
        OUTPUT_FOLDER,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load both datasets
    # --------------------------------------------------------

    animal = load_animal_dataset()

    esc50 = load_esc50_dataset()

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    combined = pd.concat(
        [
            animal,
            esc50
        ],
        ignore_index=True
    )

    # --------------------------------------------------------
    # Remove accidental duplicate rows
    # --------------------------------------------------------

    combined = combined.drop_duplicates(
        subset=[
            "dataset",
            "file_path"
        ]
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Verify files
    # --------------------------------------------------------

    combined["file_exists"] = combined[
        "file_path"
    ].apply(
        os.path.exists
    )

    missing = combined[
        ~combined["file_exists"]
    ]

    # Remove verification column
    combined = combined.drop(
        columns=["file_exists"]
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    combined.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n========================================")
    print(" COMBINED DATASET SUMMARY")
    print("========================================")

    print(
        "Total samples:",
        len(combined)
    )

    print(
        "Animal samples:",
        len(animal)
    )

    print(
        "ESC-50 samples:",
        len(esc50)
    )

    print(
        "\nSamples by dataset:"
    )

    print(
        combined[
            "dataset"
        ].value_counts()
    )

    print(
        "\nSamples by split:"
    )

    print(
        combined[
            "split"
        ].value_counts()
    )

    print(
        "\nClasses:"
    )

    print(
        sorted(
            combined[
                "class"
            ].unique()
        )
    )

    print(
        "\nNumber of classes:",
        combined["class"].nunique()
    )

    print(
        "\nClass distribution:"
    )

    print(
        combined[
            "class"
        ].value_counts().sort_index()
    )

    print(
        "\nMissing audio files:",
        len(missing)
    )

    if len(missing) > 0:

        print("\nMISSING FILES:")

        for path in missing[
            "file_path"
        ].tolist():

            print(
                path
            )

    print(
        "\nOutput:",
        OUTPUT_FILE
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    expected_classes = sorted(
        [
            "Bird",
            "Cat",
            "Chicken",
            "Cow",
            "Dog",
            "Donkey",
            "Frog",
            "Lion",
            "Monkey",
            "Sheep",
            "car_horn",
            "siren",
            "door_wood_knock",
            "crying_baby",
            "glass_breaking"
        ]
    )

    actual_classes = sorted(
        combined[
            "class"
        ].unique()
    )

    print("\n========================================")
    print(" VALIDATION")
    print("========================================")

    if actual_classes == expected_classes:
        print("15 classes verified.")
    else:
        print("ERROR: Class list does not match.")

    if len(missing) == 0:
        print("All audio files verified.")
    else:
        print(
            "ERROR:",
            len(missing),
            "audio files are missing."
        )

    if len(combined) == 1307:
        print(
            "Total sample count verified: 1307"
        )
    else:
        print(
            "WARNING: Expected 1307 samples, found",
            len(combined)
        )

    print("\n========================================")
    print(" COMBINED DATASET PREPARATION COMPLETE")
    print("========================================")


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
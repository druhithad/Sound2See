import os
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

FEATURE_FILE = r"data\processed\features.csv"

METADATA_FILE = (
    r"data\raw\esc50_selected\selected_metadata.csv"
)

OUTPUT_FOLDER = r"data\processed\splits"


# ============================================================
# FOLD ASSIGNMENT
# ============================================================

TRAIN_FOLDS = [1, 2, 3]

VALIDATION_FOLDS = [4]

TEST_FOLDS = [5]


# ============================================================
# LOAD DATA
# ============================================================

def main():

    print("========================================")
    print("       SOUND2SEE DATASET SPLITTING")
    print("========================================")

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not os.path.exists(FEATURE_FILE):

        print("\nERROR: Feature file not found.")
        print(FEATURE_FILE)
        return

    if not os.path.exists(METADATA_FILE):

        print("\nERROR: Metadata file not found.")
        print(METADATA_FILE)
        return

    # --------------------------------------------------------
    # Read files
    # --------------------------------------------------------

    features = pd.read_csv(
        FEATURE_FILE
    )

    metadata = pd.read_csv(
        METADATA_FILE
    )

    print(
        f"\nFeature rows: {len(features)}"
    )

    print(
        f"Metadata rows: {len(metadata)}"
    )

    # --------------------------------------------------------
    # Select required metadata columns
    # --------------------------------------------------------

    metadata_subset = metadata[
        ["filename", "fold"]
    ].copy()

    # --------------------------------------------------------
    # Merge fold information
    # --------------------------------------------------------

    dataset = features.merge(
        metadata_subset,
        on="filename",
        how="left"
    )

    # --------------------------------------------------------
    # Check missing fold values
    # --------------------------------------------------------

    missing_folds = dataset["fold"].isna().sum()

    print(
        f"Missing fold information: {missing_folds}"
    )

    if missing_folds > 0:

        print(
            "\nERROR: Some feature files do not "
            "have fold information."
        )

        return

    # Convert fold to integer
    dataset["fold"] = dataset["fold"].astype(int)

    # --------------------------------------------------------
    # Create splits
    # --------------------------------------------------------

    train = dataset[
        dataset["fold"].isin(TRAIN_FOLDS)
    ].copy()

    validation = dataset[
        dataset["fold"].isin(VALIDATION_FOLDS)
    ].copy()

    test = dataset[
        dataset["fold"].isin(TEST_FOLDS)
    ].copy()

    # --------------------------------------------------------
    # Create output folder
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_FOLDER,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save splits
    # --------------------------------------------------------

    train_file = os.path.join(
        OUTPUT_FOLDER,
        "train.csv"
    )

    validation_file = os.path.join(
        OUTPUT_FOLDER,
        "validation.csv"
    )

    test_file = os.path.join(
        OUTPUT_FOLDER,
        "test.csv"
    )

    train.to_csv(
        train_file,
        index=False
    )

    validation.to_csv(
        validation_file,
        index=False
    )

    test.to_csv(
        test_file,
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n========================================")
    print("             SPLIT SUMMARY")
    print("========================================")

    print(
        f"Training samples   : {len(train)}"
    )

    print(
        f"Validation samples : {len(validation)}"
    )

    print(
        f"Testing samples    : {len(test)}"
    )

    print(
        f"Total samples      : {len(dataset)}"
    )

    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    print("\n========================================")
    print("        TRAINING CLASS DISTRIBUTION")
    print("========================================")

    print(
        train["class"].value_counts().sort_index()
    )

    print("\n========================================")
    print("       VALIDATION DISTRIBUTION")
    print("========================================")

    print(
        validation["class"].value_counts().sort_index()
    )

    print("\n========================================")
    print("          TEST DISTRIBUTION")
    print("========================================")

    print(
        test["class"].value_counts().sort_index()
    )

    # --------------------------------------------------------
    # Fold distribution
    # --------------------------------------------------------

    print("\n========================================")
    print("            FOLD DISTRIBUTION")
    print("========================================")

    print(
        dataset["fold"].value_counts().sort_index()
    )

    # --------------------------------------------------------
    # Verify no overlap
    # --------------------------------------------------------

    train_files = set(train["filename"])

    validation_files = set(
        validation["filename"]
    )

    test_files = set(test["filename"])

    train_validation_overlap = (
        train_files & validation_files
    )

    train_test_overlap = (
        train_files & test_files
    )

    validation_test_overlap = (
        validation_files & test_files
    )

    print("\n========================================")
    print("          OVERLAP VERIFICATION")
    print("========================================")

    print(
        "Train / Validation overlap:",
        len(train_validation_overlap)
    )

    print(
        "Train / Test overlap:",
        len(train_test_overlap)
    )

    print(
        "Validation / Test overlap:",
        len(validation_test_overlap)
    )

    # --------------------------------------------------------
    # Final verification
    # --------------------------------------------------------

    verification_passed = (
        len(train) == 144
        and len(validation) == 48
        and len(test) == 48
        and missing_folds == 0
        and len(train_validation_overlap) == 0
        and len(train_test_overlap) == 0
        and len(validation_test_overlap) == 0
    )

    print("\n========================================")

    if verification_passed:

        print(
            "DATASET SPLIT VERIFICATION: PASSED"
        )

    else:

        print(
            "DATASET SPLIT VERIFICATION: FAILED"
        )

    print("========================================")


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
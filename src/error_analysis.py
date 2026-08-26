import os
import pandas as pd
import joblib

from sklearn.metrics import confusion_matrix


# ============================================================
# FILES
# ============================================================

TEST_FILE = r"data\processed\splits\test.csv"

MODEL_FILE = (
    r"outputs\final_baseline\sound2see_svm_baseline.joblib"
)

OUTPUT_FOLDER = r"outputs\error_analysis"

ERROR_FILE = os.path.join(
    OUTPUT_FOLDER,
    "misclassified_samples.csv"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("========================================")
    print("       SOUND2SEE ERROR ANALYSIS")
    print("========================================")

    os.makedirs(
        OUTPUT_FOLDER,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load test data
    # --------------------------------------------------------

    test_data = pd.read_csv(
        TEST_FILE
    )

    X_test = test_data.drop(
        columns=["filename", "class", "fold"]
    )

    y_test = test_data["class"]

    # --------------------------------------------------------
    # Load trained model
    # --------------------------------------------------------

    model = joblib.load(
        MODEL_FILE
    )

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    predictions = model.predict(
        X_test
    )

    # --------------------------------------------------------
    # Create error table
    # --------------------------------------------------------

    analysis = test_data[
        ["filename", "class", "fold"]
    ].copy()

    analysis["predicted_class"] = predictions

    analysis["correct"] = (
        analysis["class"]
        ==
        analysis["predicted_class"]
    )

    # --------------------------------------------------------
    # Save all predictions
    # --------------------------------------------------------

    all_predictions_file = os.path.join(
        OUTPUT_FOLDER,
        "test_predictions.csv"
    )

    analysis.to_csv(
        all_predictions_file,
        index=False
    )

    # --------------------------------------------------------
    # Extract errors
    # --------------------------------------------------------

    errors = analysis[
        analysis["correct"] == False
    ].copy()

    errors.to_csv(
        ERROR_FILE,
        index=False
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n========================================")
    print("             ERROR SUMMARY")
    print("========================================")

    print(
        f"Total test samples : {len(analysis)}"
    )

    print(
        f"Correct            : "
        f"{analysis['correct'].sum()}"
    )

    print(
        f"Incorrect          : "
        f"{len(errors)}"
    )

    # --------------------------------------------------------
    # Error pairs
    # --------------------------------------------------------

    print("\n========================================")
    print("          ERROR PAIRS")
    print("========================================")

    if len(errors) > 0:

        error_pairs = (
            errors
            .groupby(
                ["class", "predicted_class"]
            )
            .size()
            .sort_values(
                ascending=False
            )
        )

        for (
            (actual, predicted),
            count
        ) in error_pairs.items():

            print(
                f"{actual:<20} -> "
                f"{predicted:<20} : "
                f"{count}"
            )

    # --------------------------------------------------------
    # Per-class accuracy
    # --------------------------------------------------------

    print("\n========================================")
    print("          PER-CLASS RESULTS")
    print("========================================")

    class_results = (
        analysis
        .groupby("class")
        .agg(
            total=("correct", "size"),
            correct=("correct", "sum")
        )
    )

    class_results["accuracy"] = (
        class_results["correct"]
        /
        class_results["total"]
    )

    for class_name, row in class_results.iterrows():

        print(
            f"{class_name:<20} "
            f"{int(row['correct'])}/"
            f"{int(row['total'])} "
            f"({row['accuracy'] * 100:.2f}%)"
        )

    print("\n========================================")
    print("             FILES SAVED")
    print("========================================")

    print(
        "All predictions :",
        all_predictions_file
    )

    print(
        "Errors           :",
        ERROR_FILE
    )

    print("\n========================================")
    print("          ERROR ANALYSIS COMPLETE")
    print("========================================")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
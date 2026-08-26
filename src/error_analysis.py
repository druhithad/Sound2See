# ============================================================
# SOUND2SEE ERROR ANALYSIS
# ============================================================

import os
import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# CONFIGURATION
# ============================================================

FEATURE_FILE = r"data\processed\combined\combined_features.csv"

MODEL_FILE = r"outputs\combined_tuned\sound2see_combined_tuned_svm.joblib"


OUTPUT_FOLDER = r"outputs\error_analysis"


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
    # Load combined feature dataset
    # --------------------------------------------------------

    print()
    print("Loading combined feature dataset...")

    data = pd.read_csv(
        FEATURE_FILE
    )

    # --------------------------------------------------------
    # Select official test split
    # --------------------------------------------------------

    if "split" not in data.columns:
        raise ValueError(
            "The combined feature dataset does not contain "
            "the 'split' column."
        )

    test_data = data[
        data["split"] == "test"
    ].copy()

    print()
    print("========================================")
    print(" TEST DATA")
    print("========================================")

    print("Total test samples:", len(test_data))

    # --------------------------------------------------------
    # Load trained model
    # --------------------------------------------------------

    print()
    print("Loading trained model...")

    model = joblib.load(
        MODEL_FILE
    )

    # --------------------------------------------------------
    # Get exact feature names expected by model
    # --------------------------------------------------------

    if not hasattr(model, "feature_names_in_"):
        raise ValueError(
            "The trained model does not contain "
            "feature_names_in_."
        )

    model_features = list(
        model.feature_names_in_
    )

    print(
        "Model expects features:",
        len(model_features)
    )

    # --------------------------------------------------------
    # Verify all required features exist
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in model_features
        if feature not in test_data.columns
    ]

    if missing_features:

        print()
        print("Missing features:")
        for feature in missing_features:
            print(feature)

        raise ValueError(
            f"{len(missing_features)} model features "
            "are missing from the combined dataset."
        )

    # --------------------------------------------------------
    # Prepare features
    # --------------------------------------------------------

    X_test = test_data[
        model_features
    ].copy()

    # --------------------------------------------------------
    # Prepare labels
    # --------------------------------------------------------

    if "class" not in test_data.columns:
        raise ValueError(
            "The combined dataset does not contain "
            "the 'class' column."
        )

    y_test = test_data[
        "class"
    ].copy()

    print()
    print("Features:", X_test.shape[1])
    print("Classes:", y_test.nunique())

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    print()
    print("Running predictions...")

    predictions = model.predict(
        X_test
    )

    print("Prediction complete.")

    # --------------------------------------------------------
    # Overall performance
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    print()
    print("========================================")
    print(" OVERALL TEST PERFORMANCE")
    print("========================================")

    print(f"Accuracy        : {accuracy:.4f}")
    print(f"Accuracy        : {accuracy * 100:.2f}%")
    print(f"Precision       : {precision:.4f}")
    print(f"Recall          : {recall:.4f}")
    print(f"Weighted F1     : {weighted_f1:.4f}")
    print(f"Macro F1        : {macro_f1:.4f}")

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    report_dict = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(
        report_dict
    ).transpose()

    report_file = os.path.join(
        OUTPUT_FOLDER,
        "classification_report.csv"
    )

    report_df.to_csv(
        report_file
    )

    print()
    print("Classification report saved:")
    print(report_file)

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    class_names = sorted(
        y_test.unique()
    )

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=class_names
    )

    cm_df = pd.DataFrame(
        cm,
        index=class_names,
        columns=class_names
    )

    cm_file = os.path.join(
        OUTPUT_FOLDER,
        "confusion_matrix.csv"
    )

    cm_df.to_csv(
        cm_file
    )

    print("Confusion matrix saved:")
    print(cm_file)

    # --------------------------------------------------------
    # Create error analysis table
    # --------------------------------------------------------

    metadata_columns = [
        column
        for column in [
            "dataset",
            "filename",
            "file_path",
            "class",
            "split"
        ]
        if column in test_data.columns
    ]

    analysis = test_data[
        metadata_columns
    ].copy()

    analysis["predicted_class"] = predictions

    analysis["correct"] = (
        analysis["class"]
        ==
        analysis["predicted_class"]
    )

    # --------------------------------------------------------
    # Confidence scores
    # --------------------------------------------------------

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(
            X_test
        )

        analysis["confidence"] = (
            probabilities.max(axis=1)
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

    print()
    print("Test predictions saved:")
    print(all_predictions_file)

    # --------------------------------------------------------
    # Save only misclassified samples
    # --------------------------------------------------------

    errors = analysis[
        analysis["correct"] == False
    ].copy()

    errors_file = os.path.join(
        OUTPUT_FOLDER,
        "misclassified_samples.csv"
    )

    errors.to_csv(
        errors_file,
        index=False
    )

    print("Misclassified samples:", len(errors))

    print("Misclassified samples saved:")
    print(errors_file)

    # --------------------------------------------------------
    # Per-class results
    # --------------------------------------------------------

    print()
    print("========================================")
    print(" PER-CLASS RESULTS")
    print("========================================")

    class_results = (
        analysis
        .groupby("class")
        .agg(
            total=("correct", "count"),
            correct=("correct", "sum")
        )
    )

    class_results["errors"] = (
        class_results["total"]
        -
        class_results["correct"]
    )

    class_results["accuracy"] = (
        class_results["correct"]
        /
        class_results["total"]
    )

    class_results = class_results.sort_values(
        "accuracy"
    )

    class_results_file = os.path.join(
        OUTPUT_FOLDER,
        "per_class_results.csv"
    )

    class_results.to_csv(
        class_results_file
    )

    for class_name, row in class_results.iterrows():

        print(
            f"{class_name:<20} "
            f"Accuracy: {row['accuracy'] * 100:6.2f}% "
            f"Errors: {int(row['errors'])}/{int(row['total'])}"
        )

    # --------------------------------------------------------
    # Most common confusion pairs
    # --------------------------------------------------------

    print()
    print("========================================")
    print(" TOP CONFUSION PAIRS")
    print("========================================")

    confusion_pairs = (
        analysis[
            analysis["correct"] == False
        ]
        .groupby(
            ["class", "predicted_class"]
        )
        .size()
        .reset_index(
            name="count"
        )
        .sort_values(
            "count",
            ascending=False
        )
    )

    confusion_pairs_file = os.path.join(
        OUTPUT_FOLDER,
        "confusion_pairs.csv"
    )

    confusion_pairs.to_csv(
        confusion_pairs_file,
        index=False
    )

    if len(confusion_pairs) > 0:

        for _, row in confusion_pairs.head(15).iterrows():

            print(
                f"{row['class']} -> "
                f"{row['predicted_class']} : "
                f"{int(row['count'])}"
            )

    else:

        print("No misclassified samples.")

    # --------------------------------------------------------
    # Low confidence errors
    # --------------------------------------------------------

    if "confidence" in analysis.columns:

        low_confidence_errors = (
            errors
            .sort_values(
                "confidence"
            )
        )

        low_confidence_file = os.path.join(
            OUTPUT_FOLDER,
            "low_confidence_errors.csv"
        )

        low_confidence_errors.to_csv(
            low_confidence_file,
            index=False
        )

        print()
        print(
            "Low-confidence error analysis saved:"
        )
        print(
            low_confidence_file
        )

    # --------------------------------------------------------
    # Completion
    # --------------------------------------------------------

    print()
    print("========================================")
    print(" ERROR ANALYSIS COMPLETE")
    print("========================================")

    print()
    print("Output folder:")
    print(OUTPUT_FOLDER)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
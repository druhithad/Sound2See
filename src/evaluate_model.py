import os
import joblib
import numpy as np
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
# SETTINGS
# ============================================================

FEATURE_FILE = (
    r"data\processed\combined\enhanced_features.csv"
)

MODEL_FILE = (
    r"models\logisticregression_baseline.joblib"
)

OUTPUT_DIR = (
    r"data\processed\model_results"
)

IDENTIFIER_COLUMNS = [
    "dataset",
    "filename",
    "file_path",
    "class",
    "split"
]


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("========================================")
    print(" SOUND2SEE MODEL ERROR ANALYSIS")
    print("========================================")

    print("\nLoading feature data...")

    df = pd.read_csv(
        FEATURE_FILE
    )

    print(
        "Total samples:",
        len(df)
    )

    print(
        "Total columns:",
        len(df.columns)
    )

    feature_columns = [
        column
        for column in df.columns
        if column not in IDENTIFIER_COLUMNS
    ]

    test_df = df[
        df["split"] == "test"
    ].copy()

    X_test = test_df[
        feature_columns
    ]

    y_test = test_df[
        "class"
    ]

    print(
        "Test samples:",
        len(test_df)
    )

    print(
        "Features:",
        len(feature_columns)
    )

    return (
        df,
        test_df,
        X_test,
        y_test,
        feature_columns
    )


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("\n========================================")
    print(" LOADING MODEL")
    print("========================================")

    print(
        "Model:",
        MODEL_FILE
    )

    model = joblib.load(
        MODEL_FILE
    )

    print(
        "Model loaded successfully."
    )

    return model


# ============================================================
# PREDICTIONS
# ============================================================

def generate_predictions(
    model,
    X_test
):

    print("\n========================================")
    print(" GENERATING PREDICTIONS")
    print("========================================")

    predictions = model.predict(
        X_test
    )

    probabilities = None

    if hasattr(
        model,
        "predict_proba"
    ):

        probabilities = model.predict_proba(
            X_test
        )

    print(
        "Predictions generated:",
        len(predictions)
    )

    return (
        predictions,
        probabilities
    )


# ============================================================
# BASIC METRICS
# ============================================================

def calculate_metrics(
    y_test,
    predictions
):

    print("\n========================================")
    print(" TEST PERFORMANCE")
    print("========================================")

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    macro_precision = precision_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    print(
        f"Accuracy          : {accuracy:.4f}"
    )

    print(
        f"Macro Precision   : {macro_precision:.4f}"
    )

    print(
        f"Macro Recall      : {macro_recall:.4f}"
    )

    print(
        f"Macro F1          : {macro_f1:.4f}"
    )

    print(
        f"Weighted F1       : {weighted_f1:.4f}"
    )


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

def generate_classification_report(
    y_test,
    predictions
):

    print("\n========================================")
    print(" CLASSIFICATION REPORT")
    print("========================================")

    report = classification_report(
        y_test,
        predictions,
        zero_division=0
    )

    print(report)

    return report


# ============================================================
# CONFUSION MATRIX
# ============================================================

def analyze_confusion_matrix(
    y_test,
    predictions
):

    print("\n========================================")
    print(" CONFUSION MATRIX")
    print("========================================")

    labels = sorted(
        y_test.unique()
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=labels
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=labels,
        columns=labels
    )

    print(
        matrix_df
    )

    return (
        labels,
        matrix_df
    )


# ============================================================
# MOST CONFUSED CLASS PAIRS
# ============================================================

def find_confused_pairs(
    matrix_df
):

    print("\n========================================")
    print(" MOST CONFUSED CLASS PAIRS")
    print("========================================")

    confused_pairs = []

    labels = matrix_df.index.tolist()

    for actual in labels:

        for predicted in labels:

            if actual == predicted:
                continue

            count = matrix_df.loc[
                actual,
                predicted
            ]

            if count > 0:

                confused_pairs.append(
                    {
                        "actual": actual,
                        "predicted": predicted,
                        "count": int(count)
                    }
                )

    confused_pairs_df = pd.DataFrame(
        confused_pairs
    )

    if len(confused_pairs_df) > 0:

        confused_pairs_df = (
            confused_pairs_df
            .sort_values(
                "count",
                ascending=False
            )
        )

        print(
            confused_pairs_df.head(15)
        )

    else:

        print(
            "No classification errors found."
        )

    return confused_pairs_df


# ============================================================
# MISCLASSIFIED SAMPLES
# ============================================================

def analyze_misclassifications(
    test_df,
    predictions,
    probabilities,
    model
):

    print("\n========================================")
    print(" MISCLASSIFIED SAMPLES")
    print("========================================")

    results = test_df.copy()

    results[
        "predicted_class"
    ] = predictions

    results[
        "correct"
    ] = (
        results["class"]
        ==
        results["predicted_class"]
    )

    if probabilities is not None:

        results[
            "prediction_confidence"
        ] = np.max(
            probabilities,
            axis=1
        )

    else:

        results[
            "prediction_confidence"
        ] = np.nan

    misclassified = results[
        ~results["correct"]
    ].copy()

    print(
        "Total test samples:",
        len(results)
    )

    print(
        "Correct predictions:",
        results["correct"].sum()
    )

    print(
        "Incorrect predictions:",
        len(misclassified)
    )

    print(
        "\nTop misclassified samples:"
    )

    columns = [
        "filename",
        "class",
        "predicted_class",
        "prediction_confidence"
    ]

    print(
        misclassified[
            columns
        ].head(20)
    )

    return (
        results,
        misclassified
    )


# ============================================================
# LOW CONFIDENCE PREDICTIONS
# ============================================================

def analyze_confidence(
    results
):

    print("\n========================================")
    print(" PREDICTION CONFIDENCE")
    print("========================================")

    if (
        "prediction_confidence"
        not in results.columns
    ):

        print(
            "Confidence unavailable."
        )

        return

    confidence = results[
        "prediction_confidence"
    ]

    print(
        f"Mean confidence: "
        f"{confidence.mean():.4f}"
    )

    print(
        f"Minimum confidence: "
        f"{confidence.min():.4f}"
    )

    print(
        f"Maximum confidence: "
        f"{confidence.max():.4f}"
    )

    low_confidence = results[
        confidence < 0.50
    ]

    print(
        "Predictions below 50% confidence:",
        len(low_confidence)
    )

    print(
        "\nLowest-confidence predictions:"
    )

    print(
        low_confidence[
            [
                "filename",
                "class",
                "predicted_class",
                "prediction_confidence"
            ]
        ]
        .sort_values(
            "prediction_confidence"
        )
        .head(20)
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    results,
    misclassified,
    confused_pairs_df,
    matrix_df
):

    print("\n========================================")
    print(" SAVING ERROR ANALYSIS")
    print("========================================")

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    predictions_file = os.path.join(
        OUTPUT_DIR,
        "test_predictions.csv"
    )

    misclassified_file = os.path.join(
        OUTPUT_DIR,
        "misclassified_samples.csv"
    )

    confusion_file = os.path.join(
        OUTPUT_DIR,
        "confusion_matrix.csv"
    )

    confused_pairs_file = os.path.join(
        OUTPUT_DIR,
        "most_confused_pairs.csv"
    )

    results.to_csv(
        predictions_file,
        index=False
    )

    misclassified.to_csv(
        misclassified_file,
        index=False
    )

    matrix_df.to_csv(
        confusion_file
    )

    confused_pairs_df.to_csv(
        confused_pairs_file,
        index=False
    )

    print(
        "Saved:",
        predictions_file
    )

    print(
        "Saved:",
        misclassified_file
    )

    print(
        "Saved:",
        confusion_file
    )

    print(
        "Saved:",
        confused_pairs_file
    )


# ============================================================
# MAIN
# ============================================================

def main():

    (
        df,
        test_df,
        X_test,
        y_test,
        feature_columns
    ) = load_data()

    model = load_model()

    (
        predictions,
        probabilities
    ) = generate_predictions(
        model,
        X_test
    )

    calculate_metrics(
        y_test,
        predictions
    )

    generate_classification_report(
        y_test,
        predictions
    )

    (
        labels,
        matrix_df
    ) = analyze_confusion_matrix(
        y_test,
        predictions
    )

    confused_pairs_df = find_confused_pairs(
        matrix_df
    )

    (
        results,
        misclassified
    ) = analyze_misclassifications(
        test_df,
        predictions,
        probabilities,
        model
    )

    analyze_confidence(
        results
    )

    save_results(
        results,
        misclassified,
        confused_pairs_df,
        matrix_df
    )

    print("\n========================================")
    print(" ERROR ANALYSIS COMPLETE")
    print("========================================")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
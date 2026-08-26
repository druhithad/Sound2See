import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# FILES
# ============================================================

FEATURE_FILE = (
    r"data\processed\temporal_features.csv"
)

OUTPUT_FOLDER = (
    r"outputs\final_temporal"
)

MODEL_FILE = os.path.join(
    OUTPUT_FOLDER,
    "sound2see_temporal_svm.joblib"
)

RESULT_FILE = os.path.join(
    OUTPUT_FOLDER,
    "final_temporal_results.csv"
)

CONFUSION_MATRIX_FILE = os.path.join(
    OUTPUT_FOLDER,
    "confusion_matrix.png"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("========================================")
    print("      SOUND2SEE FINAL TEMPORAL MODEL")
    print("========================================")

    os.makedirs(
        OUTPUT_FOLDER,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load temporal feature dataset
    # --------------------------------------------------------

    data = pd.read_csv(
        FEATURE_FILE
    )

    # --------------------------------------------------------
    # Training data
    # Folds 1-4 = 192 samples
    # --------------------------------------------------------

    train = data[
        data["fold"].isin([1, 2, 3, 4])
    ].copy()

    # --------------------------------------------------------
    # Test data
    # Fold 5 = 48 samples
    # --------------------------------------------------------

    test = data[
        data["fold"] == 5
    ].copy()

    drop_columns = [
        "filename",
        "class",
        "fold"
    ]

    X_train = train.drop(
        columns=drop_columns
    )

    y_train = train["class"]

    X_test = test.drop(
        columns=drop_columns
    )

    y_test = test["class"]

    # --------------------------------------------------------
    # Dataset information
    # --------------------------------------------------------

    print("\nDATASET INFORMATION")

    print(
        "Training samples :",
        len(X_train)
    )

    print(
        "Test samples     :",
        len(X_test)
    )

    print(
        "Features         :",
        X_train.shape[1]
    )

    print(
        "Test fold        : 5 (reserved)"
    )

    # --------------------------------------------------------
    # Create SVM
    # --------------------------------------------------------

    model = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "classifier",
        SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            random_state=42
        )
    )
])

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("\nTraining final temporal SVM...")

    model.fit(
        X_train,
        y_train
    )

    print(
        "Training complete."
    )

    # --------------------------------------------------------
    # FINAL TEST PREDICTION
    # --------------------------------------------------------

    predictions = model.predict(
        X_test
    )

    # --------------------------------------------------------
    # Metrics
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

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    print("\n========================================")
    print("      FINAL TEMPORAL TEST RESULTS")
    print("========================================")

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Accuracy : {accuracy * 100:.2f}%"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1-score : {f1:.4f}"
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    print("\n========================================")
    print("       CLASSIFICATION REPORT")
    print("========================================")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=model.classes_
    )

    print("\n========================================")
    print("          CONFUSION MATRIX")
    print("========================================")

    print(matrix)

    # --------------------------------------------------------
    # Save confusion matrix
    # --------------------------------------------------------

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=model.classes_
    )

    display.plot(
        xticks_rotation=45
    )

    plt.title(
        "Sound2See - Final Temporal SVM"
    )

    plt.tight_layout()

    plt.savefig(
        CONFUSION_MATRIX_FILE,
        dpi=150
    )

    plt.close()

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results = pd.DataFrame([
        {
            "model": "Temporal SVM",
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "features": X_train.shape[1],
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
    ])

    results.to_csv(
        RESULT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    joblib.dump(
        model,
        MODEL_FILE
    )

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    print("\n========================================")
    print("             FILES SAVED")
    print("========================================")

    print(
        "Results          :",
        RESULT_FILE
    )

    print(
        "Confusion matrix :",
        CONFUSION_MATRIX_FILE
    )

    print(
        "Model            :",
        MODEL_FILE
    )

    print("\n========================================")
    print("       FINAL TEMPORAL MODEL COMPLETE")
    print("========================================")


if __name__ == "__main__":
    main()
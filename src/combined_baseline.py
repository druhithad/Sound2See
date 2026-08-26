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
    r"data\processed\combined\combined_features.csv"
)

OUTPUT_FOLDER = (
    r"outputs\combined_baseline"
)

MODEL_FILE = os.path.join(
    OUTPUT_FOLDER,
    "sound2see_combined_baseline.joblib"
)

RESULT_FILE = os.path.join(
    OUTPUT_FOLDER,
    "combined_baseline_results.csv"
)

CONFUSION_MATRIX_FILE = os.path.join(
    OUTPUT_FOLDER,
    "confusion_matrix.png"
)

CLASSIFICATION_REPORT_FILE = os.path.join(
    OUTPUT_FOLDER,
    "classification_report.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset():

    data = pd.read_csv(
        FEATURE_FILE
    )

    feature_columns = [
        column
        for column in data.columns
        if column not in [
            "dataset",
            "filename",
            "file_path",
            "class",
            "split"
        ]
    ]

    train = data[
        data["split"] == "train"
    ].copy()

    validation = data[
        data["split"] == "validation"
    ].copy()

    test = data[
        data["split"] == "test"
    ].copy()

    X_train = train[feature_columns]
    y_train = train["class"]

    X_validation = validation[feature_columns]
    y_validation = validation["class"]

    X_test = test[feature_columns]
    y_test = test["class"]

    return (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
        feature_columns
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("========================================")
    print(" SOUND2SEE COMBINED BASELINE SVM")
    print("========================================")

    os.makedirs(
        OUTPUT_FOLDER,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
        feature_columns
    ) = load_dataset()

    print("\nDATASET INFORMATION")
    print("----------------------------------------")

    print(
        "Training samples   :",
        len(X_train)
    )

    print(
        "Validation samples :",
        len(X_validation)
    )

    print(
        "Test samples       :",
        len(X_test)
    )

    print(
        "Features           :",
        len(feature_columns)
    )

    print(
        "Classes            :",
        len(y_train.unique())
    )

    print(
        "Class names        :",
        sorted(y_train.unique())
    )

    # --------------------------------------------------------
    # Create SVM pipeline
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
                probability=True,
                random_state=42
            )
        )
    ])

    # --------------------------------------------------------
    # Train baseline
    # --------------------------------------------------------

    print("\n========================================")
    print(" TRAINING BASELINE MODEL")
    print("========================================")

    print(
        "Model: StandardScaler + RBF SVM"
    )

    print(
        "C: default"
    )

    print(
        "Gamma: scale"
    )

    model.fit(
        X_train,
        y_train
    )

    print(
        "\nTraining complete."
    )

    # --------------------------------------------------------
    # Validation prediction
    # --------------------------------------------------------

    validation_predictions = model.predict(
        X_validation
    )

    validation_accuracy = accuracy_score(
        y_validation,
        validation_predictions
    )

    validation_precision = precision_score(
        y_validation,
        validation_predictions,
        average="weighted",
        zero_division=0
    )

    validation_recall = recall_score(
        y_validation,
        validation_predictions,
        average="weighted",
        zero_division=0
    )

    validation_f1 = f1_score(
        y_validation,
        validation_predictions,
        average="weighted",
        zero_division=0
    )

    # --------------------------------------------------------
    # Validation results
    # --------------------------------------------------------

    print("\n========================================")
    print(" VALIDATION PERFORMANCE")
    print("========================================")

    print(
        f"Accuracy : {validation_accuracy:.4f}"
    )

    print(
        f"Precision: {validation_precision:.4f}"
    )

    print(
        f"Recall   : {validation_recall:.4f}"
    )

    print(
        f"F1-score : {validation_f1:.4f}"
    )

    # --------------------------------------------------------
    # Validation classification report
    # --------------------------------------------------------

    print("\n========================================")
    print(" VALIDATION CLASSIFICATION REPORT")
    print("========================================")

    print(
        classification_report(
            y_validation,
            validation_predictions,
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # Final training
    #
    # Validation is now used for final training.
    # Test remains untouched.
    # --------------------------------------------------------

    X_train_final = pd.concat(
        [
            X_train,
            X_validation
        ],
        ignore_index=True
    )

    y_train_final = pd.concat(
        [
            y_train,
            y_validation
        ],
        ignore_index=True
    )

    print("\n========================================")
    print(" FINAL BASELINE TRAINING")
    print("========================================")

    print(
        "Training samples:",
        len(X_train_final)
    )

    print(
        "Test samples:",
        len(X_test)
    )

    final_model = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            SVC(
                kernel="rbf",
                probability=True,
                random_state=42
            )
        )
    ])

    final_model.fit(
        X_train_final,
        y_train_final
    )

    print(
        "Final training complete."
    )

    # --------------------------------------------------------
    # Final test prediction
    # --------------------------------------------------------

    test_predictions = final_model.predict(
        X_test
    )

    # --------------------------------------------------------
    # Test metrics
    # --------------------------------------------------------

    test_accuracy = accuracy_score(
        y_test,
        test_predictions
    )

    test_precision = precision_score(
        y_test,
        test_predictions,
        average="weighted",
        zero_division=0
    )

    test_recall = recall_score(
        y_test,
        test_predictions,
        average="weighted",
        zero_division=0
    )

    test_f1 = f1_score(
        y_test,
        test_predictions,
        average="weighted",
        zero_division=0
    )

    # --------------------------------------------------------
    # Final test results
    # --------------------------------------------------------

    print("\n========================================")
    print(" FINAL TEST PERFORMANCE")
    print("========================================")

    print(
        f"Accuracy : {test_accuracy:.4f}"
    )

    print(
        f"Accuracy : {test_accuracy * 100:.2f}%"
    )

    print(
        f"Precision: {test_precision:.4f}"
    )

    print(
        f"Recall   : {test_recall:.4f}"
    )

    print(
        f"F1-score : {test_f1:.4f}"
    )

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    print("\n========================================")
    print(" FINAL TEST CLASSIFICATION REPORT")
    print("========================================")

    report_dict = classification_report(
        y_test,
        test_predictions,
        zero_division=0,
        output_dict=True
    )

    report = classification_report(
        y_test,
        test_predictions,
        zero_division=0
    )

    print(report)

    report_dataframe = pd.DataFrame(
        report_dict
    ).transpose()

    report_dataframe.to_csv(
        CLASSIFICATION_REPORT_FILE
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    matrix = confusion_matrix(
        y_test,
        test_predictions,
        labels=final_model.classes_
    )

    print("\n========================================")
    print(" CONFUSION MATRIX")
    print("========================================")

    print(matrix)

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=final_model.classes_
    )

    display.plot(
        xticks_rotation=45
    )

    plt.title(
        "Sound2See - Combined 15-Class Baseline SVM"
    )

    plt.tight_layout()

    plt.savefig(
        CONFUSION_MATRIX_FILE,
        dpi=150
    )

    plt.close()

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    results = pd.DataFrame([
        {
            "model": "Combined Baseline SVM",
            "datasets": "Animal Sounds + ESC-50",
            "classes": 15,
            "features": len(feature_columns),
            "training_samples": len(X_train_final),
            "test_samples": len(X_test),
            "accuracy": test_accuracy,
            "precision": test_precision,
            "recall": test_recall,
            "f1_score": test_f1
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
        final_model,
        MODEL_FILE
    )

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    print("\n========================================")
    print(" FILES SAVED")
    print("========================================")

    print(
        "Results            :",
        RESULT_FILE
    )

    print(
        "Classification     :",
        CLASSIFICATION_REPORT_FILE
    )

    print(
        "Confusion matrix   :",
        CONFUSION_MATRIX_FILE
    )

    print(
        "Model              :",
        MODEL_FILE
    )

    print("\n========================================")
    print(" COMBINED BASELINE COMPLETE")
    print("========================================")


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
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

TRAIN_FILE = r"data\processed\splits\train.csv"
VALIDATION_FILE = r"data\processed\splits\validation.csv"
TEST_FILE = r"data\processed\splits\test.csv"

OUTPUT_FOLDER = r"outputs\final_baseline"

MODEL_FILE = os.path.join(
    OUTPUT_FOLDER,
    "sound2see_svm_baseline.joblib"
)

RESULT_FILE = os.path.join(
    OUTPUT_FOLDER,
    "final_baseline_results.csv"
)

CONFUSION_MATRIX_FILE = os.path.join(
    OUTPUT_FOLDER,
    "confusion_matrix.png"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset(file_path):

    data = pd.read_csv(file_path)

    X = data.drop(
        columns=["filename", "class", "fold"]
    )

    y = data["class"]

    return X, y


# ============================================================
# MAIN
# ============================================================

def main():

    print("========================================")
    print("      SOUND2SEE FINAL BASELINE")
    print("========================================")

    os.makedirs(
        OUTPUT_FOLDER,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    X_train, y_train = load_dataset(
        TRAIN_FILE
    )

    X_validation, y_validation = load_dataset(
        VALIDATION_FILE
    )

    X_test, y_test = load_dataset(
        TEST_FILE
    )

    # --------------------------------------------------------
    # Combine training + validation
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

    print()
    print("Training samples       :", len(X_train))
    print("Validation samples     :", len(X_validation))
    print("Final training samples :", len(X_train_final))
    print("Test samples           :", len(X_test))
    print("Features               :", X_train_final.shape[1])

    # --------------------------------------------------------
    # Create final SVM pipeline
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
    # Train final model
    # --------------------------------------------------------

    print("\nTraining final SVM...")

    model.fit(
        X_train_final,
        y_train_final
    )

    print("Training complete.")

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
    # RESULTS
    # ========================================================

    print("\n========================================")
    print("       FINAL TEST PERFORMANCE")
    print("========================================")

    print(
        f"Accuracy : {accuracy:.4f}"
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

    report = classification_report(
        y_test,
        predictions,
        zero_division=0
    )

    print(report)

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    matrix = confusion_matrix(
        y_test,
        predictions
    )

    print("========================================")
    print("          CONFUSION MATRIX")
    print("========================================")

    print(matrix)

    # --------------------------------------------------------
    # Save confusion matrix image
    # --------------------------------------------------------

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=model.classes_
    )

    display.plot(
        xticks_rotation=45
    )

    plt.title(
        "Sound2See - Final SVM Confusion Matrix"
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
            "model": "SVM",
            "training_samples": len(X_train_final),
            "test_samples": len(X_test),
            "features": X_train_final.shape[1],
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
    # Save trained model
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
    print("     FINAL BASELINE COMPLETE")
    print("========================================")


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
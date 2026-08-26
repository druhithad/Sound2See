import os
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

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

TRAIN_FILE = r"data\processed\splits\train.csv"
VALIDATION_FILE = r"data\processed\splits\validation.csv"
TEST_FILE = r"data\processed\splits\test.csv"

OUTPUT_FOLDER = r"outputs\baseline"


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset(file_path):

    data = pd.read_csv(file_path)

    # Remove identifiers and fold information
    X = data.drop(
        columns=["filename", "class", "fold"]
    )

    y = data["class"]

    return X, y


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(model_name, model, X_train, y_train,
                   X_validation, y_validation,
                   X_test, y_test):

    print()
    print("========================================")
    print(f"       {model_name.upper()}")
    print("========================================")

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model.fit(
        X_train,
        y_train
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
    # Test prediction
    # --------------------------------------------------------

    test_predictions = model.predict(
        X_test
    )

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
    # Print validation results
    # --------------------------------------------------------

    print("\nVALIDATION RESULTS")

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
    # Print test results
    # --------------------------------------------------------

    print("\nTEST RESULTS")

    print(
        f"Accuracy : {test_accuracy:.4f}"
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

    print("\nTEST CLASSIFICATION REPORT")

    print(
        classification_report(
            y_test,
            test_predictions,
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    matrix = confusion_matrix(
        y_test,
        test_predictions
    )

    print("TEST CONFUSION MATRIX")

    print(matrix)

    return {
        "model": model_name,
        "validation_accuracy": validation_accuracy,
        "validation_precision": validation_precision,
        "validation_recall": validation_recall,
        "validation_f1": validation_f1,
        "test_accuracy": test_accuracy,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_f1": test_f1
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("========================================")
    print("      SOUND2SEE BASELINE EXPERIMENT")
    print("========================================")

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    for file_path in [
        TRAIN_FILE,
        VALIDATION_FILE,
        TEST_FILE
    ]:

        if not os.path.exists(file_path):

            print(
                f"\nERROR: File not found:"
            )

            print(file_path)

            return

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

    print("\nDataset sizes:")

    print(
        f"Training   : {len(X_train)}"
    )

    print(
        f"Validation : {len(X_validation)}"
    )

    print(
        f"Testing    : {len(X_test)}"
    )

    print(
        f"Features   : {X_train.shape[1]}"
    )

    # ========================================================
    # DEFINE MODELS
    # ========================================================

    models = {

        "Logistic Regression": Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=42
                )
            )
        ]),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42
        ),

        "SVM": Pipeline([
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
    }

    # ========================================================
    # TRAIN AND EVALUATE
    # ========================================================

    results = []

    trained_models = {}

    for model_name, model in models.items():

        result = evaluate_model(
            model_name,
            model,
            X_train,
            y_train,
            X_validation,
            y_validation,
            X_test,
            y_test
        )

        results.append(result)

        trained_models[model_name] = model

    # ========================================================
    # RESULTS TABLE
    # ========================================================

    results_data = pd.DataFrame(
        results
    )

    os.makedirs(
        OUTPUT_FOLDER,
        exist_ok=True
    )

    results_file = os.path.join(
        OUTPUT_FOLDER,
        "baseline_results.csv"
    )

    results_data.to_csv(
        results_file,
        index=False
    )

    print("\n========================================")
    print("          MODEL COMPARISON")
    print("========================================")

    print(
        results_data[
            [
                "model",
                "validation_accuracy",
                "validation_f1",
                "test_accuracy",
                "test_f1"
            ]
        ].to_string(index=False)
    )

    print(
        f"\nResults saved to:"
    )

    print(results_file)

    print("\n========================================")
    print("       BASELINE EXPERIMENT COMPLETE")
    print("========================================")


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
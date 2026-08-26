import os
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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
# CONFIGURATION
# ============================================================

FEATURE_FILE = r"data\processed\combined\combined_features.csv"

OUTPUT_FOLDER = r"outputs\class_weighted_svm"

MODEL_FILE = os.path.join(
    OUTPUT_FOLDER,
    "sound2see_class_weighted_svm.joblib"
)

REPORT_FILE = os.path.join(
    OUTPUT_FOLDER,
    "classification_report.csv"
)

RESULT_FILE = os.path.join(
    OUTPUT_FOLDER,
    "class_weight_results.csv"
)

CONFUSION_FILE = os.path.join(
    OUTPUT_FOLDER,
    "confusion_matrix.csv"
)


# ============================================================
# IMPORTS FOR MODEL SAVING
# ============================================================

import joblib


# ============================================================
# MAIN
# ============================================================

def main():

    print("========================================")
    print(" SOUND2SEE CLASS-WEIGHTED SVM")
    print("========================================")

    os.makedirs(
        OUTPUT_FOLDER,
        exist_ok=True
    )

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    print("\nLoading combined feature dataset...")

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

    X = data[feature_columns]
    y = data["class"]

    # --------------------------------------------------------
    # SPLIT DATA
    # --------------------------------------------------------

    train_data = data[
        data["split"] == "train"
    ]

    validation_data = data[
        data["split"] == "validation"
    ]

    test_data = data[
        data["split"] == "test"
    ]

    train_validation = pd.concat(
        [
            train_data,
            validation_data
        ],
        ignore_index=True
    )

    X_train = train_data[
        feature_columns
    ]

    y_train = train_data[
        "class"
    ]

    X_validation = validation_data[
        feature_columns
    ]

    y_validation = validation_data[
        "class"
    ]

    X_train_validation = train_validation[
        feature_columns
    ]

    y_train_validation = train_validation[
        "class"
    ]

    X_test = test_data[
        feature_columns
    ]

    y_test = test_data[
        "class"
    ]

    print("\n========================================")
    print(" DATASET INFORMATION")
    print("========================================")

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
        y.nunique()
    )

    print(
        "Class names        :",
        sorted(y.unique())
    )

    # --------------------------------------------------------
    # CLASS DISTRIBUTION
    # --------------------------------------------------------

    print("\nClass distribution:")

    print(
        y_train.value_counts().sort_index()
    )

    # ========================================================
    # EXPERIMENT 1
    # ========================================================

    print("\n========================================")
    print(" EXPERIMENT 1: BALANCED CLASS WEIGHTS")
    print("========================================")

    balanced_model = Pipeline([
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
                class_weight="balanced",
                random_state=42
            )
        )
    ])

    print("\nTraining balanced SVM...")

    balanced_model.fit(
        X_train,
        y_train
    )

    validation_pred = balanced_model.predict(
        X_validation
    )

    validation_accuracy = accuracy_score(
        y_validation,
        validation_pred
    )

    validation_precision = precision_score(
        y_validation,
        validation_pred,
        average="weighted",
        zero_division=0
    )

    validation_recall = recall_score(
        y_validation,
        validation_pred,
        average="weighted",
        zero_division=0
    )

    validation_f1 = f1_score(
        y_validation,
        validation_pred,
        average="weighted",
        zero_division=0
    )

    validation_macro_f1 = f1_score(
        y_validation,
        validation_pred,
        average="macro",
        zero_division=0
    )

    print("\nVALIDATION PERFORMANCE")

    print(
        "Accuracy        :",
        f"{validation_accuracy:.4f}"
    )

    print(
        "Precision       :",
        f"{validation_precision:.4f}"
    )

    print(
        "Recall          :",
        f"{validation_recall:.4f}"
    )

    print(
        "Weighted F1     :",
        f"{validation_f1:.4f}"
    )

    print(
        "Macro F1        :",
        f"{validation_macro_f1:.4f}"
    )

    print("\nVALIDATION CLASSIFICATION REPORT")

    print(
        classification_report(
            y_validation,
            validation_pred,
            zero_division=0
        )
    )

    # ========================================================
    # FINAL TRAINING
    # ========================================================

    print("\n========================================")
    print(" FINAL BALANCED MODEL TRAINING")
    print("========================================")

    final_model = Pipeline([
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
                class_weight="balanced",
                random_state=42
            )
        )
    ])

    print(
        "Training samples:",
        len(X_train_validation)
    )

    final_model.fit(
        X_train_validation,
        y_train_validation
    )

    print("Final model training complete.")

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    test_pred = final_model.predict(
        X_test
    )

    test_accuracy = accuracy_score(
        y_test,
        test_pred
    )

    test_precision = precision_score(
        y_test,
        test_pred,
        average="weighted",
        zero_division=0
    )

    test_recall = recall_score(
        y_test,
        test_pred,
        average="weighted",
        zero_division=0
    )

    test_f1 = f1_score(
        y_test,
        test_pred,
        average="weighted",
        zero_division=0
    )

    test_macro_f1 = f1_score(
        y_test,
        test_pred,
        average="macro",
        zero_division=0
    )

    print("\n========================================")
    print(" FINAL TEST PERFORMANCE")
    print("========================================")

    print(
        "Accuracy        :",
        f"{test_accuracy:.4f}"
    )

    print(
        "Accuracy        :",
        f"{test_accuracy * 100:.2f}%"
    )

    print(
        "Precision       :",
        f"{test_precision:.4f}"
    )

    print(
        "Recall          :",
        f"{test_recall:.4f}"
    )

    print(
        "Weighted F1     :",
        f"{test_f1:.4f}"
    )

    print(
        "Macro F1        :",
        f"{test_macro_f1:.4f}"
    )

    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    report = classification_report(
        y_test,
        test_pred,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(
        report
    ).transpose()

    report_df.to_csv(
        REPORT_FILE
    )

    print("\n========================================")
    print(" TEST CLASSIFICATION REPORT")
    print("========================================")

    print(
        classification_report(
            y_test,
            test_pred,
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    classes = final_model[
        "classifier"
    ].classes_

    cm = confusion_matrix(
        y_test,
        test_pred,
        labels=classes
    )

    cm_df = pd.DataFrame(
        cm,
        index=classes,
        columns=classes
    )

    cm_df.to_csv(
        CONFUSION_FILE
    )

    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    joblib.dump(
        final_model,
        MODEL_FILE
    )

    # --------------------------------------------------------
    # SAVE SUMMARY
    # --------------------------------------------------------

    results = pd.DataFrame([
        {
            "model": "Class-Weighted RBF SVM",
            "C": 10,
            "gamma": "scale",
            "class_weight": "balanced",
            "validation_accuracy": validation_accuracy,
            "validation_weighted_f1": validation_f1,
            "validation_macro_f1": validation_macro_f1,
            "test_accuracy": test_accuracy,
            "test_weighted_f1": test_f1,
            "test_macro_f1": test_macro_f1
        }
    ])

    results.to_csv(
        RESULT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # FILES
    # --------------------------------------------------------

    print("\n========================================")
    print(" FILES SAVED")
    print("========================================")

    print(
        "Results        :",
        RESULT_FILE
    )

    print(
        "Classification :",
        REPORT_FILE
    )

    print(
        "Confusion      :",
        CONFUSION_FILE
    )

    print(
        "Model          :",
        MODEL_FILE
    )

    print("\n========================================")
    print(" CLASS-WEIGHTED SVM COMPLETE")
    print("========================================")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
import os
import joblib
import pandas as pd

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
# SETTINGS
# ============================================================

FEATURE_FILE = r"data\processed\combined\combined_features.csv"

OUTPUT_FOLDER = r"outputs\combined_class_weighted"

MODEL_FILE = os.path.join(
    OUTPUT_FOLDER,
    "sound2see_combined_class_weighted_svm.joblib"
)

RESULT_FILE = os.path.join(
    OUTPUT_FOLDER,
    "class_weighted_results.csv"
)

CLASSIFICATION_FILE = os.path.join(
    OUTPUT_FOLDER,
    "classification_report.csv"
)

CONFUSION_MATRIX_FILE = os.path.join(
    OUTPUT_FOLDER,
    "confusion_matrix.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset():

    print("\nLoading combined feature dataset...")

    data = pd.read_csv(
        FEATURE_FILE
    )

    identifier_columns = [
        "dataset",
        "filename",
        "file_path",
        "class",
        "split"
    ]

    feature_columns = [
        column
        for column in data.columns
        if column not in identifier_columns
    ]

    X = data[feature_columns]
    y = data["class"]
    split = data["split"]

    print(
        "Total samples:",
        len(data)
    )

    print(
        "Features:",
        len(feature_columns)
    )

    print(
        "Classes:",
        y.nunique()
    )

    return X, y, split, feature_columns


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
    # Load dataset
    # --------------------------------------------------------

    (
        X,
        y,
        split,
        feature_columns
    ) = load_dataset()

    # --------------------------------------------------------
    # Existing official split
    # --------------------------------------------------------

    train_mask = split == "train"
    validation_mask = split == "validation"
    test_mask = split == "test"

    X_train = X[train_mask]
    y_train = y[train_mask]

    X_validation = X[validation_mask]
    y_validation = y[validation_mask]

    X_test = X[test_mask]
    y_test = y[test_mask]

    print("\n========================================")
    print(" DATASET SPLIT")
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

    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    print("\n========================================")
    print(" TRAINING CLASS DISTRIBUTION")
    print("========================================")

    print(
        y_train.value_counts().sort_index()
    )

    # ========================================================
    # CLASS-WEIGHTED SVM
    # ========================================================

    print("\n========================================")
    print(" CLASS-WEIGHTED SVM")
    print("========================================")

    print(
        "Kernel       : RBF"
    )

    print(
        "C            : 10"
    )

    print(
        "Gamma        : scale"
    )

    print(
        "Class weight : balanced"
    )

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
                class_weight="balanced",
                probability=True,
                random_state=42
            )
        )
    ])

    # ========================================================
    # VALIDATION TRAINING
    # ========================================================

    print("\n========================================")
    print(" VALIDATION TRAINING")
    print("========================================")

    print(
        "Training on:",
        len(X_train),
        "samples"
    )

    model.fit(
        X_train,
        y_train
    )

    print(
        "Training complete."
    )

    # ========================================================
    # VALIDATION PREDICTION
    # ========================================================

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

    validation_macro_f1 = f1_score(
        y_validation,
        validation_predictions,
        average="macro",
        zero_division=0
    )

    # ========================================================
    # VALIDATION RESULTS
    # ========================================================

    print("\n========================================")
    print(" VALIDATION PERFORMANCE")
    print("========================================")

    print(
        f"Accuracy        : {validation_accuracy:.4f}"
    )

    print(
        f"Accuracy        : {validation_accuracy * 100:.2f}%"
    )

    print(
        f"Precision       : {validation_precision:.4f}"
    )

    print(
        f"Recall          : {validation_recall:.4f}"
    )

    print(
        f"Weighted F1     : {validation_f1:.4f}"
    )

    print(
        f"Macro F1        : {validation_macro_f1:.4f}"
    )

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

    # ========================================================
    # FINAL TRAINING
    #
    # Validation data is now added.
    # Test data remains untouched.
    # ========================================================

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
    print(" FINAL CLASS-WEIGHTED TRAINING")
    print("========================================")

    print(
        "Training samples:",
        len(X_train_final)
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
                C=10,
                gamma="scale",
                class_weight="balanced",
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

    # ========================================================
    # FINAL TEST
    # ========================================================

    print("\n========================================")
    print(" FINAL TEST EVALUATION")
    print("========================================")

    test_predictions = final_model.predict(
        X_test
    )

    # --------------------------------------------------------
    # Metrics
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

    test_macro_f1 = f1_score(
        y_test,
        test_predictions,
        average="macro",
        zero_division=0
    )

    # ========================================================
    # TEST RESULTS
    # ========================================================

    print("\n========================================")
    print(" FINAL TEST PERFORMANCE")
    print("========================================")

    print(
        f"Accuracy        : {test_accuracy:.4f}"
    )

    print(
        f"Accuracy        : {test_accuracy * 100:.2f}%"
    )

    print(
        f"Precision       : {test_precision:.4f}"
    )

    print(
        f"Recall          : {test_recall:.4f}"
    )

    print(
        f"Weighted F1     : {test_f1:.4f}"
    )

    print(
        f"Macro F1        : {test_macro_f1:.4f}"
    )

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    report_dict = classification_report(
        y_test,
        test_predictions,
        zero_division=0,
        output_dict=True
    )

    report_dataframe = pd.DataFrame(
        report_dict
    ).transpose()

    report_dataframe.to_csv(
        CLASSIFICATION_FILE
    )

    print("\n========================================")
    print(" TEST CLASSIFICATION REPORT")
    print("========================================")

    print(
        classification_report(
            y_test,
            test_predictions,
            zero_division=0
        )
    )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    matrix = confusion_matrix(
        y_test,
        test_predictions,
        labels=final_model.classes_
    )

    matrix_dataframe = pd.DataFrame(
        matrix,
        index=final_model.classes_,
        columns=final_model.classes_
    )

    matrix_dataframe.to_csv(
        CONFUSION_MATRIX_FILE
    )

    print(
        "\nConfusion matrix saved:",
        CONFUSION_MATRIX_FILE
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results = pd.DataFrame([
        {
            "model": "Combined Class-Weighted RBF SVM",
            "datasets": "Animal Sounds + ESC-50",
            "classes": 15,
            "features": len(feature_columns),
            "training_samples": len(X_train_final),
            "test_samples": len(X_test),

            "validation_accuracy":
                validation_accuracy,

            "validation_precision":
                validation_precision,

            "validation_recall":
                validation_recall,

            "validation_weighted_f1":
                validation_f1,

            "validation_macro_f1":
                validation_macro_f1,

            "test_accuracy":
                test_accuracy,

            "test_precision":
                test_precision,

            "test_recall":
                test_recall,

            "test_weighted_f1":
                test_f1,

            "test_macro_f1":
                test_macro_f1,

            "C": 10,

            "gamma": "scale",

            "class_weight":
                "balanced"
        }
    ])

    results.to_csv(
        RESULT_FILE,
        index=False
    )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    joblib.dump(
        final_model,
        MODEL_FILE
    )

    # ========================================================
    # FINAL STATUS
    # ========================================================

    print("\n========================================")
    print(" FILES SAVED")
    print("========================================")

    print(
        "Results            :",
        RESULT_FILE
    )

    print(
        "Classification     :",
        CLASSIFICATION_FILE
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
    print(" CLASS-WEIGHTED SVM COMPLETE")
    print("========================================")


if __name__ == "__main__":
    main()
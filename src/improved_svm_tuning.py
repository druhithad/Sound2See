import os
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
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

TEMPORAL_FEATURE_FILE = r"data\processed\temporal_features.csv"

OUTPUT_FOLDER = r"outputs\improved_svm"

MODEL_FILE = os.path.join(
    OUTPUT_FOLDER,
    "sound2see_improved_svm.joblib"
)

RESULT_FILE = os.path.join(
    OUTPUT_FOLDER,
    "svm_tuning_results.csv"
)

CONFUSION_MATRIX_FILE = os.path.join(
    OUTPUT_FOLDER,
    "improved_confusion_matrix.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset():

    print("\nLoading temporal feature dataset...")

    data = pd.read_csv(
        TEMPORAL_FEATURE_FILE
    )

    print(
        "Rows    :",
        len(data)
    )

    print(
        "Columns :",
        len(data.columns)
    )

    # --------------------------------------------------------
    # Identify columns
    # --------------------------------------------------------

    identifier_columns = [
        "filename",
        "class",
        "fold"
    ]

    feature_columns = [
        column
        for column in data.columns
        if column not in identifier_columns
    ]

    X = data[feature_columns]

    y = data["class"]

    folds = data["fold"]

    print(
        "Features:",
        len(feature_columns)
    )

    print("\nFold distribution:")

    print(
        folds.value_counts().sort_index()
    )

    return X, y, folds


# ============================================================
# MAIN
# ============================================================

def main():

    print("========================================")
    print(" SOUND2SEE IMPROVED SVM TUNING")
    print("========================================")

    os.makedirs(
        OUTPUT_FOLDER,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    X, y, folds = load_dataset()

    # --------------------------------------------------------
    # Keep fold 5 as final untouched test set
    # --------------------------------------------------------

    train_validation_mask = folds != 5
    test_mask = folds == 5

    X_train_validation = X[
        train_validation_mask
    ]

    y_train_validation = y[
        train_validation_mask
    ]

    folds_train_validation = folds[
        train_validation_mask
    ]

    X_test = X[
        test_mask
    ]

    y_test = y[
        test_mask
    ]

    print("\n========================================")
    print(" DATASET SPLIT")
    print("========================================")

    print(
        "Training + validation:",
        len(X_train_validation)
    )

    print(
        "Final test           :",
        len(X_test)
    )

    print(
        "Test fold            : 5"
    )

    # --------------------------------------------------------
    # SVM pipeline
    # --------------------------------------------------------

    pipeline = Pipeline([
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
    # Hyperparameter grid
    # --------------------------------------------------------

    parameter_grid = {
        "classifier__C": [
            0.1,
            1,
            10,
            100
        ],

        "classifier__gamma": [
            "scale",
            0.001,
            0.01,
            0.1
        ]
    }

    print("\n========================================")
    print(" HYPERPARAMETER SEARCH")
    print("========================================")

    print(
        "C values     :",
        parameter_grid["classifier__C"]
    )

    print(
        "Gamma values :",
        parameter_grid["classifier__gamma"]
    )

    print(
        "Configurations:",
        16
    )

    # --------------------------------------------------------
    # Group-aware fold validation
    # --------------------------------------------------------
    #
    # Each original ESC-50 recording belongs to a fold.
    # We use the ESC-50 fold number as the group.
    #
    # Fold 5 remains completely untouched.
    # --------------------------------------------------------

    cv = StratifiedGroupKFold(
        n_splits=4,
        shuffle=True,
        random_state=42
    )

    # --------------------------------------------------------
    # Grid search
    # --------------------------------------------------------

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        scoring="f1_weighted",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        return_train_score=False
    )

    print("\nStarting hyperparameter search...")

    search.fit(
        X_train_validation,
        y_train_validation,
        groups=folds_train_validation
    )

    print("\nHyperparameter search complete.")

    # --------------------------------------------------------
    # Best configuration
    # --------------------------------------------------------

    print("\n========================================")
    print(" BEST CONFIGURATION")
    print("========================================")

    print(
        "Best C     :",
        search.best_params_[
            "classifier__C"
        ]
    )

    print(
        "Best gamma :",
        search.best_params_[
            "classifier__gamma"
        ]
    )

    print(
        "Best CV F1 :",
        f"{search.best_score_:.4f}"
    )

    # --------------------------------------------------------
    # Show all configurations
    # --------------------------------------------------------

    cv_results = pd.DataFrame(
        search.cv_results_
    )

    result_table = cv_results[
        [
            "param_classifier__C",
            "param_classifier__gamma",
            "mean_test_score",
            "std_test_score",
            "rank_test_score"
        ]
    ].sort_values(
        "rank_test_score"
    )

    print("\n========================================")
    print(" ALL SVM CONFIGURATIONS")
    print("========================================")

    print(
        result_table.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Save tuning results
    # --------------------------------------------------------

    result_table.to_csv(
        RESULT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Final model training
    # --------------------------------------------------------
    #
    # IMPORTANT:
    # The final model is now trained using folds 1-4.
    # Fold 5 is still untouched.
    # --------------------------------------------------------

    best_model = search.best_estimator_

    print("\n========================================")
    print(" FINAL IMPROVED MODEL")
    print("========================================")

    print(
        "Training using folds 1-4..."
    )

    best_model.fit(
        X_train_validation,
        y_train_validation
    )

    print(
        "Training complete."
    )

    # --------------------------------------------------------
    # Final test prediction
    # --------------------------------------------------------

    predictions = best_model.predict(
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
    print(" FINAL TEST PERFORMANCE")
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
    print(" CLASSIFICATION REPORT")
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
        predictions
    )

    print("========================================")
    print(" CONFUSION MATRIX")
    print("========================================")

    print(matrix)

    matrix_dataframe = pd.DataFrame(
        matrix,
        index=best_model.classes_,
        columns=best_model.classes_
    )

    matrix_dataframe.to_csv(
        CONFUSION_MATRIX_FILE
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    joblib.dump(
        best_model,
        MODEL_FILE
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n========================================")
    print(" FILES SAVED")
    print("========================================")

    print(
        "Tuning results     :",
        RESULT_FILE
    )

    print(
        "Confusion matrix   :",
        CONFUSION_MATRIX_FILE
    )

    print(
        "Improved model     :",
        MODEL_FILE
    )

    print("\n========================================")
    print(" IMPROVED SVM TUNING COMPLETE")
    print("========================================")


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
import os
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
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

OUTPUT_FOLDER = r"outputs\combined_tuned"

MODEL_FILE = os.path.join(
    OUTPUT_FOLDER,
    "sound2see_combined_tuned_svm.joblib"
)

RESULT_FILE = os.path.join(
    OUTPUT_FOLDER,
    "svm_tuning_results.csv"
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

    print(
        "Total samples:",
        len(data)
    )

    print(
        "Total columns:",
        len(data.columns)
    )

    # --------------------------------------------------------
    # Feature columns
    # --------------------------------------------------------

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

    X = data[
        feature_columns
    ]

    y = data["class"]

    split = data["split"]

    print(
        "Features:",
        len(feature_columns)
    )

    print(
        "Classes:",
        y.nunique()
    )

    print(
        "\nClass distribution:"
    )

    print(
        y.value_counts().sort_index()
    )

    return X, y, split


# ============================================================
# MAIN
# ============================================================

def main():

    print("========================================")
    print(" SOUND2SEE COMBINED SVM TUNING")
    print("========================================")

    os.makedirs(
        OUTPUT_FOLDER,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    X, y, split = load_dataset()

    # --------------------------------------------------------
    # Existing split
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

    # --------------------------------------------------------
    # Combine training + validation
    #
    # These are used for hyperparameter selection through CV.
    # Test remains completely untouched.
    # --------------------------------------------------------

    X_train_validation = pd.concat(
        [
            X_train,
            X_validation
        ],
        ignore_index=True
    )

    y_train_validation = pd.concat(
        [
            y_train,
            y_validation
        ],
        ignore_index=True
    )

    print(
        "\nTraining + validation:",
        len(X_train_validation)
    )

    # ========================================================
    # SVM PIPELINE
    # ========================================================

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

    # ========================================================
    # HYPERPARAMETER GRID
    # ========================================================

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
        "C values:",
        parameter_grid["classifier__C"]
    )

    print(
        "Gamma values:",
        parameter_grid["classifier__gamma"]
    )

    print(
        "Configurations: 16"
    )

    # --------------------------------------------------------
    # Stratified cross-validation
    # --------------------------------------------------------

    cv = StratifiedKFold(
        n_splits=5,
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

    print(
        "\nStarting hyperparameter search..."
    )

    search.fit(
        X_train_validation,
        y_train_validation
    )

    print(
        "\nHyperparameter search complete."
    )

    # ========================================================
    # BEST CONFIGURATION
    # ========================================================

    print("\n========================================")
    print(" BEST CONFIGURATION")
    print("========================================")

    print(
        "Best C:",
        search.best_params_[
            "classifier__C"
        ]
    )

    print(
        "Best gamma:",
        search.best_params_[
            "classifier__gamma"
        ]
    )

    print(
        f"Best CV weighted F1: "
        f"{search.best_score_:.4f}"
    )

    # ========================================================
    # SAVE ALL TUNING RESULTS
    # ========================================================

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

    result_table.to_csv(
        RESULT_FILE,
        index=False
    )

    print(
        "\nTuning results saved:",
        RESULT_FILE
    )

    print("\n========================================")
    print(" ALL CONFIGURATIONS")
    print("========================================")

    print(
        result_table.to_string(
            index=False
        )
    )

    # ========================================================
    # FINAL MODEL
    # ========================================================

    best_model = search.best_estimator_

    print("\n========================================")
    print(" FINAL MODEL TRAINING")
    print("========================================")

    print(
        "Training on:",
        len(X_train_validation),
        "samples"
    )

    best_model.fit(
        X_train_validation,
        y_train_validation
    )

    print(
        "Final model training complete."
    )

    # ========================================================
    # FINAL TEST
    # ========================================================

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

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    report = classification_report(
        y_test,
        predictions,
        zero_division=0,
        output_dict=True
    )

    report_dataframe = pd.DataFrame(
        report
    ).transpose()

    report_dataframe.to_csv(
        CLASSIFICATION_FILE
    )

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

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=best_model.classes_
    )

    matrix_dataframe = pd.DataFrame(
        matrix,
        index=best_model.classes_,
        columns=best_model.classes_
    )

    matrix_dataframe.to_csv(
        CONFUSION_MATRIX_FILE
    )

    print(
        "Confusion matrix saved:",
        CONFUSION_MATRIX_FILE
    )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    joblib.dump(
        best_model,
        MODEL_FILE
    )

    print(
        "\nModel saved:",
        MODEL_FILE
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n========================================")
    print(" COMBINED SVM TUNING COMPLETE")
    print("========================================")

    print(
        "Best C       :",
        search.best_params_["classifier__C"]
    )

    print(
        "Best gamma   :",
        search.best_params_["classifier__gamma"]
    )

    print(
        f"CV F1        : {search.best_score_:.4f}"
    )

    print(
        f"Test accuracy: {accuracy:.4f}"
    )

    print(
        f"Test F1      : {f1:.4f}"
    )

    print("========================================")


if __name__ == "__main__":
    main()
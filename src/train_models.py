import os
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

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

INPUT_FILE = (
    r"data\processed\combined\enhanced_features.csv"
)

MODEL_DIR = (
    r"models"
)

RESULTS_DIR = (
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

    print("\n========================================")
    print(" LOADING FEATURE DATA")
    print("========================================")

    df = pd.read_csv(INPUT_FILE)

    feature_columns = [
        column
        for column in df.columns
        if column not in IDENTIFIER_COLUMNS
    ]

    X = df[feature_columns]
    y = df["class"]

    train_mask = df["split"] == "train"
    validation_mask = df["split"] == "validation"
    test_mask = df["split"] == "test"

    X_train = X[train_mask]
    y_train = y[train_mask]

    X_validation = X[validation_mask]
    y_validation = y[validation_mask]

    X_test = X[test_mask]
    y_test = y[test_mask]

    print("Total samples      :", len(df))
    print("Features            :", len(feature_columns))
    print("Training samples    :", len(X_train))
    print("Validation samples  :", len(X_validation))
    print("Test samples        :", len(X_test))
    print("Classes             :", y.nunique())

    return (
        df,
        feature_columns,
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test
    )


# ============================================================
# DEFINE MODELS
# ============================================================

def create_models():

    models = {

        "SVM": Pipeline([
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
                    class_weight="balanced"
                )
            )
        ]),

        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        ),

        "LogisticRegression": Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=42
                )
            )
        ])
    }

    return models


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    model,
    X,
    y,
    dataset_name
):

    predictions = model.predict(X)

    accuracy = accuracy_score(
        y,
        predictions
    )

    precision = precision_score(
        y,
        predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y,
        predictions,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y,
        predictions,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y,
        predictions,
        average="weighted",
        zero_division=0
    )

    print(
        f"\n{dataset_name} RESULTS"
    )

    print(
        "Accuracy          :",
        f"{accuracy:.4f}"
    )

    print(
        "Macro Precision   :",
        f"{precision:.4f}"
    )

    print(
        "Macro Recall      :",
        f"{recall:.4f}"
    )

    print(
        "Macro F1          :",
        f"{f1:.4f}"
    )

    print(
        "Weighted F1       :",
        f"{weighted_f1:.4f}"
    )

    return {
        "dataset": dataset_name,
        "accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "weighted_f1": weighted_f1
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("========================================")
    print(" SOUND2SEE BASELINE MODEL TRAINING")
    print("========================================")

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    (
        df,
        feature_columns,
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test
    ) = load_data()

    models = create_models()

    all_results = []

    # ========================================================
    # TRAIN MODELS
    # ========================================================

    for model_name, model in models.items():

        print("\n========================================")
        print(
            f" TRAINING: {model_name}"
        )
        print("========================================")

        model.fit(
            X_train,
            y_train
        )

        print(
            f"{model_name} training complete."
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        validation_result = evaluate_model(
            model,
            X_validation,
            y_validation,
            "Validation"
        )

        validation_result[
            "model"
        ] = model_name

        all_results.append(
            validation_result
        )

        # ----------------------------------------------------
        # Save model
        # ----------------------------------------------------

        model_file = os.path.join(
            MODEL_DIR,
            f"{model_name.lower()}_baseline.joblib"
        )

        joblib.dump(
            model,
            model_file
        )

        print(
            "Model saved:",
            model_file
        )

    # ========================================================
    # VALIDATION COMPARISON
    # ========================================================

    results_df = pd.DataFrame(
        all_results
    )

    validation_results = (
        results_df[
            results_df["dataset"] == "Validation"
        ]
        .sort_values(
            "macro_f1",
            ascending=False
        )
    )

    print("\n========================================")
    print(" VALIDATION MODEL COMPARISON")
    print("========================================")

    print(
        validation_results[
            [
                "model",
                "accuracy",
                "macro_precision",
                "macro_recall",
                "macro_f1",
                "weighted_f1"
            ]
        ].to_string(
            index=False
        )
    )

    # ========================================================
    # SELECT BEST MODEL
    # ========================================================

    best_model_name = (
        validation_results
        .iloc[0]["model"]
    )

    best_model = models[
        best_model_name
    ]

    print("\n========================================")
    print(" BEST BASELINE MODEL")
    print("========================================")

    print(
        "Selected model:",
        best_model_name
    )

    print(
        "Selection metric: Validation Macro F1"
    )

    # ========================================================
    # TEST BEST MODEL
    # ========================================================

    print("\n========================================")
    print(" FINAL TEST EVALUATION")
    print("========================================")

    test_result = evaluate_model(
        best_model,
        X_test,
        y_test,
        "Test"
    )

    test_result[
        "model"
    ] = best_model_name

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    test_predictions = (
        best_model.predict(X_test)
    )

    print("\n========================================")
    print(" CLASSIFICATION REPORT")
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

    classes = sorted(
        y_test.unique()
    )

    matrix = confusion_matrix(
        y_test,
        test_predictions,
        labels=classes
    )

    confusion_df = pd.DataFrame(
        matrix,
        index=classes,
        columns=classes
    )

    print("\n========================================")
    print(" CONFUSION MATRIX")
    print("========================================")

    print(
        confusion_df
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_df = pd.concat(
        [
            results_df,
            pd.DataFrame(
                [test_result]
            )
        ],
        ignore_index=True
    )

    results_file = os.path.join(
        RESULTS_DIR,
        "baseline_model_results.csv"
    )

    results_df.to_csv(
        results_file,
        index=False
    )

    confusion_file = os.path.join(
        RESULTS_DIR,
        "baseline_confusion_matrix.csv"
    )

    confusion_df.to_csv(
        confusion_file
    )

    feature_file = os.path.join(
        RESULTS_DIR,
        "training_features.csv"
    )

    pd.DataFrame(
        {
            "feature": feature_columns
        }
    ).to_csv(
        feature_file,
        index=False
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n========================================")
    print(" BASELINE TRAINING COMPLETE")
    print("========================================")

    print(
        "Best model:",
        best_model_name
    )

    print(
        "Test accuracy:",
        f"{test_result['accuracy']:.4f}"
    )

    print(
        "Test Macro F1:",
        f"{test_result['macro_f1']:.4f}"
    )

    print(
        "Test Weighted F1:",
        f"{test_result['weighted_f1']:.4f}"
    )

    print(
        "\nResults saved to:",
        RESULTS_DIR
    )

    print(
        "Models saved to:",
        MODEL_DIR
    )

    print("\n========================================")


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
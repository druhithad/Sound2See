import os
import warnings

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

warnings.filterwarnings("ignore")


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = (
    r"data\processed\combined\enhanced_features.csv"
)

OUTPUT_DIR = r"data\processed\model_results"
MODEL_DIR = r"models"

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)



# ============================================================
# LOAD DATA
# ============================================================

print("========================================")
print(" SOUND2SEE IMPROVED MODEL TRAINING")
print("========================================")

print("\n========================================")
print(" LOADING FEATURE DATA")
print("========================================")

df = pd.read_csv(INPUT_FILE)

identifier_columns = [
    "dataset",
    "filename",
    "file_path",
    "class",
    "split"
]

feature_columns = [
    column
    for column in df.columns
    if column not in identifier_columns
]

X = df[feature_columns]
y = df["class"]
split = df["split"]


X_train = X[split == "train"]
y_train = y[split == "train"]

X_validation = X[split == "validation"]
y_validation = y[split == "validation"]

X_test = X[split == "test"]
y_test = y[split == "test"]


print(
    f"Total samples      : {len(df)}"
)

print(
    f"Features            : {len(feature_columns)}"
)

print(
    f"Training samples    : {len(X_train)}"
)

print(
    f"Validation samples  : {len(X_validation)}"
)

print(
    f"Test samples        : {len(X_test)}"
)

print(
    f"Classes             : {y.nunique()}"
)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    model,
    model_name
):

    print("\n========================================")
    print(f" TRAINING: {model_name}")
    print("========================================")

    model.fit(
        X_train,
        y_train
    )

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
        average="macro",
        zero_division=0
    )

    validation_recall = recall_score(
        y_validation,
        validation_predictions,
        average="macro",
        zero_division=0
    )

    validation_f1 = f1_score(
        y_validation,
        validation_predictions,
        average="macro",
        zero_division=0
    )

    validation_weighted_f1 = f1_score(
        y_validation,
        validation_predictions,
        average="weighted",
        zero_division=0
    )

    print("\nValidation RESULTS")

    print(
        f"Accuracy          : {validation_accuracy:.4f}"
    )

    print(
        f"Macro Precision   : {validation_precision:.4f}"
    )

    print(
        f"Macro Recall      : {validation_recall:.4f}"
    )

    print(
        f"Macro F1          : {validation_f1:.4f}"
    )

    print(
        f"Weighted F1       : {validation_weighted_f1:.4f}"
    )

    # --------------------------------------------------------
    # Test evaluation
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
        average="macro",
        zero_division=0
    )

    test_recall = recall_score(
        y_test,
        test_predictions,
        average="macro",
        zero_division=0
    )

    test_f1 = f1_score(
        y_test,
        test_predictions,
        average="macro",
        zero_division=0
    )

    test_weighted_f1 = f1_score(
        y_test,
        test_predictions,
        average="weighted",
        zero_division=0
    )

    print("\nTest RESULTS")

    print(
        f"Accuracy          : {test_accuracy:.4f}"
    )

    print(
        f"Macro Precision   : {test_precision:.4f}"
    )

    print(
        f"Macro Recall      : {test_recall:.4f}"
    )

    print(
        f"Macro F1          : {test_f1:.4f}"
    )

    print(
        f"Weighted F1       : {test_weighted_f1:.4f}"
    )

    return {
        "model": model_name,

        "validation_accuracy":
            validation_accuracy,

        "validation_macro_precision":
            validation_precision,

        "validation_macro_recall":
            validation_recall,

        "validation_macro_f1":
            validation_f1,

        "validation_weighted_f1":
            validation_weighted_f1,

        "test_accuracy":
            test_accuracy,

        "test_macro_precision":
            test_precision,

        "test_macro_recall":
            test_recall,

        "test_macro_f1":
            test_f1,

        "test_weighted_f1":
            test_weighted_f1
    }


# ============================================================
# MODELS
# ============================================================

models = {

    # --------------------------------------------------------
    # 1. Balanced Logistic Regression
    # --------------------------------------------------------

    "LogisticRegression_Balanced":

        LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=42
        ),

    # --------------------------------------------------------
    # 2. Scaled Logistic Regression
    # --------------------------------------------------------

    "Scaled_LogisticRegression":

        Pipeline([
            (
                "scaler",
                StandardScaler()
            ),

            (
                "classifier",
                LogisticRegression(
                    max_iter=3000,
                    random_state=42
                )
            )
        ]),

    # --------------------------------------------------------
    # 3. Scaled Balanced Logistic Regression
    # --------------------------------------------------------

    "Scaled_LogisticRegression_Balanced":

        Pipeline([
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
        ]),

    # --------------------------------------------------------
    # 4. Balanced SVM
    # --------------------------------------------------------

    "SVM_Balanced":

        SVC(
            kernel="rbf",
            class_weight="balanced"
        ),

    # --------------------------------------------------------
    # 5. Scaled SVM
    # --------------------------------------------------------

    "Scaled_SVM":

        Pipeline([
            (
                "scaler",
                StandardScaler()
            ),

            (
                "classifier",
                SVC(
                    kernel="rbf"
                )
            )
        ]),

    # --------------------------------------------------------
    # 6. Scaled Balanced SVM
    # --------------------------------------------------------

    "Scaled_SVM_Balanced":

        Pipeline([
            (
                "scaler",
                StandardScaler()
            ),

            (
                "classifier",
                SVC(
                    kernel="rbf",
                    class_weight="balanced"
                )
            )
        ]),

    # --------------------------------------------------------
    # 7. PCA + Logistic Regression
    # --------------------------------------------------------

    "PCA_LogisticRegression":

        Pipeline([
            (
                "scaler",
                StandardScaler()
            ),

            (
                "pca",
                PCA(
                    n_components=0.95,
                    random_state=42
                )
            ),

            (
                "classifier",
                LogisticRegression(
                    max_iter=3000,
                    random_state=42
                )
            )
        ]),

    # --------------------------------------------------------
    # 8. PCA + Balanced Logistic Regression
    # --------------------------------------------------------

    "PCA_LogisticRegression_Balanced":

        Pipeline([
            (
                "scaler",
                StandardScaler()
            ),

            (
                "pca",
                PCA(
                    n_components=0.95,
                    random_state=42
                )
            ),

            (
                "classifier",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=42
                )
            )
        ]),

    # --------------------------------------------------------
    # 9. PCA + SVM
    # --------------------------------------------------------

    "PCA_SVM":

        Pipeline([
            (
                "scaler",
                StandardScaler()
            ),

            (
                "pca",
                PCA(
                    n_components=0.95,
                    random_state=42
                )
            ),

            (
                "classifier",
                SVC(
                    kernel="rbf"
                )
            )
        ]),

    # --------------------------------------------------------
    # 10. PCA + Balanced SVM
    # --------------------------------------------------------

    "PCA_SVM_Balanced":

        Pipeline([
            (
                "scaler",
                StandardScaler()
            ),

            (
                "pca",
                PCA(
                    n_components=0.95,
                    random_state=42
                )
            ),

            (
                "classifier",
                SVC(
                    kernel="rbf",
                    class_weight="balanced"
                )
            )
        ])
}


# ============================================================
# TRAIN ALL MODELS
# ============================================================

results = []

trained_models = {}

for model_name, model in models.items():

    result = evaluate_model(
        model,
        model_name
    )

    results.append(
        result
    )

    trained_models[
        model_name
    ] = model


# ============================================================
# COMPARISON
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    by="validation_macro_f1",
    ascending=False
)

print("\n========================================")
print(" MODEL COMPARISON")
print("========================================")

print(
    results_df[
        [
            "model",
            "validation_accuracy",
            "validation_macro_f1",
            "test_accuracy",
            "test_macro_f1",
            "test_weighted_f1"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# BEST MODEL
# ============================================================

best_model_name = results_df.iloc[0]["model"]

best_validation_f1 = results_df.iloc[0][
    "validation_macro_f1"
]

best_test_f1 = results_df.iloc[0][
    "test_macro_f1"
]

best_test_accuracy = results_df.iloc[0][
    "test_accuracy"
]

# ============================================================
# SAVE BEST MODEL
# ============================================================

best_model = trained_models[
    best_model_name
]

final_model_file = os.path.join(
    MODEL_DIR,
    "sound2see_final.joblib"
)

import joblib

joblib.dump(
    best_model,
    final_model_file
)

print(
    f"\nFinal model saved to: {final_model_file}"
)


print("\n========================================")
print(" BEST IMPROVED MODEL")
print("========================================")

print(
    f"Selected model       : {best_model_name}"
)

print(
    f"Validation Macro F1  : {best_validation_f1:.4f}"
)

print(
    f"Test Accuracy        : {best_test_accuracy:.4f}"
)

print(
    f"Test Macro F1        : {best_test_f1:.4f}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_file = os.path.join(
    OUTPUT_DIR,
    "improved_model_comparison.csv"
)

results_df.to_csv(
    results_file,
    index=False
)

print(
    f"\nResults saved to: {results_file}"
)

print("\n========================================")
print(" IMPROVED MODEL TRAINING COMPLETE")
print("========================================")
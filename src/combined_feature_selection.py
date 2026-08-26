import os
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


# ============================================================
# CONFIGURATION
# ============================================================

FEATURE_FILE = r"data\processed\combined\combined_features.csv"

OUTPUT_FOLDER = r"outputs\feature_selection"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Current best SVM configuration
C_VALUE = 10
GAMMA_VALUE = "scale"

# Feature counts to test
FEATURE_COUNTS = [20, 40, 60, 80, 100, 120, 140]


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 40)
print(" SOUND2SEE FEATURE SELECTION EXPERIMENT")
print("=" * 40)

print("\nLoading combined feature dataset...")

data = pd.read_csv(FEATURE_FILE)

print("Total samples:", len(data))
print("Total columns:", len(data.columns))


# ============================================================
# IDENTIFY FEATURE COLUMNS
# ============================================================

NON_FEATURE_COLUMNS = [
    "dataset",
    "filename",
    "file_path",
    "class",
    "split"
]

feature_columns = [
    column
    for column in data.columns
    if column not in NON_FEATURE_COLUMNS
]

print("Features:", len(feature_columns))
print("Classes:", data["class"].nunique())


# ============================================================
# SPLIT DATA
# ============================================================

train_data = data[data["split"] == "train"].copy()
validation_data = data[data["split"] == "validation"].copy()
test_data = data[data["split"] == "test"].copy()

print("\n" + "=" * 40)
print(" DATASET SPLIT")
print("=" * 40)

print("Training samples   :", len(train_data))
print("Validation samples :", len(validation_data))
print("Test samples       :", len(test_data))


# ============================================================
# CREATE MATRICES
# ============================================================

X_train = train_data[feature_columns].values
y_train = train_data["class"].values

X_validation = validation_data[feature_columns].values
y_validation = validation_data["class"].values

X_test = test_data[feature_columns].values
y_test = test_data["class"].values


# ============================================================
# FEATURE SELECTION EXPERIMENT
# ============================================================

results = []

print("\n" + "=" * 40)
print(" FEATURE SELECTION")
print("=" * 40)

print("\nTesting feature counts:")
print(FEATURE_COUNTS)

for k in FEATURE_COUNTS:

    print("\n----------------------------------------")
    print("Testing", k, "features")
    print("----------------------------------------")

    # --------------------------------------------------------
    # SelectKBest
    # --------------------------------------------------------

    selector = SelectKBest(
        score_func=f_classif,
        k=k
    )

    # --------------------------------------------------------
    # SVM PIPELINE
    # --------------------------------------------------------

    model = Pipeline(
        [
            (
                "selector",
                selector
            ),
            (
                "scaler",
                StandardScaler()
            ),
            (
                "classifier",
                SVC(
                    C=C_VALUE,
                    gamma=GAMMA_VALUE,
                    kernel="rbf",
                    random_state=42
                )
            )
        ]
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    validation_pred = model.predict(
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

    print(
        "Validation Accuracy :",
        f"{validation_accuracy:.4f}"
    )

    print(
        "Validation F1       :",
        f"{validation_f1:.4f}"
    )

    # --------------------------------------------------------
    # STORE VALIDATION RESULTS
    # --------------------------------------------------------

    results.append(
        {
            "features": k,
            "validation_accuracy": validation_accuracy,
            "validation_precision": validation_precision,
            "validation_recall": validation_recall,
            "validation_f1": validation_f1
        }
    )


# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="validation_f1",
    ascending=False
).reset_index(
    drop=True
)

results_file = os.path.join(
    OUTPUT_FOLDER,
    "feature_selection_results.csv"
)

results_df.to_csv(
    results_file,
    index=False
)

print("\n" + "=" * 40)
print(" FEATURE SELECTION RESULTS")
print("=" * 40)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# BEST FEATURE COUNT
# ============================================================

best_k = int(
    results_df.iloc[0]["features"]
)

best_validation_f1 = results_df.iloc[0][
    "validation_f1"
]

best_validation_accuracy = results_df.iloc[0][
    "validation_accuracy"
]

print("\n" + "=" * 40)
print(" BEST FEATURE CONFIGURATION")
print("=" * 40)

print("Best number of features :", best_k)
print(
    "Validation accuracy     :",
    f"{best_validation_accuracy:.4f}"
)
print(
    "Validation weighted F1  :",
    f"{best_validation_f1:.4f}"
)


# ============================================================
# FINAL MODEL
# TRAIN ON TRAIN + VALIDATION
# ============================================================

print("\n" + "=" * 40)
print(" FINAL MODEL TRAINING")
print("=" * 40)

X_train_full = np.concatenate(
    [
        X_train,
        X_validation
    ],
    axis=0
)

y_train_full = np.concatenate(
    [
        y_train,
        y_validation
    ],
    axis=0
)

print(
    "Training samples:",
    len(X_train_full)
)

final_model = Pipeline(
    [
        (
            "selector",
            SelectKBest(
                score_func=f_classif,
                k=best_k
            )
        ),
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            SVC(
                C=C_VALUE,
                gamma=GAMMA_VALUE,
                kernel="rbf",
                random_state=42
            )
        )
    ]
)

final_model.fit(
    X_train_full,
    y_train_full
)

print("Final model training complete.")


# ============================================================
# TEST
# ============================================================

print("\n" + "=" * 40)
print(" FINAL TEST PERFORMANCE")
print("=" * 40)

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

print(
    "Features        :",
    best_k
)

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


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 40)
print(" CLASSIFICATION REPORT")
print("=" * 40)

report = classification_report(
    y_test,
    test_pred,
    zero_division=0
)

print(report)

report_dict = classification_report(
    y_test,
    test_pred,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(
    report_dict
).transpose()

report_file = os.path.join(
    OUTPUT_FOLDER,
    "classification_report.csv"
)

report_df.to_csv(
    report_file
)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

import joblib

model_file = os.path.join(
    OUTPUT_FOLDER,
    "sound2see_feature_selected_svm.joblib"
)

joblib.dump(
    {
        "model": final_model,
        "feature_columns": feature_columns,
        "selected_features": best_k,
        "C": C_VALUE,
        "gamma": GAMMA_VALUE
    },
    model_file
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = pd.DataFrame(
    [
        {
            "model": "Feature Selected RBF SVM",
            "features": best_k,
            "C": C_VALUE,
            "gamma": GAMMA_VALUE,
            "test_accuracy": test_accuracy,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "test_weighted_f1": test_f1,
            "test_macro_f1": test_macro_f1
        }
    ]
)

summary_file = os.path.join(
    OUTPUT_FOLDER,
    "feature_selected_model_results.csv"
)

summary.to_csv(
    summary_file,
    index=False
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 40)
print(" FILES SAVED")
print("=" * 40)

print(
    "Feature selection results :",
    results_file
)

print(
    "Classification report     :",
    report_file
)

print(
    "Model                     :",
    model_file
)

print(
    "Model results             :",
    summary_file
)

print("\n" + "=" * 40)
print(" FEATURE SELECTION COMPLETE")
print("=" * 40)
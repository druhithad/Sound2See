import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# FILES
# ============================================================

BASELINE_FILE = (
    r"data\processed\features.csv"
)

TEMPORAL_FILE = (
    r"data\processed\temporal_features.csv"
)

METADATA_FILE = (
    r"data\raw\esc50_selected\selected_metadata.csv"
)


# ============================================================
# MODEL
# ============================================================

def create_svm():

    return Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            SVC(
                kernel="rbf",
                random_state=42
            )
        )
    ])


# ============================================================
# ADD FOLD INFORMATION
# ============================================================

def add_fold_information(data):

    metadata = pd.read_csv(
        METADATA_FILE
    )

    # Keep only filename and fold
    fold_data = metadata[
        ["filename", "fold"]
    ].drop_duplicates(
        subset=["filename"]
    )

    # Remove fold if already present
    if "fold" in data.columns:
        data = data.drop(
            columns=["fold"]
        )

    # Add fold information
    data = data.merge(
        fold_data,
        on="filename",
        how="left"
    )

    # Check for missing folds
    missing_folds = data["fold"].isna().sum()

    if missing_folds > 0:

        raise ValueError(
            f"{missing_folds} recordings "
            "could not be assigned a fold."
        )

    return data


# ============================================================
# CROSS VALIDATION
# ============================================================

def run_cross_validation(
    data,
    feature_name
):

    print("\n========================================")
    print(
        f" {feature_name.upper()}"
    )
    print("========================================")

    results = []

    available_folds = [
        1,
        2,
        3,
        4
    ]

    for validation_fold in available_folds:

        training_folds = [
            fold
            for fold in available_folds
            if fold != validation_fold
        ]

        train = data[
            data["fold"].isin(
                training_folds
            )
        ]

        validation = data[
            data["fold"] == validation_fold
        ]

        X_train = train.drop(
            columns=[
                "filename",
                "class",
                "fold"
            ]
        )

        y_train = train["class"]

        X_validation = validation.drop(
            columns=[
                "filename",
                "class",
                "fold"
            ]
        )

        y_validation = validation["class"]

        model = create_svm()

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(
            X_validation
        )

        accuracy = accuracy_score(
            y_validation,
            predictions
        )

        precision = precision_score(
            y_validation,
            predictions,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            y_validation,
            predictions,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            y_validation,
            predictions,
            average="weighted",
            zero_division=0
        )

        results.append({
            "validation_fold": validation_fold,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })

        print(
            f"Fold {validation_fold}: "
            f"Accuracy={accuracy:.4f}, "
            f"Precision={precision:.4f}, "
            f"Recall={recall:.4f}, "
            f"F1={f1:.4f}"
        )

    results_df = pd.DataFrame(
        results
    )

    print("\n----------------------------------------")
    print("AVERAGE")
    print("----------------------------------------")

    print(
        f"Accuracy : "
        f"{results_df['accuracy'].mean():.4f}"
    )

    print(
        f"Precision: "
        f"{results_df['precision'].mean():.4f}"
    )

    print(
        f"Recall   : "
        f"{results_df['recall'].mean():.4f}"
    )

    print(
        f"F1-score : "
        f"{results_df['f1'].mean():.4f}"
    )

    print(
        f"Accuracy Std: "
        f"{results_df['accuracy'].std():.4f}"
    )

    return results_df


# ============================================================
# MAIN
# ============================================================

def main():

    print("========================================")
    print(" SOUND2SEE CROSS-VALIDATION EXPERIMENT")
    print("========================================")

    # --------------------------------------------------------
    # Load baseline features
    # --------------------------------------------------------

    baseline = pd.read_csv(
        BASELINE_FILE
    )

    # Add fold information to baseline
    baseline = add_fold_information(
        baseline
    )

    print("\nBaseline dataset:")
    print(
        f"Rows    : {len(baseline)}"
    )

    print(
        f"Features: "
        f"{len(baseline.columns) - 3}"
    )

    print(
        "Fold distribution:"
    )

    print(
        baseline["fold"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # --------------------------------------------------------
    # Load temporal features
    # --------------------------------------------------------

    temporal = pd.read_csv(
        TEMPORAL_FILE
    )

    # Ensure temporal dataset has fold
    temporal = add_fold_information(
        temporal
    )

    print("\nTemporal dataset:")
    print(
        f"Rows    : {len(temporal)}"
    )

    print(
        f"Features: "
        f"{len(temporal.columns) - 3}"
    )

    print(
        "Fold distribution:"
    )

    print(
        temporal["fold"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # --------------------------------------------------------
    # Run baseline
    # --------------------------------------------------------

    baseline_results = run_cross_validation(
        baseline,
        "88-Feature Baseline SVM"
    )

    # --------------------------------------------------------
    # Run temporal
    # --------------------------------------------------------

    temporal_results = run_cross_validation(
        temporal,
        "140-Feature Temporal SVM"
    )

    # --------------------------------------------------------
    # Final comparison
    # --------------------------------------------------------

    print("\n========================================")
    print("       CROSS-VALIDATION COMPARISON")
    print("========================================")

    print(
        f"\nBaseline mean accuracy : "
        f"{baseline_results['accuracy'].mean():.4f}"
    )

    print(
        f"Temporal mean accuracy : "
        f"{temporal_results['accuracy'].mean():.4f}"
    )

    print(
        f"\nBaseline mean F1       : "
        f"{baseline_results['f1'].mean():.4f}"
    )

    print(
        f"Temporal mean F1       : "
        f"{temporal_results['f1'].mean():.4f}"
    )

    print(
        f"\nBaseline accuracy std  : "
        f"{baseline_results['accuracy'].std():.4f}"
    )

    print(
        f"Temporal accuracy std  : "
        f"{temporal_results['accuracy'].std():.4f}"
    )

    print("\n========================================")
    print(" CROSS-VALIDATION COMPLETE")
    print("========================================")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
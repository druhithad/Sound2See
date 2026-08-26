import os
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
# FILE
# ============================================================

FEATURE_FILE = (
    r"data\processed\temporal_features.csv"
)

OUTPUT_FOLDER = (
    r"outputs\temporal_experiment"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("========================================")
    print("   SOUND2SEE TEMPORAL FEATURE EXPERIMENT")
    print("========================================")

    data = pd.read_csv(
        FEATURE_FILE
    )

    # --------------------------------------------------------
    # Split using existing ESC-50 folds
    # --------------------------------------------------------

    train = data[
        data["fold"].isin([1, 2, 3])
    ]

    validation = data[
        data["fold"] == 4
    ]

    # IMPORTANT:
    # Fold 5 is deliberately NOT used here.

    print(
        f"\nTraining samples   : {len(train)}"
    )

    print(
        f"Validation samples : {len(validation)}"
    )

    print(
        "Test fold 5        : RESERVED"
    )

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    drop_columns = [
        "filename",
        "class",
        "fold"
    ]

    X_train = train.drop(
        columns=drop_columns
    )

    y_train = train["class"]

    X_validation = validation.drop(
        columns=drop_columns
    )

    y_validation = validation["class"]

    print(
        f"Features            : {X_train.shape[1]}"
    )

    # --------------------------------------------------------
    # SVM
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

    print("\nTraining temporal SVM...")

    model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # Validation only
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\n========================================")
    print("      TEMPORAL SVM VALIDATION")
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
    # Save validation result
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_FOLDER,
        exist_ok=True
    )

    result = pd.DataFrame([
        {
            "experiment": "Temporal MFCC + spectral + SVM",
            "training_samples": len(train),
            "validation_samples": len(validation),
            "features": X_train.shape[1],
            "validation_accuracy": accuracy,
            "validation_precision": precision,
            "validation_recall": recall,
            "validation_f1": f1
        }
    ])

    result_file = os.path.join(
        OUTPUT_FOLDER,
        "temporal_validation_results.csv"
    )

    result.to_csv(
        result_file,
        index=False
    )

    print(
        f"\nResults saved to:"
    )

    print(result_file)

    print("\n========================================")
    print("       TEMPORAL EXPERIMENT COMPLETE")
    print("========================================")


if __name__ == "__main__":
    main()
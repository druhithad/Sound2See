import os
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix
)


# ============================================================
# FILES
# ============================================================

TEST_FILE = r"data\processed\splits\test.csv"

MODEL_FILE = (
    r"outputs\final_baseline\sound2see_svm_baseline.joblib"
)

RESULT_FILE = (
    r"outputs\final_baseline\final_baseline_results.csv"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("========================================")
    print("       SOUND2SEE BASELINE VERIFY")
    print("========================================")

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    for file_path in [
        TEST_FILE,
        MODEL_FILE,
        RESULT_FILE
    ]:

        print(
            f"\nChecking: {file_path}"
        )

        if not os.path.exists(file_path):

            print("ERROR: File not found.")
            return

        print("OK")

    # --------------------------------------------------------
    # Load test dataset
    # --------------------------------------------------------

    test_data = pd.read_csv(
        TEST_FILE
    )

    X_test = test_data.drop(
        columns=[
            "filename",
            "class",
            "fold"
        ]
    )

    y_test = test_data["class"]

    # --------------------------------------------------------
    # Load saved model
    # --------------------------------------------------------

    model = joblib.load(
        MODEL_FILE
    )

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    predictions = model.predict(
        X_test
    )

    # --------------------------------------------------------
    # Calculate current accuracy
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=model.classes_
    )

    # --------------------------------------------------------
    # Print current result
    # --------------------------------------------------------

    print("\n========================================")
    print("        CURRENT SAVED MODEL")
    print("========================================")

    print(
        "Test samples :",
        len(y_test)
    )

    print(
        "Correct      :",
        (predictions == y_test).sum()
    )

    print(
        "Incorrect    :",
        (predictions != y_test).sum()
    )

    print(
        f"Accuracy     : {accuracy:.4f}"
    )

    print(
        f"Accuracy %   : {accuracy * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Class order
    # --------------------------------------------------------

    print("\nClass order:")

    for index, class_name in enumerate(
        model.classes_
    ):

        print(
            f"{index} -> {class_name}"
        )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    print("\n========================================")
    print("          CURRENT CONFUSION MATRIX")
    print("========================================")

    print(matrix)

    # --------------------------------------------------------
    # Load saved result
    # --------------------------------------------------------

    saved_results = pd.read_csv(
        RESULT_FILE
    )

    print("\n========================================")
    print("        SAVED RESULT FILE")
    print("========================================")

    print(
        saved_results.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Compare
    # --------------------------------------------------------

    saved_accuracy = float(
        saved_results.iloc[0]["accuracy"]
    )

    print("\n========================================")
    print("          CONSISTENCY CHECK")
    print("========================================")

    print(
        f"Current accuracy : {accuracy:.4f}"
    )

    print(
        f"Saved accuracy   : {saved_accuracy:.4f}"
    )

    if abs(
        accuracy - saved_accuracy
    ) < 1e-10:

        print(
            "\nRESULT: CONSISTENT"
        )

    else:

        print(
            "\nRESULT: INCONSISTENT"
        )

    print("\n========================================")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
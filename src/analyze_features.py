import os
import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

INPUT_FILE = (
    r"data\processed\combined\enhanced_features.csv"
)

OUTPUT_DIR = (
    r"data\processed\analysis"
)

CORRELATION_THRESHOLD = 0.95
LOW_VARIANCE_THRESHOLD = 1e-5


# ============================================================
# MAIN FEATURE ANALYSIS
# ============================================================

def main():

    print("========================================")
    print(" SOUND2SEE FEATURE ANALYSIS")
    print("========================================")

    # ========================================================
    # LOAD DATA
    # ========================================================

    if not os.path.exists(INPUT_FILE):

        print("\nERROR: Input file not found:")
        print(INPUT_FILE)
        return

    df = pd.read_csv(INPUT_FILE)

    print("\nInput file:")
    print(INPUT_FILE)

    print("\nDataset shape:")
    print(df.shape)

    # ========================================================
    # IDENTIFIER COLUMNS
    # ========================================================

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

    print("\n========================================")
    print(" BASIC INFORMATION")
    print("========================================")

    print(
        "Total recordings :",
        len(df)
    )

    print(
        "Feature columns  :",
        len(feature_columns)
    )

    print(
        "Metadata columns :",
        len(identifier_columns)
    )

    print(
        "Numeric columns  :",
        X.select_dtypes(
            include=np.number
        ).shape[1]
    )

    # ========================================================
    # DATA TYPE CHECK
    # ========================================================

    print("\n========================================")
    print(" DATA TYPE CHECK")
    print("========================================")

    non_numeric = [
        column
        for column in feature_columns
        if not pd.api.types.is_numeric_dtype(
            X[column]
        )
    ]

    if len(non_numeric) == 0:

        print(
            "All feature columns are numeric."
        )

    else:

        print(
            "WARNING: Non-numeric features found:"
        )

        for column in non_numeric:
            print(
                " -",
                column
            )

    # ========================================================
    # MISSING VALUES
    # ========================================================

    print("\n========================================")
    print(" MISSING VALUE CHECK")
    print("========================================")

    nan_counts = X.isna().sum()

    total_nan = nan_counts.sum()

    print(
        "Total NaN values:",
        total_nan
    )

    if total_nan > 0:

        print("\nFeatures containing NaN:")

        print(
            nan_counts[
                nan_counts > 0
            ]
        )

    else:

        print(
            "No NaN values found."
        )

    # ========================================================
    # INFINITE VALUES
    # ========================================================

    print("\n========================================")
    print(" INFINITE VALUE CHECK")
    print("========================================")

    infinite_counts = pd.Series(
        np.isinf(
            X.select_dtypes(
                include=np.number
            ).values
        ).sum(axis=0),
        index=X.select_dtypes(
            include=np.number
        ).columns
    )

    total_infinite = infinite_counts.sum()

    print(
        "Total infinite values:",
        total_infinite
    )

    if total_infinite > 0:

        print("\nFeatures containing infinity:")

        print(
            infinite_counts[
                infinite_counts > 0
            ]
        )

    else:

        print(
            "No infinite values found."
        )

    # ========================================================
    # DESCRIPTIVE STATISTICS
    # ========================================================

    print("\n========================================")
    print(" FEATURE STATISTICS")
    print("========================================")

    statistics = X.describe().T

    statistics["range"] = (
        statistics["max"]
        -
        statistics["min"]
    )

    statistics["variance"] = X.var()

    print(
        statistics[
            [
                "mean",
                "std",
                "min",
                "max",
                "range",
                "variance"
            ]
        ].head(20)
    )

    # ========================================================
    # LOW VARIANCE FEATURES
    # ========================================================

    print("\n========================================")
    print(" LOW VARIANCE ANALYSIS")
    print("========================================")

    variances = X.var()

    low_variance_features = (
        variances[
            variances <= LOW_VARIANCE_THRESHOLD
        ]
        .sort_values()
    )

    print(
        "Low-variance threshold:",
        LOW_VARIANCE_THRESHOLD
    )

    print(
        "Low-variance features:",
        len(low_variance_features)
    )

    if len(low_variance_features) > 0:

        print("\nFeatures:")

        for feature, variance in (
            low_variance_features.items()
        ):

            print(
                f"{feature}: "
                f"{variance:.10f}"
            )

    else:

        print(
            "No low-variance features found."
        )

    # ========================================================
    # HIGHLY CORRELATED FEATURES
    # ========================================================

    print("\n========================================")
    print(" CORRELATION ANALYSIS")
    print("========================================")

    correlation_matrix = X.corr()

    upper_triangle = (
        correlation_matrix.where(
            np.triu(
                np.ones(
                    correlation_matrix.shape
                ),
                k=1
            ).astype(bool)
        )
    )

    highly_correlated_pairs = []

    for column in upper_triangle.columns:

        for row in upper_triangle.index:

            correlation = (
                upper_triangle.loc[
                    row,
                    column
                ]
            )

            if pd.notna(correlation):

                if abs(correlation) >= CORRELATION_THRESHOLD:

                    highly_correlated_pairs.append(
                        {
                            "feature_1": row,
                            "feature_2": column,
                            "correlation": correlation
                        }
                    )

    correlation_pairs = pd.DataFrame(
        highly_correlated_pairs
    )

    print(
        "Correlation threshold:",
        CORRELATION_THRESHOLD
    )

    print(
        "Highly correlated pairs:",
        len(correlation_pairs)
    )

    if len(correlation_pairs) > 0:

        print(
            "\nTop highly correlated pairs:"
        )

        print(
            correlation_pairs
            .sort_values(
                "correlation",
                key=abs,
                ascending=False
            )
            .head(30)
            .to_string(
                index=False
            )
        )

    else:

        print(
            "No highly correlated feature pairs found."
        )

    # ========================================================
    # CLASS DISTRIBUTION
    # ========================================================

    print("\n========================================")
    print(" CLASS DISTRIBUTION")
    print("========================================")

    class_distribution = (
        df["class"]
        .value_counts()
        .sort_index()
    )

    print(
        class_distribution
    )

    # ========================================================
    # DATASET DISTRIBUTION
    # ========================================================

    print("\n========================================")
    print(" DATASET DISTRIBUTION")
    print("========================================")

    print(
        df["dataset"]
        .value_counts()
    )

    # ========================================================
    # SPLIT DISTRIBUTION
    # ========================================================

    print("\n========================================")
    print(" SPLIT DISTRIBUTION")
    print("========================================")

    split_distribution = pd.crosstab(
        df["split"],
        df["dataset"]
    )

    print(
        split_distribution
    )

    print("\nDetailed split counts:")

    print(
        df["split"]
        .value_counts()
    )

    # ========================================================
    # CLASS × SPLIT
    # ========================================================

    print("\n========================================")
    print(" CLASS × SPLIT DISTRIBUTION")
    print("========================================")

    class_split = pd.crosstab(
        df["class"],
        df["split"]
    )

    print(
        class_split
    )

    # ========================================================
    # FEATURE GROUP ANALYSIS
    # ========================================================

    print("\n========================================")
    print(" FEATURE GROUP ANALYSIS")
    print("========================================")

    feature_groups = {
        "MFCC temporal": [
            column
            for column in feature_columns
            if column.startswith("mfcc_s")
        ],

        "MFCC delta": [
            column
            for column in feature_columns
            if column.startswith("delta_")
        ],

        "MFCC delta-delta": [
            column
            for column in feature_columns
            if column.startswith("delta2_")
        ],

        "Basic spectral": [
            column
            for column in feature_columns
            if (
                column.startswith(
                    "spectral_centroid"
                )
                or column.startswith(
                    "spectral_bandwidth"
                )
                or column.startswith(
                    "spectral_rolloff"
                )
                or column.startswith(
                    "zero_crossing_rate"
                )
                or column.startswith(
                    "rms_energy"
                )
            )
        ],

        "Chroma": [
            column
            for column in feature_columns
            if column.startswith("chroma_")
        ],

        "Spectral contrast": [
            column
            for column in feature_columns
            if column.startswith(
                "spectral_contrast_"
            )
        ],

        "Tonnetz": [
            column
            for column in feature_columns
            if column.startswith("tonnetz_")
        ]
    }

    for group_name, columns in feature_groups.items():

        print(
            f"{group_name:20s}: "
            f"{len(columns)} features"
        )

    # ========================================================
    # SAVE ANALYSIS RESULTS
    # ========================================================

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    statistics_file = os.path.join(
        OUTPUT_DIR,
        "feature_statistics.csv"
    )

    statistics.to_csv(
        statistics_file
    )

    correlation_file = os.path.join(
        OUTPUT_DIR,
        "high_correlation_pairs.csv"
    )

    correlation_pairs.to_csv(
        correlation_file,
        index=False
    )

    low_variance_file = os.path.join(
        OUTPUT_DIR,
        "low_variance_features.csv"
    )

    low_variance_features.to_csv(
        low_variance_file,
        header=["variance"]
    )

    class_split_file = os.path.join(
        OUTPUT_DIR,
        "class_split_distribution.csv"
    )

    class_split.to_csv(
        class_split_file
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n========================================")
    print(" FEATURE ANALYSIS SUMMARY")
    print("========================================")

    print(
        "Recordings              :",
        len(df)
    )

    print(
        "Features                :",
        len(feature_columns)
    )

    print(
        "NaN values              :",
        total_nan
    )

    print(
        "Infinite values         :",
        total_infinite
    )

    print(
        "Low variance features   :",
        len(low_variance_features)
    )

    print(
        "Highly correlated pairs :",
        len(correlation_pairs)
    )

    print(
        "\nAnalysis files saved to:"
    )

    print(
        OUTPUT_DIR
    )

    print("\n========================================")
    print(" FEATURE ANALYSIS COMPLETE")
    print("========================================")


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
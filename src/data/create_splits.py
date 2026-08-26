from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


# ------------------------------------------------
# PATHS
# ------------------------------------------------

METADATA_PATH = Path("data/processed/metadata/metadata.csv")
SPLIT_DIR = Path("data/processed/splits")

SPLIT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------
# SETTINGS
# ------------------------------------------------

RANDOM_STATE = 42

TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15


# ------------------------------------------------
# LOAD METADATA
# ------------------------------------------------

print("Creating dataset splits...")

df = pd.read_csv(METADATA_PATH)

print(f"Total samples : {len(df)}")
print(f"Total classes : {df['label'].nunique()}")


# ------------------------------------------------
# CHECK CLASS COUNTS
# ------------------------------------------------

print("\nClass distribution:")
print(df["label"].value_counts())


# ------------------------------------------------
# FIRST SPLIT
# Train = 70%
# Temporary = 30%
# ------------------------------------------------

train_df, temp_df = train_test_split(
    df,
    test_size=(VAL_SIZE + TEST_SIZE),
    random_state=RANDOM_STATE,
    stratify=df["label"]
)


# ------------------------------------------------
# SECOND SPLIT
# Validation = 15%
# Test = 15%
# ------------------------------------------------

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=temp_df["label"]
)


# ------------------------------------------------
# RESET INDEX
# ------------------------------------------------

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)


# ------------------------------------------------
# SAVE SPLITS
# ------------------------------------------------

train_df.to_csv(
    SPLIT_DIR / "train.csv",
    index=False
)

val_df.to_csv(
    SPLIT_DIR / "validation.csv",
    index=False
)

test_df.to_csv(
    SPLIT_DIR / "test.csv",
    index=False
)


# ------------------------------------------------
# DISPLAY RESULTS
# ------------------------------------------------

print("\n==============================")
print("DATASET SPLITS CREATED")
print("==============================")

print(f"Training samples   : {len(train_df)}")
print(f"Validation samples : {len(val_df)}")
print(f"Test samples       : {len(test_df)}")
print(f"Total samples      : {len(train_df) + len(val_df) + len(test_df)}")

print("\nTraining distribution:")
print(train_df["label"].value_counts().sort_index())

print("\nValidation distribution:")
print(val_df["label"].value_counts().sort_index())

print("\nTest distribution:")
print(test_df["label"].value_counts().sort_index())

print("\nSaved to:")
print(SPLIT_DIR)

print("==============================")
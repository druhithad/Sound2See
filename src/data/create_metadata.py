from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/processed/audio")
OUTPUT = Path("data/processed/metadata/metadata.csv")

rows = []

for class_dir in sorted(DATA_DIR.iterdir()):

    if not class_dir.is_dir():
        continue

    for audio_file in sorted(class_dir.glob("*.wav")):

        rows.append({
            "file_path": str(audio_file).replace("\\", "/"),
            "label": class_dir.name
        })

df = pd.DataFrame(rows)

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(OUTPUT, index=False)

print("\n==============================")
print("METADATA CREATED")
print("==============================")
print(f"Total samples : {len(df)}")
print(f"Classes       : {df['label'].nunique()}")
print("\nClass distribution:")
print(df["label"].value_counts().sort_index())
print("==============================")
print(f"\nSaved to: {OUTPUT}")
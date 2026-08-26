from pathlib import Path
import librosa
import soundfile as sf

DATASETS = [
    Path("data/raw/animal_sounds_caoofficial/Animal-SDataset"),
    Path("data/raw/animal_sounds_ggquang/new"),
]

OUTPUT_DIR = Path("data/processed/audio")
TARGET_SR = 16000
TARGET_DURATION = 4
TARGET_LENGTH = TARGET_SR * TARGET_DURATION

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

processed = 0
failed = 0

print("\nAudio preprocessing started...\n")

for dataset in DATASETS:
    for animal_dir in dataset.iterdir():

        if not animal_dir.is_dir():
            continue

        animal = animal_dir.name
        output_class_dir = OUTPUT_DIR / animal
        output_class_dir.mkdir(parents=True, exist_ok=True)

        for audio_file in animal_dir.rglob("*.wav"):

            try:
                audio, _ = librosa.load(
                    audio_file,
                    sr=TARGET_SR,
                    mono=True
                )

                # Trim or pad to exactly 4 seconds
                if len(audio) > TARGET_LENGTH:
                    audio = audio[:TARGET_LENGTH]

                elif len(audio) < TARGET_LENGTH:
                    audio = librosa.util.fix_length(
                        audio,
                        size=TARGET_LENGTH
                    )

                # Unique filename based on dataset + original filename
                dataset_name = dataset.parts[-2]
                output_name = f"{dataset_name}_{audio_file.stem}.wav"

                output_path = output_class_dir / output_name

                sf.write(
                    output_path,
                    audio,
                    TARGET_SR,
                    subtype="PCM_16"
                )

                processed += 1

            except Exception as e:
                failed += 1
                print(f"FAILED: {audio_file}")
                print(f"Reason: {e}")

print("\n==============================")
print("PREPROCESSING COMPLETE")
print("==============================")
print(f"Processed : {processed}")
print(f"Failed    : {failed}")
print("==============================\n")
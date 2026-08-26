from pathlib import Path
import librosa
import soundfile as sf

DATASETS = [
    Path("data/raw/animal_sounds_caoofficial/Animal-SDataset"),
    Path("data/raw/animal_sounds_ggquang/new"),
]

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

total = 0
valid = 0
invalid = 0

print("\nAudio validation started...\n")

for dataset in DATASETS:
    if not dataset.exists():
        print(f"Dataset not found: {dataset}")
        continue

    for animal_dir in dataset.iterdir():
        if not animal_dir.is_dir():
            continue

        for audio_file in animal_dir.rglob("*"):
            if audio_file.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            total += 1

            try:
                audio, sr = librosa.load(audio_file, sr=None, mono=True)

                if len(audio) == 0 or sr <= 0:
                    raise ValueError("Empty or invalid audio")

                valid += 1

            except Exception as e:
                invalid += 1
                print(f"INVALID: {audio_file}")
                print(f"Reason: {e}")

print("\n==============================")
print("AUDIO VALIDATION COMPLETE")
print("==============================")
print(f"Total files : {total}")
print(f"Valid files : {valid}")
print(f"Invalid     : {invalid}")
print("==============================\n")
import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt


# ============================================================
# FILES
# ============================================================

ERROR_FILE = (
    r"outputs\error_analysis\misclassified_samples.csv"
)

AUDIO_FOLDER = (
    r"data\processed\esc50_16khz"
)

OUTPUT_FOLDER = (
    r"outputs\error_audio_analysis"
)


# ============================================================
# SETTINGS
# ============================================================

SAMPLE_RATE = 16000


# ============================================================
# MAIN
# ============================================================

def main():

    print("========================================")
    print("     SOUND2SEE ERROR AUDIO ANALYSIS")
    print("========================================")

    os.makedirs(
        OUTPUT_FOLDER,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load error information
    # --------------------------------------------------------

    errors = pd.read_csv(
        ERROR_FILE
    )

    print(
        f"\nMisclassified recordings: {len(errors)}"
    )

    # --------------------------------------------------------
    # Process each error
    # --------------------------------------------------------

    for index, row in errors.iterrows():

        filename = row["filename"]

        actual_class = row["class"]

        predicted_class = row["predicted_class"]

        audio_file = os.path.join(
            AUDIO_FOLDER,
            actual_class,
            filename
        )

        print("\n----------------------------------------")

        print(
            f"File      : {filename}"
        )

        print(
            f"Actual    : {actual_class}"
        )

        print(
            f"Predicted : {predicted_class}"
        )

        if not os.path.exists(audio_file):

            print(
                "ERROR: Audio file not found:"
            )

            print(audio_file)

            continue

        # ----------------------------------------------------
        # Load audio
        # ----------------------------------------------------

        audio, sr = librosa.load(
            audio_file,
            sr=SAMPLE_RATE,
            mono=True
        )

        # ----------------------------------------------------
        # Create waveform
        # ----------------------------------------------------

        time = (
            librosa.frames_to_time(
                range(len(audio)),
                sr=sr
            )
        )

        # Use simple sample-based time axis
        time = (
            range(len(audio))
        )

        time = [
            sample / sr
            for sample in time
        ]

        plt.figure(
            figsize=(12, 4)
        )

        plt.plot(
            time,
            audio
        )

        plt.title(
            f"Actual: {actual_class} | "
            f"Predicted: {predicted_class}"
        )

        plt.xlabel(
            "Time (seconds)"
        )

        plt.ylabel(
            "Amplitude"
        )

        plt.tight_layout()

        waveform_file = os.path.join(
            OUTPUT_FOLDER,
            f"{index + 1}_{filename[:-4]}_waveform.png"
        )

        plt.savefig(
            waveform_file,
            dpi=150
        )

        plt.close()

        # ----------------------------------------------------
        # Create Mel-spectrogram
        # ----------------------------------------------------

        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=64,
            n_fft=1024,
            hop_length=512
        )

        mel_db = librosa.power_to_db(
            mel,
            ref=np.max
        )
        
        plt.figure(
            figsize=(12, 5)
        )

        librosa.display.specshow(
            mel_db,
            sr=sr,
            hop_length=512,
            x_axis="time",
            y_axis="mel"
        )

        plt.colorbar(
            format="%+2.0f dB"
        )

        plt.title(
            f"Actual: {actual_class} | "
            f"Predicted: {predicted_class}"
        )

        plt.tight_layout()

        spectrogram_file = os.path.join(
            OUTPUT_FOLDER,
            f"{index + 1}_{filename[:-4]}_melspectrogram.png"
        )

        plt.savefig(
            spectrogram_file,
            dpi=150
        )

        plt.close()

        print(
            "Waveform saved:"
        )

        print(
            waveform_file
        )

        print(
            "Mel-spectrogram saved:"
        )

        print(
            spectrogram_file
        )

    # --------------------------------------------------------
    # Final message
    # --------------------------------------------------------

    print("\n========================================")
    print("       ERROR VISUALIZATION COMPLETE")
    print("========================================")

    print(
        "\nFiles saved in:"
    )

    print(
        OUTPUT_FOLDER
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
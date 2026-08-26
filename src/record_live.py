import os
import sounddevice as sd
import soundfile as sf
import numpy as np


# ============================================================
# SETTINGS
# ============================================================

SAMPLE_RATE = 48000
CHANNELS = 2
DURATION = 5
DEVICE = 11

OUTPUT_FILE = r"data\inference\live_test.wav"


# ============================================================
# MAIN
# ============================================================

def main():

    print("========================================")
    print("       SOUND2SEE LIVE RECORDER")
    print("========================================")

    os.makedirs(
        r"data\inference",
        exist_ok=True
    )

    print("\nSample rate :", SAMPLE_RATE, "Hz")
    print("Channels    :", CHANNELS)
    print("Duration    :", DURATION, "seconds")
    print("Device      :", DEVICE)

    input("\nPress ENTER to start recording...")

    print("\nRecording...")
    print(">>> PLAY THE KNOWN SOUND <<<")

    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        device=DEVICE
    )

    sd.wait()

    print("\nRecording complete.")

    # --------------------------------------------------------
    # Check both channels
    # --------------------------------------------------------

    rms1 = np.sqrt(
        np.mean(
            audio[:, 0] ** 2
        )
    )

    rms2 = np.sqrt(
        np.mean(
            audio[:, 1] ** 2
        )
    )

    peak1 = np.max(
        np.abs(
            audio[:, 0]
        )
    )

    peak2 = np.max(
        np.abs(
            audio[:, 1]
        )
    )

    print("\nChannel 1 RMS  :", round(float(rms1), 4))
    print("Channel 1 Peak :", round(float(peak1), 4))

    print("\nChannel 2 RMS  :", round(float(rms2), 4))
    print("Channel 2 Peak :", round(float(peak2), 4))

    # --------------------------------------------------------
    # Select stronger channel
    # --------------------------------------------------------

    if rms2 > rms1:
        selected = audio[:, 1]
        selected_channel = 2
    else:
        selected = audio[:, 0]
        selected_channel = 1

    print(
        "\nSelected channel :",
        selected_channel
    )

    # --------------------------------------------------------
    # Save as mono 48 kHz WAV
    # --------------------------------------------------------

    sf.write(
        OUTPUT_FILE,
        selected,
        SAMPLE_RATE
    )

    print(
        "\nSaved recording:"
    )

    print(
        OUTPUT_FILE
    )

    print("\n========================================")
    print("       RECORDING COMPLETE")
    print("========================================")


if __name__ == "__main__":
    main()
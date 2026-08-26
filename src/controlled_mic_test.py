import sounddevice as sd
import soundfile as sf
import numpy as np
import os
import time
import librosa


REFERENCE = r"C:\Users\D LAHARI\Downloads\ESC-50-master\ESC-50-master\audio\5-179860-A-43.wav"
OUTPUT = r"data\inference\controlled_car_horn.wav"

RATE = 48000
CHANNELS = 2
DURATION = 5
INPUT_DEVICE = 11
OUTPUT_DEVICE = 4


print("=" * 55)
print("SOUND2SEE CONTROLLED CAR HORN TEST")
print("=" * 55)

audio, sr = sf.read(
    REFERENCE,
    dtype="float32"
)

if audio.ndim > 1:
    audio = np.mean(audio, axis=1)

audio = librosa.resample(
    audio,
    orig_sr=sr,
    target_sr=RATE
)

target = RATE * DURATION

if len(audio) > target:
    audio = audio[:target]

elif len(audio) < target:
    audio = np.pad(
        audio,
        (0, target - len(audio))
    )

play_audio = np.column_stack(
    [audio, audio]
)

print("\nReference:")
print("5-179860-A-43.wav")
print("Known class: car_horn")

print("\nPlayback device :", OUTPUT_DEVICE)
print("Microphone      :", INPUT_DEVICE)
print("Sample rate     :", RATE)
print("Duration        :", DURATION)

print("\nIMPORTANT")
print("1. Put the microphone close to the laptop speakers.")
print("2. Keep the room quiet.")
print("3. Do not speak.")
print("4. Let the complete car horn play.")

input("\nPress ENTER to start...")

print("\nRecording + playback starting...")

recording = sd.rec(
    target,
    samplerate=RATE,
    channels=CHANNELS,
    dtype="float32",
    device=INPUT_DEVICE
)

time.sleep(0.5)

sd.play(
    play_audio,
    samplerate=RATE,
    device=OUTPUT_DEVICE
)

sd.wait()
sd.wait()

print("\nRecording complete.")

recording = np.asarray(recording)

os.makedirs(
    os.path.dirname(OUTPUT),
    exist_ok=True
)

sf.write(
    OUTPUT,
    recording,
    RATE
)

print("\nSaved:")
print(OUTPUT)

for ch in range(CHANNELS):

    x = recording[:, ch]

    rms = np.sqrt(
        np.mean(x ** 2)
    )

    peak = np.max(
        np.abs(x)
    )

    print(
        f"\nChannel {ch + 1}"
    )

    print(
        "RMS  :",
        round(float(rms), 6)
    )

    print(
        "Peak :",
        round(float(peak), 6)
    )

print("\n" + "=" * 55)
print("CONTROLLED TEST COMPLETE")
print("=" * 55)
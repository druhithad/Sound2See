import sounddevice as sd
import soundfile as sf
import numpy as np


SAMPLE_RATE = 16000
DURATION = 5
CHANNELS = 1

OUTPUT_FILE = r"data\inference\microphone_test.wav"


print("========================================")
print("       SOUND2SEE MICROPHONE TEST")
print("========================================")

print()
print(f"Sample rate : {SAMPLE_RATE} Hz")
print(f"Duration    : {DURATION} seconds")
print(f"Channels    : {CHANNELS}")

print()
print("Get ready...")
input("Press ENTER to start recording...")

print()
print("Recording...")
print(">>> MAKE A SOUND <<<")

audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=CHANNELS,
    dtype="float32"
)

sd.wait()

print()
print("Recording complete.")

# Convert shape from (samples, 1) to (samples,)
audio = np.squeeze(audio)

# Save WAV
sf.write(
    OUTPUT_FILE,
    audio,
    SAMPLE_RATE
)

print()
print("========================================")
print("       RECORDING SAVED")
print("========================================")

print()
print(f"File     : {OUTPUT_FILE}")
print(f"Samples  : {len(audio)}")
print(f"Duration : {len(audio) / SAMPLE_RATE:.2f} seconds")
print(f"Minimum  : {np.min(audio):.4f}")
print(f"Maximum  : {np.max(audio):.4f}")

print()
print("MICROPHONE TEST COMPLETE")
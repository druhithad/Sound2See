import sounddevice as sd
import soundfile as sf
import numpy as np
import librosa
import time
import sys
import os

# ============================================================
# SOUND2SEE SPEAKER -> MICROPHONE TEST
# ============================================================

INPUT_DEVICE = 11
OUTPUT_DEVICE = 4

INPUT_RATE = 48000
OUTPUT_RATE = 48000

DURATION = 5
CHANNELS = 2


# ============================================================
# ARGUMENTS
# ============================================================

if len(sys.argv) != 3:
    print("Usage:")
    print("python src\\speaker_microphone_test.py <source.wav> <output.wav>")
    sys.exit(1)

SOURCE_FILE = sys.argv[1]
OUTPUT_FILE = sys.argv[2]


# ============================================================
# HEADER
# ============================================================

print("=" * 60)
print("SOUND2SEE SPEAKER -> MICROPHONE TEST")
print("=" * 60)


# ============================================================
# LOAD SOURCE AUDIO
# ============================================================

print()
print("Loading source audio...")
print("Source :", SOURCE_FILE)

audio, sr = sf.read(SOURCE_FILE)

if audio.ndim > 1:
    audio = np.mean(audio, axis=1)

audio = audio.astype(np.float32)

print("Original sample rate :", sr)
print("Original samples     :", len(audio))
print("Original duration    :", len(audio) / sr, "seconds")


# ============================================================
# RESAMPLE
# ============================================================

if sr != OUTPUT_RATE:

    print()
    print(f"Resampling {sr} Hz -> {OUTPUT_RATE} Hz")

    audio = librosa.resample(
        audio,
        orig_sr=sr,
        target_sr=OUTPUT_RATE
    )

    sr = OUTPUT_RATE


# ============================================================
# MAKE EXACTLY 5 SECONDS
# ============================================================

target_samples = int(OUTPUT_RATE * DURATION)

if len(audio) > target_samples:

    audio = audio[:target_samples]

elif len(audio) < target_samples:

    audio = np.pad(
        audio,
        (0, target_samples - len(audio))
    )


# ============================================================
# NORMALIZE PLAYBACK
# ============================================================

peak = np.max(np.abs(audio))

if peak > 0:

    playback_audio = audio / peak * 0.7

else:

    print("ERROR: Source audio contains no signal.")
    sys.exit(1)


# ============================================================
# SOURCE INFORMATION
# ============================================================

source_name = os.path.basename(SOURCE_FILE)

print()
print("Source file        :", source_name)
print("Playback device    :", OUTPUT_DEVICE)
print("Microphone device  :", INPUT_DEVICE)
print("Playback rate      :", OUTPUT_RATE)
print("Recording rate     :", INPUT_RATE)
print("Duration           :", DURATION, "seconds")

print()
print("IMPORTANT:")
print("1. Place microphone near the speaker.")
print("2. Keep speaker volume reasonably high.")
print("3. Do not touch or move the microphone.")
print()


# ============================================================
# WAIT
# ============================================================

input("Press ENTER to start...")

print()
print("Starting in 2 seconds...")
time.sleep(2)


# ============================================================
# RECORDING
# ============================================================

print()
print("=" * 60)
print("PLAYING SOURCE + RECORDING MICROPHONE")
print("=" * 60)

print("Source :", source_name)
print(">>> DO NOT MOVE THE MICROPHONE <<<")
print()


# ------------------------------------------------------------
# Explicit InputStream
# ------------------------------------------------------------

recorded_chunks = []


def audio_callback(indata, frames, time_info, status):

    if status:
        print("Audio status:", status)

    recorded_chunks.append(indata.copy())


print("Starting microphone stream...")

with sd.InputStream(
    samplerate=INPUT_RATE,
    channels=CHANNELS,
    dtype="float32",
    device=INPUT_DEVICE,
    callback=audio_callback
):

    print("Microphone stream active.")

    time.sleep(0.3)

    print("Starting playback...")

    sd.play(
        playback_audio,
        samplerate=OUTPUT_RATE,
        device=OUTPUT_DEVICE
    )

    # Wait until playback finishes
    sd.wait()

    # Small extra recording time
    time.sleep(0.2)


print()
print("Recording complete.")


# ============================================================
# COMBINE RECORDED DATA
# ============================================================

if len(recorded_chunks) == 0:

    print("ERROR: No microphone data was received.")
    sys.exit(1)

recording = np.concatenate(
    recorded_chunks,
    axis=0
)

print("Recorded shape :", recording.shape)


# ============================================================
# TRIM TO 5 SECONDS
# ============================================================

target_record_samples = int(INPUT_RATE * DURATION)

if len(recording) > target_record_samples:

    recording = recording[:target_record_samples]

elif len(recording) < target_record_samples:

    missing = target_record_samples - len(recording)

    recording = np.pad(
        recording,
        ((0, missing), (0, 0))
    )


# ============================================================
# CHANNEL ANALYSIS
# ============================================================

print()
print("=" * 60)
print("MICROPHONE CHANNEL ANALYSIS")
print("=" * 60)

for i in range(recording.shape[1]):

    channel = recording[:, i]

    channel_rms = np.sqrt(
        np.mean(channel ** 2)
    )

    channel_peak = np.max(
        np.abs(channel)
    )

    print()
    print("Channel", i + 1)
    print("RMS  :", channel_rms)
    print("Peak :", channel_peak)


# ============================================================
# SELECT STRONGEST CHANNEL
# ============================================================

channel_rms_values = []

for i in range(recording.shape[1]):

    channel = recording[:, i]

    rms = np.sqrt(
        np.mean(channel ** 2)
    )

    channel_rms_values.append(rms)


selected_channel = int(
    np.argmax(channel_rms_values)
)

recording_mono = recording[:, selected_channel]


# ============================================================
# FINAL STATISTICS
# ============================================================

rms = np.sqrt(
    np.mean(recording_mono ** 2)
)

peak = np.max(
    np.abs(recording_mono)
)


print()
print("=" * 60)
print("RECORDING RESULT")
print("=" * 60)

print("Selected channel :", selected_channel + 1)
print("Samples          :", len(recording_mono))
print("Duration         :", len(recording_mono) / INPUT_RATE)
print("RMS              :", rms)
print("Peak             :", peak)


# ============================================================
# SAVE
# ============================================================

sf.write(
    OUTPUT_FILE,
    recording_mono,
    INPUT_RATE
)

print()
print("Saved:")
print(OUTPUT_FILE)


# ============================================================
# SIGNAL CHECK
# ============================================================

print()

if rms < 0.001:

    print("=" * 60)
    print("WARNING: VERY WEAK MICROPHONE SIGNAL")
    print("=" * 60)

    print("RMS :", rms)

    print()
    print("The microphone is receiving almost no audio.")
    print("Check:")
    print("1. Speaker volume")
    print("2. Distance between speaker and microphone")
    print("3. Windows microphone input level")
    print("4. Microphone permissions")
    print("5. Microphone device 11")

else:

    print("=" * 60)
    print("MICROPHONE SIGNAL DETECTED")
    print("=" * 60)

    print("RMS :", rms)
    print("Peak:", peak)


print()
print("=" * 60)
print("TEST COMPLETE")
print("=" * 60)
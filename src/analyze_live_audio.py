import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os


FILE = r"data\inference\live_test.wav"
OUTPUT = r"outputs\live_audio_analysis"

os.makedirs(OUTPUT, exist_ok=True)


print("=" * 50)
print("SOUND2SEE LIVE AUDIO ANALYSIS")
print("=" * 50)

audio, sr = librosa.load(
    FILE,
    sr=16000,
    mono=True
)

print("\nSample rate :", sr)
print("Samples     :", len(audio))
print("Duration    :", len(audio) / sr)
print("Minimum     :", np.min(audio))
print("Maximum     :", np.max(audio))

rms = np.sqrt(np.mean(audio ** 2))
peak = np.max(np.abs(audio))

print("RMS         :", rms)
print("Peak        :", peak)


# ============================================================
# WAVEFORM
# ============================================================

plt.figure(figsize=(12, 5))

time = np.arange(len(audio)) / sr

plt.plot(time, audio)

plt.title("Sound2See - Live Microphone Waveform")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid(True)

waveform_path = os.path.join(
    OUTPUT,
    "live_waveform.png"
)

plt.savefig(
    waveform_path,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# MEL SPECTROGRAM
# ============================================================

mel = librosa.feature.melspectrogram(
    y=audio,
    sr=sr,
    n_fft=1024,
    hop_length=512,
    n_mels=128
)

mel_db = librosa.power_to_db(
    mel,
    ref=np.max
)

plt.figure(figsize=(12, 6))

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
    "Sound2See - Live Microphone Mel Spectrogram"
)

plt.tight_layout()

spectrogram_path = os.path.join(
    OUTPUT,
    "live_melspectrogram.png"
)

plt.savefig(
    spectrogram_path,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


print("\nFiles saved:")
print("Waveform     :", waveform_path)
print("Spectrogram  :", spectrogram_path)

print("\n========================================")
print("LIVE AUDIO ANALYSIS COMPLETE")
print("========================================")
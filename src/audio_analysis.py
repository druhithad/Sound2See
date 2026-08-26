import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------------------
# 1. Load audio file
# -----------------------------------------

audio_file = "data/samples/test_tone.wav"

audio, sample_rate = sf.read(audio_file)


# -----------------------------------------
# 2. Calculate basic information
# -----------------------------------------

number_of_samples = len(audio)
duration = number_of_samples / sample_rate

minimum_amplitude = np.min(audio)
maximum_amplitude = np.max(audio)
average_amplitude = np.mean(np.abs(audio))


# -----------------------------------------
# 3. Display audio information
# -----------------------------------------

print("========================================")
print("       SOUND2SEE AUDIO ANALYSIS")
print("========================================")

print(f"Audio file       : {audio_file}")
print(f"Sample rate      : {sample_rate} Hz")
print(f"Number of samples: {number_of_samples}")
print(f"Duration         : {duration:.2f} seconds")
print(f"Minimum amplitude: {minimum_amplitude:.4f}")
print(f"Maximum amplitude: {maximum_amplitude:.4f}")
print(f"Average amplitude: {average_amplitude:.4f}")


# -----------------------------------------
# 4. Create time axis
# -----------------------------------------

time = np.arange(number_of_samples) / sample_rate


# -----------------------------------------
# 5. Plot waveform
# -----------------------------------------

plt.figure(figsize=(10, 4))

plt.plot(time, audio)

plt.title("Sound2See - Audio Waveform")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")

plt.tight_layout()


# -----------------------------------------
# 6. Save waveform
# -----------------------------------------

output_file = "outputs/test_tone_waveform.png"

plt.savefig(output_file, dpi=150)

print(f"Waveform saved to : {output_file}")

plt.show()
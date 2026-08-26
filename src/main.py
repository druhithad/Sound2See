import numpy as np
import soundfile as sf

# Audio settings
sample_rate = 16000
duration = 2
frequency = 440

# Create time values
time = np.linspace(
    0,
    duration,
    int(sample_rate * duration),
    endpoint=False
)

# Generate a simple sine-wave audio signal
audio = 0.5 * np.sin(2 * np.pi * frequency * time)

# Save the audio as a WAV file
output_file = "data/samples/test_tone.wav"
sf.write(output_file, audio, sample_rate)

print("================================")
print("        SOUND2SEE AUDIO TEST")
print("================================")
print(f"Sample rate : {sample_rate} Hz")
print(f"Duration    : {duration} seconds")
print(f"Frequency   : {frequency} Hz")
print(f"Audio file  : {output_file}")
print("Audio file created successfully!")
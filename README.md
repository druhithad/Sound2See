# Sound2See

## Intelligent Audio Recognition and Speech-to-Text System

Sound2See is an intelligent audio-processing project designed to recognize and interpret sounds captured through a microphone.

The system accepts real-world audio as input and automatically determines whether the captured audio contains **human speech** or **other environmental/animal sounds**.

If human speech is detected, Sound2See uses a Speech-to-Text model to convert the spoken audio into text.

If human speech is not detected, the system uses a trained Machine Learning audio-classification model to identify the type of sound.

The current system combines:

- Microphone-based audio capture
- Audio preprocessing
- Speech detection
- Speech-to-text conversion
- Audio feature extraction
- Machine-learning-based sound classification
- Confidence estimation
- A unified automatic decision pipeline

The project is being developed as the foundation for a future user-facing interface and mobile application that will use mobile input devices such as the microphone.

---

# 1. Project Objective

The main objective of Sound2See is to create a system capable of understanding different types of audio input instead of treating every sound in the same way.

Traditional audio-classification systems generally attempt to classify every input directly into a predefined sound class.

Sound2See follows a different approach.

The system first asks:

> "Is this human speech?"

If the answer is **Yes**, the system processes the input using Speech-to-Text.

If the answer is **No**, the system processes the input using the trained sound-classification model.

Therefore, the overall decision process is:

```text
Audio Input
     |
     v
Microphone Recording
     |
     v
Speech Detection
     |
     +----------------------+
     |                      |
     | Speech               | No Speech
     v                      v
Speech-to-Text       Sound Classification
     |                      |
     v                      v
Recognized Text       Predicted Sound
     |                      |
     +----------+-----------+
                |
                v
          Sound2See Result
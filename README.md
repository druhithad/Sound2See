Sound2See
AI-Based Real-Time Audio Recognition and Accessibility System

Sound2See is an intelligent audio recognition system designed to identify and interpret sounds from the surrounding environment and present the information in a form that is easier for the user to understand.

The system is being developed with two major input categories:

Human Speech
Environmental / Non-Speech Sounds

For human speech, Sound2See detects speech and converts it into text using a Speech-to-Text model.

For environmental sounds, Sound2See detects that the input is not speech and classifies the sound into one of the trained sound categories using a machine-learning model.

The current implementation has been developed progressively from individual audio-processing modules into a unified end-to-end audio pipeline.

1. Project Objective

The primary objective of Sound2See is to develop an intelligent audio interpretation system capable of:

Capturing audio from a microphone.
Processing real-world microphone input.
Detecting whether the captured audio contains human speech.
Converting detected human speech into text.
Classifying non-speech environmental sounds.
Providing the predicted sound category and confidence.
Eventually presenting the results through a user-friendly interface.
Extending the system into a mobile application that can use the mobile device's microphone and other available input capabilities.

The project is being developed in stages so that every major component can be tested independently before integration.

2. Core Project Idea

The core idea of Sound2See is:

Audio Input
     |
     v
Microphone
     |
     v
Audio Capture
     |
     v
Speech Detection
     |
     +----------------------+
     |                      |
     | Speech               | No Speech
     v                      v
Speech-to-Text        Environmental Sound
     |                      |
     v                      v
Recognized Text       Sound Classification
     |                      |
     +----------+-----------+
                |
                v
          Final Result
                |
                v
       User Interface
                |
                v
        Mobile Application

The important design principle is that speech and environmental sounds follow different processing paths.

The system should not attempt to classify human speech as an animal/environmental sound.

Instead:

                    AUDIO
                      |
                      v
               Speech Detector
                  /        \
                 /          \
            Speech          Non-Speech
              |                 |
              v                 v
        Speech-to-Text     Sound Classifier
              |                 |
              v                 v
        Recognized Text    Sound + Confidence
                 \             /
                  \           /
                   \         /
                    v       v
                    Final Result
3. Project Development Stages

The project is being developed through the following stages.

Stage 1 — Project Definition

Define the complete Sound2See concept, objectives, expected inputs, outputs, and system workflow.

Stage 2 — Dataset Preparation

Prepare and organize audio datasets for environmental sound recognition.

Stage 3 — Audio Preprocessing

Convert and standardize audio into a format suitable for machine-learning processing.

Current processing uses:

16 kHz sample rate
Mono audio
Feature extraction using Librosa
Stage 4 — Feature Extraction

Extract meaningful acoustic features from the audio.

The current feature extraction pipeline includes:

MFCC
MFCC Delta
MFCC Delta-Delta
Spectral Centroid
Spectral Bandwidth
Spectral Rolloff
Zero Crossing Rate
RMS Energy
Chroma
Spectral Contrast
Tonnetz
Temporal MFCC segments
Stage 5 — Machine Learning Model

Train the environmental sound classification model using the extracted features.

The trained model is stored as:

models\sound2see_final.joblib
Stage 6 — Speech Detection

Develop a dedicated speech-detection module to distinguish human speech from non-speech audio.

Implemented in:

src\speech_detector.py
Stage 7 — Speech-to-Text

Integrate Whisper to convert detected human speech into text.

Implemented in:

src\speech_to_text.py
Stage 8 — Live Audio Capture

Develop microphone-based audio recording.

Implemented in:

src\record_live.py

The recorder currently:

Captures microphone input.
Records stereo audio.
Measures both channels.
Selects the stronger channel.
Converts the selected input to mono.
Saves the recording as WAV.
Stage 9 — Unified Audio Pipeline

Integrate all major modules into a single execution pipeline.

Implemented in:

src\sound2see_pipeline.py

The pipeline currently performs:

Microphone
   ↓
Recording
   ↓
Speech Detection
   ↓
Automatic Decision
   ↓
 ┌─────────────────────┐
 │                     │
Speech              Non-Speech
 │                     │
 ↓                     ↓
Whisper          ML Classifier
 │                     │
 ↓                     ↓
Text             Sound Class
 │                     │
 └──────────┬──────────┘
            ↓
       Final Result
Stage 10 — User Interface

The next major development stage is the user interface.

The interface will provide a simple way for users to:

Start audio capture.
View the detected input type.
View recognized speech text.
View environmental sound predictions.
View confidence.
Access the system without using command-line commands.
Stage 11 — Mobile Application

The final application direction is a mobile application.

The mobile application will use the phone's built-in input devices, particularly:

Microphone
Audio processing capabilities
Network connectivity where required

The mobile application will provide the user-facing interface for Sound2See.

4. Current System Architecture

The current implementation can be represented as:

                         SOUND2SEE
                            |
                            v
                    Audio Capture Layer
                            |
                            v
                     Audio WAV File
                            |
                            v
                  Speech Detection Layer
                       /           \
                      /             \
                 Speech           Non-Speech
                    |                  |
                    v                  v
             Whisper STT       Feature Extraction
                    |                  |
                    v                  v
             Recognized Text      ML Model
                                       |
                                       v
                               Sound Prediction
                                       |
                                       v
                                  Confidence
                       \               /
                        \             /
                         \           /
                          v         v
                           Final Result
                                |
                                v
                         User Interface
                                |
                                v
                       Mobile Application
5. Project Directory Structure

The current project is organized approximately as follows:

Sound2See/
│
├── .gitignore
├── README.md
├── requirements.txt
│
├── data/
│   ├── inference/
│   │   └── live_test.wav
│   │
│   ├── processed/
│   │   ├── audio/
│   │   └── combined/
│   │
│   └── ...
│
├── models/
│   └── sound2see_final.joblib
│
├── src/
│   ├── record_live.py
│   ├── speech_detector.py
│   ├── speech_to_text.py
│   ├── predict_audio.py
│   ├── sound2see_pipeline.py
│   └── ...
│
└── .venv/

The .venv directory is the local Python virtual environment and should not be committed to GitHub.

6. Environment Setup

The project is currently being developed on Windows using Python.

A virtual environment is used to keep project dependencies isolated.

Create the environment:

python -m venv .venv

Activate it:

.venv\Scripts\Activate.ps1

Install the project dependencies:

pip install -r requirements.txt
7. Main Dependencies

The current project uses Python libraries including:

numpy
pandas
librosa
scikit-learn
joblib
sounddevice
soundfile
matplotlib
scipy
Whisper

The project also requires FFmpeg for Whisper audio processing.

The FFmpeg executable installed through WinGet was located at:

C:\Users\D LAHARI\AppData\Local\Microsoft\WinGet\Packages\
Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\
ffmpeg-9.0.1-full_build\bin\ffmpeg.exe

Whisper was successfully installed and tested.

8. Audio Capture Module

File:

src\record_live.py

The live recorder captures audio through the microphone.

Current configuration:

Sample Rate : 48000 Hz
Channels    : 2
Duration    : 5 seconds
Device      : 11

The system records both microphone channels and calculates RMS and peak values.

Example:

Channel 1 RMS  : 0.0361
Channel 1 Peak : 0.4002

Channel 2 RMS  : 0.0406
Channel 2 Peak : 0.4737

Selected channel : 2

The channel with the stronger RMS value is selected.

The selected channel is then saved as:

data\inference\live_test.wav

This provides a consistent mono audio file for subsequent processing.

9. Speech Detection Module

File:

src\speech_detector.py

The speech detector analyzes the recorded audio before deciding which processing path should be used.

The current implementation analyzes:

RMS energy
Zero Crossing Rate
Spectral Centroid
Spectral Bandwidth
Frame-level speech-like characteristics

The detector produces a speech-frame ratio.

Example:

Mean RMS              : 0.021299
Mean ZCR              : 0.175147
Mean spectral centroid: 2375.83 Hz
Mean spectral bandwidth: 2128.72 Hz
Speech-like frames    : 55 / 157
Speech frame ratio    : 0.35

The output is then:

RESULT: SPEECH DETECTED

or:

RESULT: NO SPEECH DETECTED

This module is important because it acts as the decision layer between human speech and environmental sound recognition.

10. Speech-to-Text Module

File:

src\speech_to_text.py

The Speech-to-Text module uses OpenAI Whisper.

Current model:

base

The model is loaded using:

whisper.load_model("base")

The audio is then transcribed using Whisper.

Example successful result:

Recognized text:
Hi, this is the first one to speed.

Another test produced:

Recognized text:
Hello, this is a test for you want to speech detection.

The transcription accuracy depends on:

Recording quality
Microphone quality
Background noise
Distance from microphone
Speaker clarity
Whisper model size
11. Environmental Sound Classification

File:

src\predict_audio.py

The environmental sound classifier uses the trained Sound2See machine-learning model.

Model:

models\sound2see_final.joblib

Feature reference:

data\processed\combined\enhanced_features.csv

The classifier expects:

188 features

The prediction system validates that the extracted features match the training feature order before making a prediction.

The system currently supports the following classes:

Bird
Cat
Chicken
Cow
Dog
Donkey
Frog
Lion
Monkey
Sheep
car_horn
crying_baby
door_wood_knock
glass_breaking
siren
12. Feature Extraction

The current classifier extracts 188 features.

The feature groups include:

MFCC

13 MFCC coefficients are extracted.

Temporal segmentation is used to capture changes throughout the audio.

The audio is divided into:

3 temporal segments

For each segment, mean and standard deviation are calculated.

MFCC Delta

First-order temporal changes in MFCC features are calculated.

MFCC Delta-Delta

Second-order temporal changes are calculated.

Spectral Features

The system extracts:

Spectral Centroid
Spectral Bandwidth
Spectral Rolloff
Zero Crossing Rate
RMS Energy

For these features, mean and standard deviation are calculated.

Chroma

12 chroma features are extracted.

Spectral Contrast

Spectral contrast features are extracted using safe frequency-band calculations.

Tonnetz

Tonnetz features are also extracted.

These features collectively provide a representation of the acoustic characteristics of the input sound.

13. Machine Learning Prediction

After feature extraction, the 188 features are arranged in exactly the same order used during training.

The model then generates:

Predicted sound class
Class probabilities
Confidence score

Example:

Predicted sound : Bird
Confidence      : 93.97%

Example class probabilities:

Bird                : 93.97%
glass_breaking      : 3.79%
Frog                : 2.21%
crying_baby         : 0.03%
Chicken             : 0.00%
Monkey              : 0.00%
car_horn            : 0.00%
Sheep               : 0.00%
door_wood_knock     : 0.00%
siren               : 0.00%
Dog                 : 0.00%
Cat                 : 0.00%
Lion                : 0.00%
Cow                 : 0.00%
Donkey              : 0.00%
14. Unified Pipeline

File:

src\sound2see_pipeline.py

This is the main integration point of the current project.

The pipeline combines:

Audio Capture
      ↓
Speech Detection
      ↓
Automatic Decision
      ↓
 ┌───────────────┬─────────────────┐
 │               │                 │
Speech       Non-Speech            │
 │               │                 │
 ↓               ↓                 │
Whisper       ML Classifier        │
 │               │                 │
 ↓               ↓                 │
Text        Sound + Confidence     │
 │               │                 │
 └───────────────┴─────────────────┘
                 ↓
           Final Result
15. Automatic Decision Logic

The current integrated pipeline automatically chooses the processing path.

If speech is detected:

Speech detected.
Automatically selecting Speech-to-Text.

Then:

Audio
 ↓
Whisper
 ↓
Recognized Text

If speech is not detected:

No speech detected.
Automatically selecting Sound Classification.

Then:

Audio
 ↓
Feature Extraction
 ↓
ML Model
 ↓
Sound Class
 ↓
Confidence

This removes the need for the user to manually select whether the input is speech or another sound.

16. Successful Human Speech Test

A real microphone recording was tested using the complete pipeline.

The system produced:

RESULT: SPEECH DETECTED

The pipeline automatically selected:

Speech-to-Text

Whisper then generated:

Recognized text:
Hi, this is the first one to speed.

The final pipeline result was:

Input Type : Human Speech

Recognized Text:
Hi, this is the first one to speed.

This confirms that the current speech-processing path is integrated successfully.

17. Successful Environmental Sound Test

A non-speech recording was also tested.

The speech detector produced:

RESULT: NO SPEECH DETECTED

The pipeline automatically selected:

Sound Classification

The classifier produced:

Predicted sound : Bird
Confidence      : 84.32%

The class probabilities included:

Bird                : 84.32%
glass_breaking      : 14.06%
car_horn            : 1.06%
Frog                : 0.49%
Cat                 : 0.05%

This confirms that the non-speech processing path is also integrated.

18. Current End-to-End Status

At the current stage, the following components are operational:

Component	Status
Project structure	Completed
Dataset processing	Completed
Audio feature extraction	Completed
ML model	Completed
Model prediction	Completed
Live microphone recording	Completed
Speech detection	Completed
Whisper installation	Completed
Speech-to-text	Completed
Automatic speech/non-speech routing	Completed
Unified pipeline	Completed
User interface	Next stage
Mobile application	Future stage
Mobile microphone integration	Future stage
Final user-facing application	Future stage
19. Current Limitations

The current implementation is a working prototype and still has several areas for improvement.

Speech Detection

The current speech detector is feature-based and uses acoustic characteristics to identify speech-like frames.

It is not yet a dedicated neural Voice Activity Detection model.

Therefore, noisy environmental audio may sometimes appear speech-like.

Sound Classification

The classifier can only recognize the classes available in the trained model.

If an unknown sound is provided, the model will still select the closest known class.

A future version should include:

Unknown sound detection
Better confidence handling
Larger datasets
More environmental classes
Whisper

Whisper transcription quality depends on audio quality.

Background noise and unclear speech can reduce transcription accuracy.

User Interface

The current system is command-line based.

The final project requires a proper user interface.

Mobile Application

The mobile version has not yet been implemented.

The current system is being developed first as a working backend/prototype so that it can later be connected to a mobile interface.

20. Planned User Interface

The next major stage is the Sound2See user interface.

The UI should provide a simple workflow:

+--------------------------------+
|           SOUND2SEE            |
|                                |
|       [ Start Listening ]      |
|                                |
|       Listening...             |
|                                |
|   Detected: Human Speech       |
|                                |
|   "Hello, how are you?"        |
|                                |
+--------------------------------+

For environmental sounds:

+--------------------------------+
|           SOUND2SEE            |
|                                |
|       [ Start Listening ]      |
|                                |
|   Detected Sound: Bird         |
|                                |
|   Confidence: 93.97%            |
|                                |
+--------------------------------+

The interface should make the system understandable without requiring technical knowledge.

21. Planned Mobile Application

The final direction of Sound2See is a mobile application.

The mobile application will use the smartphone's built-in microphone as the primary audio input.

The intended architecture is:

                 MOBILE PHONE
                      |
                      v
                 Microphone
                      |
                      v
                Audio Capture
                      |
                      v
               Sound2See Engine
                      |
          +-----------+-----------+
          |                       |
       Speech                 Non-Speech
          |                       |
          v                       v
      Whisper                 Classifier
          |                       |
          v                       v
        Text              Sound + Confidence
          |                       |
          +-----------+-----------+
                      |
                      v
                Mobile UI
                      |
                      v
                  User

The exact mobile technology and deployment architecture will be finalized during the mobile-development stage.

22. Development Philosophy

Sound2See is being developed incrementally.

Instead of immediately creating the complete application, each major component is developed and tested independently.

The development sequence is:

Concept
  ↓
Dataset
  ↓
Preprocessing
  ↓
Feature Extraction
  ↓
ML Model
  ↓
Prediction
  ↓
Speech Detection
  ↓
Speech-to-Text
  ↓
Live Recording
  ↓
Unified Pipeline
  ↓
User Interface
  ↓
Mobile Application
  ↓
Final Integrated System

This approach makes it easier to identify errors and verify every stage independently.

23. Testing Strategy

Each module should be tested independently before being integrated.

Audio Recorder Test
python src\record_live.py
Speech Detector Test
python src\speech_detector.py "data\inference\live_test.wav"
Speech-to-Text Test
python src\speech_to_text.py "data\inference\live_test.wav"
Sound Classifier Test
python src\predict_audio.py "data\inference\live_test.wav"
Complete Pipeline Test
python src\sound2see_pipeline.py

The final pipeline test is currently the most important demonstration command.

24. Recommended Review Demonstration

For project reviews, the complete system can be demonstrated using:

python src\sound2see_pipeline.py

Then provide a human speech input.

Expected flow:

Microphone
   ↓
Speech Detection
   ↓
Speech Detected
   ↓
Whisper
   ↓
Recognized Text

Then repeat the test using an environmental sound.

Expected flow:

Microphone
   ↓
Speech Detection
   ↓
No Speech Detected
   ↓
Sound Classifier
   ↓
Predicted Sound
   ↓
Confidence

This demonstrates the core intelligence of Sound2See.

25. Future Development Roadmap

The remaining development roadmap is:

Milestone 1 — Backend Prototype

Current milestone.

Completed:

Audio capture
Speech detection
Speech-to-text
Environmental sound classification
Automatic routing
Unified pipeline
Milestone 2 — Backend Improvement

Improve:

Speech detection accuracy
Noise handling
Confidence thresholds
Unknown sound detection
Processing speed
Error handling
Milestone 3 — User Interface

Develop:

Start/stop listening
Listening indicator
Speech result screen
Sound result screen
Confidence display
History/results interface
Milestone 4 — Mobile Application

Develop:

Mobile audio capture
Mobile UI
Backend/model integration
Real-time or near-real-time processing
Result display
Milestone 5 — Final Integration

Integrate:

Mobile Input
      ↓
Sound2See Processing
      ↓
Speech / Sound Decision
      ↓
Speech-to-Text / Classification
      ↓
Mobile User Interface
Milestone 6 — Testing and Evaluation

Evaluate:

Speech detection accuracy
Sound classification accuracy
Speech recognition quality
Response time
Different noise environments
Different microphone conditions
Different distances from the sound source
26. Final Project Vision

The final Sound2See system is intended to provide a simple interface through which a user can listen to the surrounding environment and receive meaningful information about what is being heard.

The complete system will combine:

Artificial Intelligence
        +
Audio Processing
        +
Machine Learning
        +
Speech Recognition
        +
Mobile Technology
        +
User Interface

The final goal is to transform raw environmental audio into understandable information.

The user should not need to understand the underlying machine-learning or audio-processing operations.

The user simply interacts with the application:

              USER
                |
                v
        Press "Listen"
                |
                v
          Mobile Microphone
                |
                v
          Sound2See Engine
                |
        +-------+-------+
        |               |
      Speech         Sound
        |               |
        v               v
   Speech-to-Text   Classification
        |               |
        v               v
      Text        Sound + Confidence
        |               |
        +-------+-------+
                |
                v
          Mobile Interface
                |
                v
              USER
27. Current Project State

Sound2See is currently at the unified backend prototype stage.

The most important achievement at this stage is that the system can now take real microphone input and automatically decide whether to process it as human speech or as an environmental sound.

For speech:

Microphone
→ Speech Detection
→ Whisper
→ Text

For environmental sound:

Microphone
→ Speech Detection
→ Feature Extraction
→ ML Classifier
→ Sound + Confidence

The next development should therefore not change the core idea or replace this architecture.

The next stage should build the user interface on top of this working pipeline, followed by the mobile application using the mobile device's microphone.

Important project rule

The core Sound2See concept should remain:

Capture audio → distinguish human speech from non-speech → transcribe speech OR classify environmental sound → present the result to the user through the final interface/mobile application.

Future development should extend this pipeline rather than introduce unrelated functionality.
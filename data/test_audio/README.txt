Sound2See Stage 4.1 test files

01_silence.wav             Expected: NO SPEECH
02_440hz_tone.wav          Expected: NO SPEECH
03_background_noise.wav    Expected: NO SPEECH
04_non_speech_bursts.wav   Expected: NO SPEECH
05_human_speech.wav        Expected: SPEECH
06_speech_with_noise.wav   Expected: SPEECH

Local TTS generation: failed: SetVoiceByName failed with unknown return code -1 for voice: gmw/en

If the two speech files were not generated, record a real 5-second voice sample
later and use it as the positive speech test.

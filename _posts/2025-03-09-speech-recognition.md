---
title: Speech Recognition
author: Sudhir Gupta
date: 2025-03-09 18:10:01 +0300
category: Speech Processing
layout: post
---

Speech recognition (also known as automatic speech recognition or ASR) converts spoken language into written text.

#### Hands-on Example: Transcribing Speech

```python
from transformers import pipeline
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
import requests
from io import BytesIO

# Initialize the automatic speech recognition pipeline
transcriber = pipeline("automatic-speech-recognition")

# Download an audio sample
audio_url = "https://github.com/librosa/librosa/raw/main/tests/data/choice.wav"
response = requests.get(audio_url)
with open("speech_sample.wav", "wb") as f:
    f.write(response.content)

# Load the audio
audio, sr = librosa.load("speech_sample.wav", sr=16000)

# Visualize the waveform
plt.figure(figsize=(10, 4))
plt.plot(np.linspace(0, len(audio)/sr, len(audio)), audio)
plt.title("Audio Waveform")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()

# Transcribe audio
result = transcriber("speech_sample.wav")
print(f"Transcription: {result['text']}")
```

The speech recognition pipeline converts audio recordings into text, using pre-trained models that have been fine-tuned on large datasets of speech.

#### Try It Yourself:
1. Record your own voice using a tool like Audacity and transcribe it.
2. Try transcribing audio in different languages using models like `facebook/wav2vec2-large-960h-lv60-self`.
3. Experiment with audio that has background noise or multiple speakers to test model robustness.

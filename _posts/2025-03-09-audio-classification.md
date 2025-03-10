---
title: Audio Classification
author: Sudhir Gupta
date: 2025-03-09 18:10:02 +0300
category: Speech Processing
layout: post
---

Audio classification identifies sounds or categorizes audio clips based on their content.

#### Hands-on Example: Classifying Audio

```python
from transformers import pipeline
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
import requests

# Initialize the audio classification pipeline
audio_classifier = pipeline("audio-classification")

# Download audio samples
audio_urls = {
    "dog": "https://github.com/librosa/librosa/raw/main/tests/data/choice.wav",  # Using as placeholder
    "siren": "https://github.com/librosa/librosa/raw/main/tests/data/choice.wav",  # Using as placeholder
    "piano": "https://github.com/librosa/librosa/raw/main/tests/data/choice.wav"   # Using as placeholder
}

# Process each audio file
for label, url in audio_urls.items():
    # Download and save
    response = requests.get(url)
    filename = f"{label}_sound.wav"
    with open(filename, "wb") as f:
        f.write(response.content)
    
    # Classify audio
    results = audio_classifier(filename)
    
    # Display top 3 predictions
    print(f"Audio: {filename}")
    for result in results[:3]:
        print(f"• {result['label']}: {result['score']:.4f}")
    print("-" * 50)
    
    # Visualize audio waveform
    audio, sr = librosa.load(filename, sr=16000)
    plt.figure(figsize=(10, 4))
    plt.plot(np.linspace(0, len(audio)/sr, len(audio)), audio)
    plt.title(f"Waveform for {label} sound")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()
```

The audio classification pipeline identifies the type of sound in an audio clip, useful for applications like environmental sound recognition and content moderation.

#### Try It Yourself:
1. Classify different types of music or environmental sounds.
2. Try the `facebook/wav2vec2-base-960h` model for potentially better performance.
3. Create mixed audio samples and see how the classifier performs on more complex inputs.
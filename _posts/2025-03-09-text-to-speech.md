---
title: Text-to-Speech
author: Sudhir Gupta
date: 2025-03-09 18:10:03 +0300
category: Speech Processing
layout: post
---
Text-to-speech (TTS) converts written text into spoken words, enabling applications like screen readers and voice assistants.

#### Hands-on Example: Generating Speech

```python
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import torch
import soundfile as sf
from datasets import load_dataset
import IPython.display as ipd

# Load processor, model and vocoder
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

# Get speaker embeddings from a dataset
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

# Text to synthesize
texts = [
    "Welcome to Hugging Face! This is a demonstration of text to speech synthesis.",
    "Artificial intelligence is transforming how we interact with technology.",
    "Machine learning models can now generate realistic human speech."
]

# Synthesize speech for each text
for i, text in enumerate(texts):
    # Process text
    inputs = processor(text=text, return_tensors="pt")
    
    # Generate speech
    speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
    
    # Save audio
    output_file = f"synthesized_speech_{i+1}.wav"
    sf.write(output_file, speech.numpy(), samplerate=16000)
    
    print(f"Generated speech for: '{text}'")
    # In a notebook, you could play the audio with:
    # ipd.display(ipd.Audio(output_file))
```

The SpeechT5 model converts text to natural-sounding speech, demonstrating how Hugging Face models can be used for audio synthesis.

#### Try It Yourself:
1. Generate speech in different styles by trying different speaker embeddings.
2. Experiment with text that includes questions, exclamations, or different emotions.
3. Try the `facebook/fastspeech2-en-ljspeech` model for comparison.

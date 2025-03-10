---
title: Hugging Face Spaces
author: Sudhir Gupta
date: 2025-03-09 18:30:02 +0300
category: Hugging Face Ecosystem Tools
layout: post
---

Spaces allows you to create and share demos of your machine learning models with a simple web interface.

#### Hands-on Example: Creating a Gradio Demo

```python
import gradio as gr
from transformers import pipeline

# Initialize model
classifier = pipeline("sentiment-analysis")

# Define function for the interface
def analyze_sentiment(text):
    result = classifier(text)[0]
    return f"{result['label']} (Confidence: {result['score']:.4f})"

# Create Gradio interface
demo = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(lines=4, placeholder="Enter text to analyze..."),
    outputs="text",
    title="Sentiment Analysis Demo",
    description="This demo uses a pre-trained model to analyze the sentiment of text.",
    examples=[
        "I absolutely loved this movie! The acting was superb.",
        "The service at this restaurant was terrible and the food was bland.",
        "The weather today is okay, not great but not terrible either."
    ]
)

# Launch the interface
demo.launch()
```

#### Try It Yourself:
1. Create demos for different AI tasks like image classification, translation, or text generation.
2. Customize the interface with themes, additional inputs/outputs, and more advanced components.
3. Deploy your demo to Hugging Face Spaces to share it with the community.
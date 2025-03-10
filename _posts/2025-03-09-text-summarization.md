---
title: Text Summarization
author: Sudhir Gupta
date: 2025-03-09 18:00:04 +0300
category: Natural Language Processing
layout: post
---

Summarization condenses a longer text into a shorter version while preserving key information.

#### Hands-on Example: Abstractive Summarization

```python
from transformers import pipeline

# Initialize the summarization pipeline
summarizer = pipeline("summarization")

# Text to summarize
article = """
Artificial intelligence (AI) is revolutionizing industries worldwide. From healthcare to finance, AI systems are improving 
efficiency and enabling new possibilities. In healthcare, AI assists in diagnosing diseases from medical images with accuracy 
rivaling human doctors. Financial institutions use AI for fraud detection, risk assessment, and algorithmic trading. 
Transportation is being transformed through autonomous vehicles which promise to reduce accidents and congestion. 
Manufacturing benefits from predictive maintenance and quality control powered by AI. Despite these advances, concerns 
about job displacement, algorithm bias, and privacy remain. Researchers and policymakers are working to address these 
challenges while maximizing AI's positive impact. The future of AI involves more sophisticated models, enhanced human-AI 
collaboration, and wider deployment across industries.
"""

# Generate summary
summary = summarizer(article, max_length=100, min_length=30, do_sample=False)

print("Original text:")
print(article)
print("\nSummary:")
print(summary[0]['summary_text'])
```

The summarization pipeline creates a concise version of the input text, focusing on the most important points.

#### Try It Yourself:
1. Summarize a news article or research paper abstract.
2. Experiment with different `max_length` and `min_length` values to control summary length.
3. Try different models like `facebook/bart-large-cnn` or `google/pegasus-xsum` for different summarization styles.
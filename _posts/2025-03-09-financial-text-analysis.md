---
title: Financial Text Analysis
author: Sudhir Gupta
date: 2025-03-09 18:25:02 +0300
category: Domain-Specific Applications
layout: post
---

Models specialized for financial sentiment and document analysis:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F

# Load FinBERT for financial sentiment analysis
model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Analyze financial sentiment
texts = [
    "The company reported a 50% increase in quarterly profits.",
    "The stock plummeted following the earnings miss.",
    "Analysts remain neutral on the company's growth prospects."
]

for text in texts:
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = F.softmax(outputs.logits, dim=1)
    labels = ["negative", "neutral", "positive"]
    sentiment = labels[probs.argmax().item()]
    confidence = probs.max().item()
    print(f"Text: {text}")
    print(f"Sentiment: {sentiment} (confidence: {confidence:.4f})")
    print("-" * 50)
```
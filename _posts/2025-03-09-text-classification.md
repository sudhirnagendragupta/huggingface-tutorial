---
title: Text Classification
author: Sudhir Gupta
date: 2025-03-09 18:00:01 +0300
category: Natural Language Processing
layout: post
---

Text classification is the task of assigning predefined categories to text documents. Common applications include sentiment analysis, topic classification, and language detection.

#### Hands-on Example: Sentiment Analysis

```python
from transformers import pipeline

# Initialize the sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

# Analyze text
texts = [
    "I absolutely loved this movie! The acting was superb.",
    "The service at this restaurant was terrible and the food was bland.",
    "The weather today is okay, not great but not terrible either."
]

# Get results
results = sentiment_analyzer(texts)

# Display results
for text, result in zip(texts, results):
    print(f"Text: {text}")
    print(f"Sentiment: {result['label']}, Score: {result['score']:.4f}")
    print("-" * 50)
```

When you run this code, you'll see the sentiment (POSITIVE or NEGATIVE) and the confidence score for each text. The pipeline abstracts away many of the complexities, using a pre-trained model (by default, DistilBERT fine-tuned on a sentiment analysis dataset).

#### Try It Yourself:
1. Use the sentiment analysis pipeline on some reviews of your favorite products or movies.
2. What happens if you analyze more ambiguous or neutral text?
3. Try changing the model by specifying a different one: `sentiment_analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")` - this model provides more fine-grained sentiment (1-5 stars).
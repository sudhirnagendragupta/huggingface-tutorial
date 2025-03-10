---
title: Datasets Library
author: Sudhir Gupta
date: 2025-03-09 18:30:01 +0300
category: Hugging Face Ecosystem Tools
layout: post
---

The datasets library provides a unified interface for accessing and processing datasets for machine learning.

#### Hands-on Example: Working with the Datasets Library

```python
from datasets import load_dataset, load_metric
import matplotlib.pyplot as plt
import numpy as np

# Load a dataset
dataset = load_dataset("glue", "sst2")
print(f"Dataset structure: {dataset}")

# Explore dataset splits
for split in dataset:
    print(f"Split: {split}, Number of examples: {len(dataset[split])}")

# Look at sample data
print("\nSample examples from the training set:")
for i, example in enumerate(dataset["train"][:5]):
    print(f"Example {i+1}:")
    print(f"  Text: {example['sentence']}")
    print(f"  Label: {example['label']} ({dataset['train'].features['label'].names[example['label']]})")

# Dataset statistics
sentence_lengths = [len(example["sentence"].split()) for example in dataset["train"]]
plt.figure(figsize=(10, 6))
plt.hist(sentence_lengths, bins=50)
plt.title("Distribution of Sentence Lengths in SST-2 Training Set")
plt.xlabel("Number of Words")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

print(f"Average sentence length: {np.mean(sentence_lengths):.2f} words")
print(f"Median sentence length: {np.median(sentence_lengths):.2f} words")
print(f"Min sentence length: {min(sentence_lengths)} words")
print(f"Max sentence length: {max(sentence_lengths)} words")

# Class distribution
labels = [example["label"] for example in dataset["train"]]
label_counts = np.bincount(labels)
label_names = dataset["train"].features["label"].names

plt.figure(figsize=(8, 6))
plt.bar(label_names, label_counts)
plt.title("Class Distribution in SST-2 Training Set")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

for i, name in enumerate(label_names):
    print(f"Class {name}: {label_counts[i]} examples ({label_counts[i]/len(labels)*100:.2f}%)")
```

#### Try It Yourself:
1. Explore different datasets for various tasks (e.g., `imdb` for sentiment analysis, `squad` for question answering).
2. Create a custom dataset from your own data and share it on the Hugging Face Hub.
3. Use dataset transformations like filtering, mapping, and shuffling to preprocess data for training.

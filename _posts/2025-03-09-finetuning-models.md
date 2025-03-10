---
title: Finetuning Models
author: Sudhir Gupta
date: 2025-03-09 18:20:01 +0300
category: Advanced Topics
layout: post
---

Fine-tuning allows you to adapt pre-trained models to your specific data and tasks, often leading to better performance.

#### Hands-on Example: Fine-tuning a Text Classification Model

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch

# Load a small dataset (IMDB reviews subset in this example)
dataset = load_dataset("imdb", split="train[:1000]")

# Split into train and validation
dataset = dataset.train_test_split(test_size=0.2)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# Load pre-trained model and tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Tokenize data
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_eval = eval_dataset.map(tokenize_function, batched=True)

# Define metrics function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch",
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    compute_metrics=compute_metrics,
)

# Fine-tune the model
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()
print("Evaluation results:", eval_results)

# Save fine-tuned model and tokenizer
model.save_pretrained("./fine-tuned-model")
tokenizer.save_pretrained("./fine-tuned-model")

# Test the fine-tuned model on a new example
test_text = "This movie was absolutely fantastic! I loved every minute of it."
inputs = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True)
outputs = model(**inputs)
predicted_class = torch.argmax(outputs.logits, dim=1).item()
print(f"Test text: {test_text}")
print(f"Predicted class: {'Positive' if predicted_class == 1 else 'Negative'}")
```

This example demonstrates a basic fine-tuning workflow for a text classification model, including data preparation, training, evaluation, and inference.

#### Try It Yourself:
1. Try fine-tuning on your own dataset by creating a custom Dataset from a CSV or JSON file.
2. Experiment with different model architectures like BERT, RoBERTa, or DistilBERT.
3. Try different hyperparameters like learning rate, batch size, and number of epochs to optimize performance.
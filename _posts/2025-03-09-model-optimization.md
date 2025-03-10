---
title: Model Optimization Techniques
author: Sudhir Gupta
date: 2025-03-09 18:20:02 +0300
category: Advanced Topics
layout: post
---

Optimization techniques help make models more efficient in terms of size, speed, and memory usage.

#### Hands-on Example: Quantizing a Model

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from torch.quantization import quantize_dynamic
import time

# Load a pre-trained model
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Prepare sample input
sample_text = "I really enjoyed this movie. The acting was superb and the plot was engaging."
inputs = tokenizer(sample_text, return_tensors="pt")

# Function to measure inference time
def measure_inference_time(model, inputs, num_runs=100):
    start_time = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            outputs = model(**inputs)
    end_time = time.time()
    return (end_time - start_time) / num_runs

# Measure original model performance
original_size = sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024  # Size in MB
original_time = measure_inference_time(model, inputs)

print(f"Original model size: {original_size:.2f} MB")
print(f"Original model average inference time: {original_time*1000:.2f} ms")

# Quantize the model
quantized_model = quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Measure quantized model performance
quantized_size = sum(p.numel() for p in quantized_model.parameters()) * 1 / 1024 / 1024  # Approximation
quantized_time = measure_inference_time(quantized_model, inputs)

print(f"Quantized model size: {quantized_size:.2f} MB")
print(f"Quantized model average inference time: {quantized_time*1000:.2f} ms")
print(f"Size reduction: {(1 - quantized_size/original_size)*100:.2f}%")
print(f"Speed improvement: {(1 - quantized_time/original_time)*100:.2f}%")

# Check accuracy
with torch.no_grad():
    original_output = model(**inputs).logits
    quantized_output = quantized_model(**inputs).logits

original_prediction = torch.argmax(original_output, dim=1).item()
quantized_prediction = torch.argmax(quantized_output, dim=1).item()

print(f"Original model prediction: {original_prediction}")
print(f"Quantized model prediction: {quantized_prediction}")
print(f"Predictions match: {original_prediction == quantized_prediction}")
```

This example demonstrates dynamic quantization, which reduces model size and improves inference speed with minimal impact on accuracy.

#### Try It Yourself:
1. Try quantization with different models and tasks to see how it affects performance.
2. Experiment with other optimization techniques like pruning and knowledge distillation.
3. Use the `optimum` library from Hugging Face for more advanced optimization techniques.
---
title: CLIP - Bridging Vision and Language
author: Sudhir Gupta
date: 2025-03-09 18:15:03 +0300
category: Multimodal Applications
layout: post
---

CLIP (Contrastive Language-Image Pre-training) is a powerful model that understands both images and text in a shared embedding space, enabling various multimodal tasks.

#### Hands-on Example: Image-Text Similarity with CLIP

```python
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
from io import BytesIO
import torch
import matplotlib.pyplot as plt
import numpy as np

# Load CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load an image
image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"  # Cat image
response = requests.get(image_url)
image = Image.open(BytesIO(response.content))

# Display the image
plt.figure(figsize=(8, 8))
plt.imshow(image)
plt.axis('off')
plt.title("Query Image")
plt.show()

# Define text descriptions to compare against
texts = [
    "a photo of a cat",
    "a photo of a dog",
    "a photo of a pizza",
    "a photo of a sunset",
    "a drawing of a cat",
    "a close-up photo of a cat",
    "a black and white photo of a cat",
]

# Compute image-text similarity
inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)

with torch.no_grad():
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)

# Print results
print("Image-text similarity scores:")
for i, text in enumerate(texts):
    print(f"'{text}': {probs[0][i].item():.4f}")

# Visualize results
plt.figure(figsize=(10, 6))
plt.bar(range(len(texts)), probs[0].numpy())
plt.xticks(range(len(texts)), [t[:15] + "..." if len(t) > 15 else t for t in texts], rotation=45, ha="right")
plt.title("Image-Text Similarity Scores")
plt.tight_layout()
plt.show()
```

CLIP can be used for various tasks such as zero-shot image classification, image retrieval, and visual search, all without task-specific fine-tuning.

#### Try It Yourself:
1. Use CLIP for zero-shot image classification by comparing an image against a list of category descriptions.
2. Create an image retrieval system that finds the most relevant image for a text query.
3. Experiment with CLIP's understanding of visual concepts and relationships.
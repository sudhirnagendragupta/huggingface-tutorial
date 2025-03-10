---
title: Image Classification
author: Sudhir Gupta
date: 2025-03-09 18:05:01 +0300
category: Computer Vision
layout: post
---
Image classification assigns labels to entire images, identifying what they primarily depict.

#### Hands-on Example: Classifying Images

```python
from transformers import pipeline
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt

# Initialize the image classification pipeline
image_classifier = pipeline("image-classification")

# Load images from URLs
image_urls = [
    "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d9/Collage_of_Nine_Dogs.jpg/800px-Collage_of_Nine_Dogs.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/767px-Cat_November_2010-1a.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/9/91/F-15E_Strike_Eagle.jpg/800px-F-15E_Strike_Eagle.jpg"
]

# Classify each image
for url in image_urls:
    # Load image
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    
    # Classify
    results = image_classifier(image)
    
    # Display top 3 predictions
    print(f"Image: {url.split('/')[-1]}")
    for result in results[:3]:
        print(f"• {result['label']}: {result['score']:.4f}")
    print("-" * 50)
```

The image classification pipeline predicts what's depicted in the image, providing labels and confidence scores.

#### Try It Yourself:
1. Classify images from your own collection by loading them from disk: `image = Image.open("path/to/image.jpg")`.
2. Try different pre-trained models like `google/vit-base-patch16-224` or `microsoft/resnet-50` by specifying them in the pipeline.
3. See how the model performs on ambiguous images or images with multiple subjects.
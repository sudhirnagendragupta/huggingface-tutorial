---
title: Image Captioning
author: Sudhir Gupta
date: 2025-03-09 18:15:02 +0300
category: Multimodal Applications
layout: post
---

Image captioning generates descriptive text for images, useful for accessibility and content indexing.

#### Hands-on Example: Generating Captions for Images

```python
from transformers import pipeline
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt

# Initialize the image-to-text pipeline
image_captioner = pipeline("image-to-text")

# Load images from URLs
image_urls = [
    "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f3/San_Francisco_skyline_at_night_from_Pier_7.jpg/800px-San_Francisco_skyline_at_night_from_Pier_7.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d5/Giraffe_at_Kruger_National_Park%2C_South_Africa_%28square_crop%29.jpg/800px-Giraffe_at_Kruger_National_Park%2C_South_Africa_%28square_crop%29.jpg"
]

# Generate captions for each image
for i, url in enumerate(image_urls):
    # Load image
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    
    # Display image
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Image {i+1}")
    plt.show()
    
    # Generate caption
    captions = image_captioner(image)
    
    print(f"Generated captions for Image {i+1}:")
    for caption in captions:
        print(f"• {caption['generated_text']}")
    print("-" * 50)
```

The image captioning pipeline generates descriptive text for images, demonstrating how vision and language models can be combined.

#### Try It Yourself:
1. Generate captions for personal photos or artwork to see how the model interprets different visual styles.
2. Try different models like `nlpconnect/vit-gpt2-image-captioning` for comparison.
3. Test the captioning on abstract or ambiguous images to see how the model handles them.

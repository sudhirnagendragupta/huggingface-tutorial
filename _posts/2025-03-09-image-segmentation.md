---
title: Image Segmentation
author: Sudhir Gupta
date: 2025-03-09 18:05:03 +0300
category: Computer Vision
layout: post
---

Image segmentation assigns a class to each pixel in the image, providing more detailed information than bounding boxes.

#### Hands-on Example: Semantic Segmentation

```python
from transformers import pipeline
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np

# Initialize the image segmentation pipeline
segmenter = pipeline("image-segmentation")

# Load an image
image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/6/68/Busy_street_in_Delhi.jpg/800px-Busy_street_in_Delhi.jpg"
response = requests.get(image_url)
image = Image.open(BytesIO(response.content))

# Segment the image
results = segmenter(image)

# Display original image
plt.figure(figsize=(15, 8))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis('off')

# Create segmentation visualization
# Get unique masks for overlay
plt.subplot(1, 2, 2)
plt.imshow(image)
plt.title("Segmentation Overlay")
plt.axis('off')

# Generate random colors for each segment
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(results), 3)) / 255.0

# Add semi-transparent overlays
for i, segment in enumerate(results):
    mask = segment['mask'].convert('L')
    mask_array = np.array(mask)
    
    # Create colored mask
    colored_mask = np.zeros((mask_array.shape[0], mask_array.shape[1], 4))
    colored_mask[mask_array > 0] = [*colors[i], 0.4]  # Add alpha channel
    
    plt.imshow(colored_mask)
    
# Print detected segments
print("Segmented parts:")
for segment in results:
    print(f"• {segment['label']} (score: {segment.get('score', 'N/A')})")

plt.tight_layout()
plt.show()
```

The image segmentation pipeline identifies regions in the image and classifies each pixel, creating a detailed map of the image content.

#### Try It Yourself:
1. Apply segmentation to landscape images to see how it identifies terrain features.
2. Try different models like `facebook/detr-resnet-50-panoptic` for panoptic segmentation (which distinguishes individual instances).
3. Experiment with different visualization techniques for the segmentation masks.
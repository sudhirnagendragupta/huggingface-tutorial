---
title: Object Detection
author: Sudhir Gupta
date: 2025-03-09 18:05:02 +0300
category: Computer Vision
layout: post
---
Object detection identifies and localizes multiple objects within an image, providing bounding boxes around them.

#### Hands-on Example: Detecting Objects

```python
from transformers import pipeline
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO
import matplotlib.pyplot as plt

# Initialize the object detection pipeline
object_detector = pipeline("object-detection")

# Load an image
image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/f/fb/Hotdog_with_mustard.png/800px-Hotdog_with_mustard.png"
response = requests.get(image_url)
image = Image.open(BytesIO(response.content))

# Detect objects
results = object_detector(image)

# Draw bounding boxes
draw = ImageDraw.Draw(image)
for result in results:
    box = result["box"]
    label = f"{result['label']}: {result['score']:.2f}"
    
    # Draw rectangle
    draw.rectangle([(box["xmin"], box["ymin"]), (box["xmax"], box["ymax"])], 
                   outline="red", width=3)
    
    # Draw label
    draw.text((box["xmin"], box["ymin"] - 10), label, fill="red")

# Display image with detections
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.axis('off')
plt.show()

# Also print detection results
print("Detected objects:")
for result in results:
    print(f"• {result['label']} with confidence {result['score']:.4f} at position {result['box']}")
```

The object detection pipeline identifies objects in the image and provides their bounding box coordinates and confidence scores.

#### Try It Yourself:
1. Test object detection on images with multiple objects, like street scenes or group photos.
2. Try different models like `facebook/detr-resnet-50` for potentially better performance.
3. Experiment with detection threshold by filtering results: `[r for r in results if r['score'] > 0.7]`.
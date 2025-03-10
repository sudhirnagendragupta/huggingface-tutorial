---
title: Visual Question Answering
author: Sudhir Gupta
date: 2025-03-09 18:15:01 +0300
category: Multimodal Applications
layout: post
---
Visual Question Answering (VQA) answers questions about images, combining computer vision and natural language processing.

#### Hands-on Example: Answering Questions About Images

```python
from transformers import pipeline
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt

# Initialize the visual question answering pipeline
vqa = pipeline("visual-question-answering")

# Load an image
image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/800px-Cute_dog.jpg"
response = requests.get(image_url)
image = Image.open(BytesIO(response.content))

# Questions about the image
questions = [
    "What animal is in the image?",
    "What color is the dog?",
    "Is the dog inside or outside?",
    "Does the dog look happy?"
]

# Display the image
plt.figure(figsize=(8, 8))
plt.imshow(image)
plt.axis('off')
plt.title("Query Image")
plt.show()

# Answer each question
print("Visual Question Answering:")
for question in questions:
    result = vqa(image=image, question=question)
    print(f"Q: {question}")
    print(f"A: {result['answer']} (Score: {result['score']:.4f})")
    print("-" * 50)
```

The visual question answering pipeline combines image understanding with language comprehension to answer questions about visual content.

#### Try It Yourself:
1. Test VQA on complex scenes with multiple objects and ask questions about relationships between objects.
2. Try asking more abstract questions about mood, style, or aesthetic qualities.
3. Experiment with ambiguous questions to see how the model handles uncertainty.

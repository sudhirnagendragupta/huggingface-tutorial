---
title: Image Generation
author: Sudhir Gupta
date: 2025-03-09 18:05:04 +0300
category: Computer Vision
layout: post
---
Image generation creates new images based on text descriptions, enabling creative applications and content creation.

#### Hands-on Example: Text-to-Image Generation with Diffusers

```python
from diffusers import StableDiffusionPipeline
import torch
import matplotlib.pyplot as plt

# Initialize the Stable Diffusion pipeline (requires about 7GB of VRAM)
# This will download a large model (~4GB) on first run
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", 
                                               torch_dtype=torch.float16)

# Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(device)

# Generate images from prompts
prompts = [
    "A serene landscape with mountains and a lake at sunset",
    "A futuristic city with flying cars and tall skyscrapers",
    "A cute robot playing with a kitten in a garden"
]

# Create figure for displaying results
plt.figure(figsize=(15, 5 * len(prompts)))

for i, prompt in enumerate(prompts):
    print(f"Generating: {prompt}")
    image = pipe(prompt).images[0]
    
    # Display image
    plt.subplot(len(prompts), 1, i+1)
    plt.imshow(image)
    plt.title(prompt)
    plt.axis('off')

plt.tight_layout()
plt.show()
```

The Stable Diffusion pipeline generates images based on text descriptions, demonstrating the power of text-to-image models.

#### Try It Yourself:
1. Create detailed prompts that specify style, content, and mood for more controlled generation.
2. Experiment with different models like `CompVis/stable-diffusion-v1-4` or `stabilityai/stable-diffusion-2-1`.
3. Try adjusting parameters like `guidance_scale` and `num_inference_steps` to control the generation process.
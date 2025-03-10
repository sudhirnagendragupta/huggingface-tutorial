---
title: Text Generation
author: Sudhir Gupta
date: 2025-03-09 18:00:03 +0300
category: Natural Language Processing
layout: post
---

Text generation involves creating coherent text based on a prompt. This is useful for applications like chatbots, content creation, and creative writing assistance.

#### Hands-on Example: Text Completion

```python
from transformers import pipeline

# Initialize the text generation pipeline
generator = pipeline("text-generation", model="gpt2")

# Generate text from prompts
prompts = [
    "In the distant future, artificial intelligence",
    "The best way to learn programming is",
    "Climate change has affected many regions, leading to"
]

# Generate and display completions
for prompt in prompts:
    completions = generator(prompt, max_length=50, num_return_sequences=2)
    print(f"Prompt: {prompt}")
    
    for i, completion in enumerate(completions):
        print(f"Completion {i+1}: {completion['generated_text']}")
    
    print("-" * 50)
```

The text generation pipeline continues text from the given prompts, producing creative and contextually relevant completions. The default model is GPT-2, but you can specify other models as needed.

#### Try It Yourself:
1. Experiment with different prompts related to your interests.
2. Try adjusting parameters like `max_length`, `num_return_sequences`, and `temperature` (controls randomness).
3. Use different models like `EleutherAI/gpt-neo-1.3B` for potentially better completions.
---
title: Question Answering
author: Sudhir Gupta
date: 2025-03-09 18:00:02 +0300
category: Natural Language Processing
layout: post
---
Question answering systems respond to queries in natural language with relevant answers, often extracted from a given context.

#### Hands-on Example: Extractive Question Answering

```python
from transformers import pipeline

# Initialize the question answering pipeline
qa_pipeline = pipeline("question-answering")

# Context and question
context = """
Hugging Face is an AI community and platform that provides tools to build, train, and deploy machine learning models. 
Founded in 2016 by Clément Delangue, Julien Chaumond, and Thomas Wolf, the company has grown to become a leading provider 
of open-source tools in natural language processing. Their Transformers library has over 50,000 GitHub stars and supports 
frameworks like PyTorch, TensorFlow, and JAX. In 2021, Hugging Face raised $100 million in a Series C funding round.
"""

questions = [
    "Who founded Hugging Face?",
    "What is Hugging Face?",
    "How much funding did Hugging Face raise in 2021?"
]

# Get answers
for question in questions:
    result = qa_pipeline(question=question, context=context)
    print(f"Question: {question}")
    print(f"Answer: {result['answer']}")
    print(f"Confidence: {result['score']:.4f}")
    print("-" * 50)
```

The question answering pipeline identifies spans in the context that answer the given questions, along with a confidence score.

#### Try It Yourself:
1. Use a longer passage from a Wikipedia article as context and ask questions about it.
2. Try questions whose answers aren't directly in the text. How does the model respond?
3. Experiment with different models, such as `deepset/roberta-base-squad2` for improved performance.

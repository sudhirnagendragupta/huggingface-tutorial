---
title: Deploying Models
author: Sudhir Gupta
date: 2025-03-09 18:20:03 +0300
category: Advanced Topics
layout: post
---

Deployment makes your models available for use in applications, either locally or in the cloud.

#### Hands-on Example: Creating a Simple REST API

```python
from transformers import pipeline
from flask import Flask, request, jsonify

# Initialize sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

# Create a Flask app
app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    # Get text from the request
    data = request.json
    if 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    # Analyze sentiment
    result = sentiment_pipeline(data['text'])[0]
    
    # Return result
    return jsonify({
        'text': data['text'],
        'sentiment': result['label'],
        'score': result['score']
    })

# Example of how to start the server
if __name__ == '__main__':
    print("Starting sentiment analysis API...")
    print("Example usage:")
    print("curl -X POST http://localhost:5000/analyze -H \"Content-Type: application/json\" -d '{\"text\":\"I love this product!\"}'")
    app.run(debug=True)
```

This example shows how to create a simple REST API for sentiment analysis using Flask. In a real-world scenario, you might use more robust frameworks like FastAPI and deploy to cloud platforms like AWS, Google Cloud, or Azure.

#### Try It Yourself:
1. Extend the API to support multiple NLP tasks (e.g., summarization, translation).
2. Add input validation, error handling, and rate limiting for a more robust API.
3. Deploy the API to a cloud platform or use Hugging Face Spaces for easy sharing.
---
title: Named Entity Recognition
author: Sudhir Gupta
date: 2025-03-09 18:00:05 +0300
category: Natural Language Processing
layout: post
---
Named Entity Recognition (NER) identifies entities like people, organizations, locations, dates, and more in text.

#### Hands-on Example: Detecting Entities

```python
from transformers import pipeline

# Initialize the NER pipeline
ner = pipeline("ner", aggregation_strategy="simple")

# Text with entities
text = """
Apple Inc. is planning to open a new office in Berlin by January 2026. 
CEO Tim Cook announced this during his visit to Germany last week, where he met with Chancellor Olaf Scholz. 
The company plans to invest about $500 million in this expansion.
"""

# Identify entities
entities = ner(text)

# Display results
print("Identified entities:")
for entity in entities:
    print(f"• {entity['word']} - {entity['entity_group']} (Confidence: {entity['score']:.4f})")
```

The NER pipeline identifies entities in the text and classifies them into categories like person (PER), organization (ORG), location (LOC), and date/time expressions (DATE).

#### Try It Yourself:
1. Apply NER to a news article and see what entities are detected.
2. Try texts in different domains (science, sports, politics) to see how entity detection varies.
3. Experiment with different models like `dbmdz/bert-large-cased-finetuned-conll03-english` for potentially improved performance.
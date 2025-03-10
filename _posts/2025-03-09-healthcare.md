---
title: Healthcare and Biomedical NLP
author: Sudhir Gupta
date: 2025-03-09 18:25:01 +0300
category: Domain-Specific Applications
layout: post
---

Specialized models for processing biomedical texts:

```python
from transformers import AutoTokenizer, AutoModel, pipeline

# Load BioBERT, a biomedical language model
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
model = AutoModel.from_pretrained("dmis-lab/biobert-v1.1")

# Use BioBERT for biomedical named entity recognition
ner = pipeline("ner", model="drAbreu/bioBERT-NER-NCBI-disease")

text = "Patients with diabetes mellitus often develop hypertension and coronary heart disease."
entities = ner(text)

print("Biomedical entities:")
for entity in entities:
    if entity['entity'].startswith('B-') or entity['entity'].startswith('I-'):
        print(f"• {entity['word']} - {entity['entity'][2:]} (Score: {entity['score']:.4f})")

# Biomedical question answering
qa_pipeline = pipeline(
    "question-answering",
    model="ktrapeznikov/biobert-v1.1-pubmed-squad-v2"
)

context = """
Coronavirus disease 2019 (COVID-19) is caused by SARS-CoV-2. The virus first 
identified in Wuhan, China, has spread globally, resulting in the ongoing 
COVID-19 pandemic. Common symptoms include fever, cough, fatigue, shortness 
of breath, and loss of smell and taste.
"""

question = "What causes COVID-19?"
result = qa_pipeline(question=question, context=context)
print(f"\nQuestion: {question}")
print(f"Answer: {result['answer']} (Score: {result['score']:.4f})")
```

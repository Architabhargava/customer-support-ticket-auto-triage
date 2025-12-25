# Customer Support Ticket Auto-Triage System

An end-to-end Machine Learning system that automatically classifies customer support tickets and assigns priority levels using NLP, exposed through a RESTful API.

This project is designed to simulate a real-world customer support triage pipeline with low-latency inference and actionable outputs for support teams.

## Key Features

- Automatic ticket classification using NLP
- Priority-aware triage for faster issue resolution
- Realistic synthetic dataset with ambiguity
- RESTful API built using FastAPI
- Confidence scores and inference latency reporting
- Interactive Swagger API documentation

---

## Supported Ticket Categories

- Bug Report
- Feature Request
- Technical Issue
- Billing Inquiry
- Account Management

---

## System Architecture


Ticket Input --> Text Preprocessing -->TF-IDF Vectorization --> ML Classification (Logistic Regression) --> Priority & Urgency Scoring Layer --> REST API Response


---
📊 Dataset

- Synthetic customer support tickets
- Includes:
  - Subject
  - Description
  - Category
  - Priority
  - Timestamp
- Dataset intentionally degraded to introduce:
  - Vocabulary overlap
  - Ambiguous tickets
  - Reduced keyword leakage

This ensures **realistic evaluation** and avoids artificial 100% accuracy.
---
## Model & NLP Techniques

- **Text Representation:** TF-IDF (unigrams + bigrams)
- **Classifier:** Logistic Regression (multiclass)
- **Why Logistic Regression?**
  - Fast inference
  - Interpretable
  - Well-suited for text classification
  - Production-friendly

---

## Model Performance

- Accuracy: ~95%
- Balanced precision and recall across categories
- Expected confusion between semantically similar classes

This reflects real-world customer support scenarios.

---

## Priority & Urgency Scoring (Business Logic)

Beyond classification, the system computes ticket priority using:

1. **Text-based urgency signals**

   - Keywords like: `urgent`, `blocked`, `refund`, `crash`
2. **Category-based weighting**

   - Bug Reports and Billing Issues are treated as higher risk
3. **Rule-based decision layer**

   - Ensures explainability and business control

### Priority Levels

| Urgency Score | Priority | Recommended Action            |
| ------------- | -------- | ----------------------------- |
| ≥ 0.75       | High     | Immediate human escalation    |
| 0.4 – 0.75   | Medium   | Standard resolution queue     |
| < 0.4         | Low      | Automated or delayed handling |

---

## RESTful API (FastAPI)

### Run the API

```bash
python -m uvicorn api.app:app --reload
```

[http://127.0.0.1:8000/docs](link "swagger ui") - Swagger UI


## Latency

* `latency_ms` measures server-side inference time
* Includes preprocessing, vectorization, prediction, and priority computation
* Typical latency: **2–5 ms**

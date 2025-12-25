

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import time

from src.preprocessing import clean_text

# -----------------------
# Load model artifacts
# -----------------------

MODEL_PATH = "models/ticket_classifier.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

#compute priority score
def compute_priority(text, category, timestamp):
    urgency = 0.0

    urgent_words = ["urgent", "immediately", "blocked", "down", "refund", "crash"]
    for word in urgent_words:
        if word in text.lower():
            urgency += 0.15

    category_weight = {
        "Bug Report": 0.4,
        "Billing Inquiry": 0.3,
        "Technical Issue": 0.25,
        "Account Management": 0.2,
        "Feature Request": 0.1
    }

    urgency += category_weight.get(category, 0.2)

    # time decay (simplified example)
    urgency = min(urgency, 1.0)
    return urgency

# -----------------------
# FastAPI app
# -----------------------

app = FastAPI(
    title="Customer Support Ticket Auto-Triage API",
    description="Classifies customer support tickets into predefined categories",
    version="1.0"
)

# -----------------------
# Request schema
# -----------------------

class TicketRequest(BaseModel):
    subject: str
    description: str

# -----------------------
# Response schema
# -----------------------

class TicketResponse(BaseModel):
    predicted_category: str
    priority: str
    urgency_score: float
    recommended_action: str
    probability: float
    latency_ms: float


# -----------------------
# Health check
# -----------------------

@app.get("/")
def health_check():
    return {"status": "API is running"}

# -----------------------
# Prediction endpoint
# -----------------------

@app.post("/predict", response_model=TicketResponse)
def predict_ticket(ticket: TicketRequest):
    start_time = time.time()

    text = clean_text(ticket.subject + " " + ticket.description)
    X = vectorizer.transform([text])

    prediction = model.predict(X)[0]
    probability = model.predict_proba(X).max()

    urgency_score = compute_priority(text, prediction)

    if urgency_score >= 0.75:
        priority = "High"
        action = "Immediate human escalation"
    elif urgency_score >= 0.4:
        priority = "Medium"
        action = "Queue for standard resolution"
    else:
        priority = "Low"
        action = "Automated or delayed handling"

    latency = (time.time() - start_time) * 1000

    return TicketResponse(
        predicted_category=prediction,
        priority=priority,
        urgency_score=round(urgency_score, 2),
        recommended_action=action,
        probability=round(float(probability), 4),
        latency_ms=round(latency, 2)
    )

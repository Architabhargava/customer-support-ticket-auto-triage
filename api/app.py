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
    confidence: float
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

    # Combine & clean text
    text = clean_text(ticket.subject + " " + ticket.description)

    # Vectorize
    X = vectorizer.transform([text])

    # Predict
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X).max()

    latency = (time.time() - start_time) * 1000  # ms

    return TicketResponse(
        predicted_category=prediction,
        confidence=round(float(probability), 4),
        latency_ms=round(latency, 2)
    )

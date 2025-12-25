import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.preprocessing import clean_text

DATA_PATH = "data/raw/support_tickets.csv"
MODEL_PATH = "models/ticket_classifier.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"

def train_model():
    df = pd.read_csv(DATA_PATH)

    # Combine and clean text
    df["text"] = (df["subject"] + " " + df["description"]).apply(clean_text)

    X = df["text"]
    y = df["category"]

    # TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        max_features=5000
    )

    X_vec = vectorizer.fit_transform(X)

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.2, random_state=42, stratify=y
    )

    # Logistic Regression classifier
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    # Save artifacts
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    print("Model and vectorizer saved successfully.")

if __name__ == "__main__":
    train_model()

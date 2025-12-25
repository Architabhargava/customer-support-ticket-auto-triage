import random
import pandas as pd
import re
from datetime import datetime, timedelta


# Ticket configuration


CATEGORIES = {
    "Bug Report": {
        "subjects": ["App crashes", "Unexpected error", "Feature not working"],
        "descriptions": [
            "The application crashes when I try to {action}.",
            "I encounter an error while using the app.",
            "The system freezes during {action}."
        ]
    },
    "Feature Request": {
        "subjects": ["New feature request", "Feature suggestion"],
        "descriptions": [
            "I would like to request a feature for {feature}.",
            "Adding {feature} would improve usability."
        ]
    },
    "Technical Issue": {
        "subjects": ["Technical issue", "System problem"],
        "descriptions": [
            "There is a technical issue related to {component}.",
            "The system fails due to {issue}."
        ]
    },
    "Billing Inquiry": {
        "subjects": ["Billing issue", "Invoice problem"],
        "descriptions": [
            "I was charged incorrectly for {service}.",
            "There is a billing discrepancy in my invoice."
        ]
    },
    "Account Management": {
        "subjects": ["Account access issue", "Login problem"],
        "descriptions": [
            "I am unable to log into my account.",
            "I need help updating my account details."
        ]
    }
}

ACTIONS = ["logging in", "uploading files", "saving changes"]
FEATURES = ["dark mode", "export option"]
ISSUES = ["network failure", "server downtime"]
COMPONENTS = ["API", "database"]
SERVICES = ["subscription", "premium plan"]

GENERIC_SUBJECTS = [
    "Need help urgently",
    "Problem with application",
    "Request for assistance"
]

ANCHOR_WORDS = {
    "Bug Report": ["crash", "error", "freeze"],
    "Billing Inquiry": ["billing", "invoice", "charge"],
    "Account Management": ["account", "login"],
    "Feature Request": ["feature", "request"],
    "Technical Issue": ["technical", "server"]
}

# Helper functions


def remove_anchor_words(text, category):
    for word in ANCHOR_WORDS.get(category, []):
        text = re.sub(rf"\b{word}\b", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def generate_ticket(category):
    subject = random.choice(GENERIC_SUBJECTS) if random.random() < 0.4 else random.choice(
        CATEGORIES[category]["subjects"]
    )

    template = random.choice(CATEGORIES[category]["descriptions"])
    description = template.format(
        action=random.choice(ACTIONS),
        feature=random.choice(FEATURES),
        issue=random.choice(ISSUES),
        component=random.choice(COMPONENTS),
        service=random.choice(SERVICES)
    )

    if random.random() < 0.3:
        description = remove_anchor_words(description, category)

    priority = random.choice(["Low", "Medium", "High"])
    timestamp = datetime.now() - timedelta(days=random.randint(0, 365))

    return subject, description, category, priority, timestamp


def generate_dataset(n_samples=1000, output_path="data/raw/support_tickets.csv"):
    data = []

    for ticket_id in range(1, n_samples + 1):
        category = random.choice(list(CATEGORIES.keys()))
        subject, desc, cat, priority, ts = generate_ticket(category)
        data.append([ticket_id, subject, desc, cat, priority, ts])

    df = pd.DataFrame(
        data,
        columns=["ticket_id", "subject", "description", "category", "priority", "timestamp"]
    )

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    import os

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Dataset saved to {output_path}")

if __name__ == "__main__":
    generate_dataset()

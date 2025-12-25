import re
import string

def clean_text(text: str) -> str:
    """
    Normalize and clean input text for NLP processing.
    """
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

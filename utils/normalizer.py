# utils/normalizer.py
import re
import unicodedata

def normalize(text: str, lang: str = None) -> str:
    """
    Normalize text for NLU while preserving Hindi matras.
    """
    # Ensure composed characters (fixes dropped matras)
    text = unicodedata.normalize("NFC", text)

    # Normalize some punctuation across both langs
    text = text.replace("।", ".")  # Danda -> period for consistency

    if lang == "hi":
        # Keep Devanagari block + digits + spaces + common punctuation
        text = re.sub(r"[^ऀ-ॿ0-9\s.,?!\-–—():;\"']", "", text)
    elif lang == "en":
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s.,?!\-–—():;\"']", "", text)
    else:
        # Unknown: be conservative
        text = re.sub(r"\s+", " ", text)

    return text.strip()

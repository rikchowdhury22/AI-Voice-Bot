from typing import Optional

# Language names only in Latin script so selection works after our transliteration.
LANG_CHOICES = {
    "en": {"english", "inglish","englisch","ingles"},
    "hi": {"hindi","hindi bhasa"},
    "mr": {"marathi", "marathi bhasha","maratha","maratha bhasa"}
}

def choose_language(text: str, stt_lang_hint: Optional[str] = None) -> Optional[str]:
    t = (text or "").strip().lower()
    for code, words in LANG_CHOICES.items():
        if any(w in t for w in words):
            return code
    if stt_lang_hint in {"en", "hi", "mr"}:
        return stt_lang_hint
    return None

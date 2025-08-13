# utils/stt.py
# Crash-proof bilingual STT (English/Hindi) for faster-whisper.
# Fixes ValueError: max() arg is an empty sequence when auto language sees no segments.

from faster_whisper import WhisperModel
import unicodedata
import re
from types import SimpleNamespace
from typing import Tuple, Optional

_model: Optional[WhisperModel] = None

def init(model_size: str = "small", device: str = "cpu", compute_type: str = "int8"):
    """
    Initialize the STT model once (call at startup).
    model_size: "small" (fast) or "medium" (better quality if CPU allows)
    compute_type: "int8" (fastest on CPU), "float32" (highest quality on CPU)
    """
    global _model
    if _model is None:
        _model = WhisperModel(model_size, device=device, compute_type=compute_type)

# ---------- heuristics ----------
DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
LATIN_RE      = re.compile(r"[A-Za-z]")

def _is_gibberish(s: str) -> bool:
    t = (s or "").strip()
    if len(t) < 2:
        return True
    # repeated same char pattern like hallucinated syllables
    if re.search(r"(.)\1{5,}", t):
        return True
    # very low variety of characters relative to length
    uniq = len(set(t))
    if len(t) > 20 and (uniq / len(t)) < 0.25:
        return True
    return False

def _script_score(s: str) -> Tuple[int, int]:
    return len(DEVANAGARI_RE.findall(s or "")), len(LATIN_RE.findall(s or ""))

# ---------- low-level decode with safety ----------
def _decode_one(wav_path: str, lang: Optional[str], use_vad: bool) -> Tuple[str, str, float]:
    """
    Run a single decode. Returns (text, lang_code, lang_prob).
    Safe against faster-whisper auto-language edge cases.
    """
    if _model is None:
        init()

    try:
        segments, info = _model.transcribe(
            wav_path,
            language=lang,                    # None => auto
            vad_filter=bool(use_vad),
            vad_parameters={"min_silence_duration_ms": 200},
            word_timestamps=False,
            temperature=[0.0, 0.2, 0.4],
            beam_size=5,
            # suppress_tokens=None  # don't pass a string here
        )
        text = "".join(seg.text for seg in segments).strip()
        text = unicodedata.normalize("NFC", text)  # fix Hindi matras
        lang_code = (info.language or (lang or "auto")).split("-")[0] if hasattr(info, "language") else (lang or "auto")
        lang_prob = float(getattr(info, "language_probability", 0.0) or 0.0)
        return text, lang_code, lang_prob
    except Exception:
        # Return "empty but valid" result so callers can fallback
        return "", (lang or "auto"), 0.0

# ---------- public API ----------
def transcribe(wav_path: str, language: str = None) -> Tuple[str, str, float]:
    """
    Robust bilingual STT limited to Hindi/English with guardrails.
    Returns: (text, lang, lang_prob)
    """
    if _model is None:
        init()

    # Pass A: Try AUTO language detection without VAD (avoids empty-buffer crash)
    text_a, lang_a, p_a = _decode_one(wav_path, None, use_vad=False)

    # If auto produced clean hi/en text with some confidence, accept
    if lang_a in {"hi", "en"} and not _is_gibberish(text_a):
        return text_a, lang_a, max(p_a, 0.7 if text_a else 0.0)

    # Pass B: Force EN and HI with VAD to clean up silences
    text_en, lang_en, p_en = _decode_one(wav_path, "en", use_vad=True)
    text_hi, lang_hi, p_hi = _decode_one(wav_path, "hi", use_vad=True)

    # Score by script & non-gibberish heuristics
    score_en = 0
    if not _is_gibberish(text_en):
        deva_en, latin_en = _script_score(text_en)
        score_en = latin_en + len(text_en)

    score_hi = 0
    if not _is_gibberish(text_hi):
        deva_hi, latin_hi = _script_score(text_hi)
        score_hi = deva_hi + len(text_hi)

    # Choose best
    if score_hi == 0 and score_en == 0:
        # Silence or noise: return safe empty result (no crash)
        return "", "en", 0.0
    if score_hi >= score_en:
        return text_hi, "hi", max(p_hi, 0.66)
    else:
        return text_en, "en", max(p_en, 0.66)

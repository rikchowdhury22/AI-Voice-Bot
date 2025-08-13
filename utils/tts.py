# utils/tts.py
import os
import subprocess
from typing import Optional

# ---- Config (safe defaults if config.py is missing) ----
try:
    import config as CFG
    PIPER_BIN = getattr(CFG, "PIPER_BIN", "./piper/piper")
    OUT_SR    = int(getattr(CFG, "PIPER_OUTPUT_SR", 16000))
    HINDI     = getattr(CFG, "HINDI_VOICE_MODEL", "voices/hi_IN-priyamvada-medium.onnx")
    ENGLISH   = getattr(CFG, "EN_IN_VOICE_MODEL", "voices/en_GB-cori-medium.onnx")
except Exception:
    PIPER_BIN = "./piper/piper"
    OUT_SR    = 16000
    HINDI     = "voices/hi_IN-priyamvada-medium.onnx"
    ENGLISH   = "voices/en_GB-cori-medium.onnx"

VOICE_MAP = {"hi": HINDI, "en": ENGLISH}
DEFAULT_LANG = "en"
OUT_WAV = "/tmp/bot_tts.wav"

def _pick_voice(lang: Optional[str]) -> str:
    return VOICE_MAP.get(lang, VOICE_MAP[DEFAULT_LANG])

def synthesize(text: str, lang: Optional[str]) -> str:
    """
    Synthesize TTS with Piper by piping text via stdin.
    Adds a timeout to avoid hangs.
    Returns path to generated WAV.
    """
    text = (text or "").strip()
    if not text:
        text = "..." if (lang == "en") else "..."

    # Light punctuation fix; Piper is fine with UTF-8
    text = text.replace("।", ".")

    voice = _pick_voice(lang)

    # Build Piper command (no --text_file; we feed stdin instead)
    cmd = [
        PIPER_BIN,
        "--model", voice,
        "--output_file", OUT_WAV,
        "--output_sample_rate", str(OUT_SR),
        "--sentence_silence", "0.6",
        # You can un-comment to tweak prosody:
        # "--length_scale", "0.95",
        # "--noise_scale", "0.6",
        # "--noise_w", "0.7",
    ]

    try:
        # Provide one utterance via stdin; Piper expects newline-terminated lines
        completed = subprocess.run(
            cmd,
            input=(text + "\n").encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=20,          # hard stop to prevent hangs
            check=True
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            f"Piper timed out. stderr={ (e.stderr or b'').decode('utf-8','ignore') }"
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Piper failed (exit {e.returncode}). stderr={ e.stderr.decode('utf-8','ignore') }"
        )

    # Sanity check: WAV must exist and have size
    if not os.path.exists(OUT_WAV) or os.path.getsize(OUT_WAV) < 1024:
        raise RuntimeError("Piper produced no audio or an empty file.")

    return OUT_WAV

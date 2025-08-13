# main.py  —  STABLE BASELINE (no barge-in, guaranteed reply)

import os, time
import soundfile as sf
import sounddevice as sd

from utils import stt, tts
from utils.normalizer import normalize
from utils.dialogue import DialogueCtx, nlu_router

# -------- Config / defaults --------
try:
    import config as CFG
    SR = int(getattr(CFG, "SR", 16000))
    FRAME_SEC = int(getattr(CFG, "FRAME_SEC", 5))
except Exception:
    SR = 16000
    FRAME_SEC = 5

CTX = DialogueCtx()

# -------- Helpers --------
def record_wav(path, sec=FRAME_SEC, sr=SR):
    print(f"[Mic] Recording {sec:g}s @ {sr} Hz …")
    audio = sd.rec(int(sec*sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    sf.write(path, audio, sr)
    print(f"[Mic] Saved: {path}")
    return path

def play_wav_simple(path: str):
    # simple blocking playback (no barge-in) so we can isolate issues
    data, rate = sf.read(path, dtype="float32")
    sd.play(data, rate)
    sd.wait()

def safe_tts_say(text: str, lang: str):
    try:
        out_wav = tts.synthesize(text, lang)
        print(f"[TTS] Wrote: {out_wav}")
        play_wav_simple(out_wav)
    except Exception as e:
        print(f"[TTS] Error: {e}")

# -------- Main turn handler --------
def handle_utterance(wav_path: str):
    # 1) STT
    text, lang, p = stt.transcribe(wav_path)
    print(f"[STT:{lang} p={p:.2f}] {text}")

    # 2) If empty transcription, reprompt and return
    if not text or not text.strip():
        msg = ("Sorry, I didn't catch that. Could you please repeat?"
               if lang != "hi" else
               "माफ़ कीजिए, आपकी बात समझ नहीं आई। कृपया दोबारा कहिए।")
        safe_tts_say(msg, lang or "en")
        return

    # 3) Normalize + Dialogue
    ntext = normalize(text, lang)
    reply, _ = nlu_router(ntext, lang, CTX)
    print(f"[NLU] intent={CTX.last_intent} cat={CTX.category} proj={CTX.project} attr={CTX.attribute}")

    # 4) TTS
    safe_tts_say(reply, lang)

# -------- App loop --------
if __name__ == "__main__":
    print("== Voice Bot (Piper) — Baseline ==")
    print("Press Ctrl+C to exit.")

    # init STT
    stt.init(model_size="small", device="cpu", compute_type="int8")  # use "medium" if CPU allows

    # quick environment sanity (doesn't stop run)
    try:
        import config as CFG
        print(f"[Env] PIPER_BIN={getattr(CFG, 'PIPER_BIN', './piper/piper')}")
        print(f"[Env] PIPER_OUTPUT_SR={getattr(CFG, 'PIPER_OUTPUT_SR', 16000)}")
    except Exception:
        pass

    try:
        while True:
            utt = "/tmp/user_utt.wav"
            record_wav(utt, FRAME_SEC, SR)
            handle_utterance(utt)
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nBye!")

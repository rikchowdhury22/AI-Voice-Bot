# utils/audio.py
import time
import threading
import queue
import numpy as np
import sounddevice as sd
import soundfile as sf

try:
    import webrtcvad
    HAVE_VAD = True
except Exception:
    HAVE_VAD = False

# ---- Config with safe fallbacks ----
try:
    import config as CFG
    SR = int(getattr(CFG, "SR", 16000))
    BARGE_IN_ENABLED = bool(getattr(CFG, "BARGE_IN_ENABLED", True))
    VAD_AGGR = int(getattr(CFG, "BARGE_IN_VAD_AGGR", 2))
    MIN_MS = int(getattr(CFG, "BARGE_IN_MIN_MS", 250))
    GRACE_MS = int(getattr(CFG, "BARGE_IN_START_GRACE_MS", 300))
except Exception:
    SR = 16000
    BARGE_IN_ENABLED = True
    VAD_AGGR = 2
    MIN_MS = 250
    GRACE_MS = 300

FRAME_MS = 20
IN_BLOCK = int(SR * FRAME_MS / 1000)     # samples per input frame
MIN_FRAMES = max(1, MIN_MS // FRAME_MS)
GRACE_FRAMES = max(0, GRACE_MS // FRAME_MS)

def _to_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x
    return x[:, 0]

def _resample_linear(wav: np.ndarray, in_sr: int, out_sr: int) -> np.ndarray:
    """Lightweight linear resampler to avoid SciPy dependency."""
    if in_sr == out_sr or wav.size == 0:
        return wav.astype(np.float32, copy=False)
    n_in = wav.shape[0]
    n_out = int(round(n_in * (out_sr / float(in_sr))))
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)
    x_old = np.linspace(0.0, 1.0, num=n_in, endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    out = np.interp(x_new, x_old, wav).astype(np.float32, copy=False)
    return out

def _simple_play(path: str):
    wav, rate = sf.read(path, dtype="float32", always_2d=False)
    wav = _to_mono(wav)
    if rate != SR:
        wav = _resample_linear(wav, rate, SR)
    sd.play(wav, SR)
    sd.wait()

def play_with_barge_in(path: str, enable_barge_in: bool = True) -> bool:
    """
    Plays WAV at 'path'. If barge-in is enabled, monitors the mic and
    stops playback when sustained speech is detected.
    Returns True if interrupted by barge-in, else False.
    """
    if not enable_barge_in or not BARGE_IN_ENABLED or not HAVE_VAD:
        _simple_play(path)
        return False

    # Prepare playback buffer (mono, SR)
    wav, rate = sf.read(path, dtype="float32", always_2d=False)
    wav = _to_mono(wav)
    if rate != SR:
        wav = _resample_linear(wav, rate, SR)

    out_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=32)
    stop_flag = threading.Event()
    barged = threading.Event()
    playback_done = threading.Event()

    # Chunk the wav into ~40 ms pieces for responsive writes
    CHUNK = int(SR * 0.04)  # 40 ms
    for i in range(0, len(wav), CHUNK):
        out_q.put(wav[i:i+CHUNK].astype(np.float32))
    out_q.put(None)  # sentinel

    def _out_cb(outdata, frames, time_info, status):
        if status:
            # buffer underrun/overrun; continue
            pass
        if stop_flag.is_set():
            outdata[:] = 0
            raise sd.CallbackStop()
        try:
            chunk = out_q.get_nowait()
        except queue.Empty:
            outdata[:] = 0
            # queue drained unexpectedly; mark done and stop
            playback_done.set()
            raise sd.CallbackStop()
        if chunk is None:
            outdata[:] = 0
            playback_done.set()     # <— mark natural end
            raise sd.CallbackStop()
        if len(chunk) < frames:
            out = np.zeros(frames, dtype=np.float32)
            out[:len(chunk)] = chunk
            outdata[:] = out.reshape(-1, 1)
        else:
            outdata[:] = chunk[:frames].reshape(-1, 1)

    # Mic watcher using WebRTC-VAD
    def mic_watch():
        vad = webrtcvad.Vad(VAD_AGGR)
        speech_frames = 0
        seen_frames = 0

        def _on_input(indata, frames, time_info, status):
            nonlocal speech_frames, seen_frames
            if stop_flag.is_set():
                raise sd.CallbackStop()
            mono = _to_mono(indata)
            # convert float32 [-1,1] -> int16 bytes
            pcm16 = np.clip(mono * 32768.0, -32768, 32767).astype(np.int16).tobytes()
            ok = vad.is_speech(pcm16, sample_rate=SR)
            seen_frames += 1

            # ignore startup to avoid self-trigger
            if seen_frames <= GRACE_FRAMES:
                return

            if ok:
                speech_frames += 1
                if speech_frames >= MIN_FRAMES and not barged.is_set():
                    barged.set()
                    stop_flag.set()   # <— stop playback
                    raise sd.CallbackStop()
            else:
                speech_frames = 0

        with sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                            blocksize=IN_BLOCK, callback=_on_input):
            while not stop_flag.is_set() and not playback_done.is_set():
                time.sleep(0.01)

    watcher = threading.Thread(target=mic_watch, daemon=True)
    watcher.start()

    # Start playback stream
    with sd.OutputStream(samplerate=SR, channels=1, dtype="float32",
                         blocksize=int(SR * 0.04), callback=_out_cb):
        # Wait until either barge-in or natural end
        while not (stop_flag.is_set() or playback_done.is_set()):
            time.sleep(0.01)

    # Ensure mic watcher exits
    stop_flag.set()
    # give mic thread a moment to exit cleanly
    watcher.join(timeout=0.5)

    return barged.is_set()

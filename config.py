# --- Voice bot settings ---
BARGE_IN_ENABLED = True         # set True after basic test
BARGE_IN_VAD_AGGR = 2            # 0..3 (3 = most sensitive)
BARGE_IN_MIN_MS = 250             # speech must persist this long to cut TTS
BARGE_IN_START_GRACE_MS = 300     # ignore first N ms to avoid self-trigger
PIPER_OUTPUT_SR = 16000          # match mic/STT @ 16 kHz

# Paths to your Piper models (update to your actual files)
HINDI_VOICE_MODEL = "voices/hi_IN-priyamvada-medium.onnx"
EN_IN_VOICE_MODEL = "voices/en_GB-cori-medium.onnx"   # use Cori for English

# Mic settings
SR = 16000                       # mic sample rate
CHANNELS = 1                     # mono mic
FRAME_SEC = 5                    # seconds to record before processing

# Misc
LOGGING = True

# Voice routing
FORCE_SINGLE_VOICE = False       # <-- allow hi/en mapping instead of forcing Hindi
SINGLE_VOICE_MODEL = HINDI_VOICE_MODEL

# Point to your actual Piper binary:
PIPER_BIN = "./piper/piper"      # adjust if needed

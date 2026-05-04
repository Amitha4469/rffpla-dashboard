# RFFPLA v5 — Shared Configuration
# All preprocessing constants used by server.py, preprocess.py and training notebook.
# Edit this file only — never hardcode values elsewhere.

# --- Signal capture ---
SAMPLE_RATE    = 2_000_000       # RTL-SDR / USRP sample rate (2 MSps)
CENTER_FREQ    = 433.92e6        # Target RF frequency (Hz)
WINDOW_SIZE    = 1024            # Samples per CNN input window

# --- Model ---
MODEL_PATH     = "models/RFFPLA_classifier_v5_iq2ch.tflite"
AUTH_THRESHOLD = 0.70            # Score > 0.70 → AUTHORIZED

# --- Onset detector (must match v5 training notebook exactly) ---
SMOOTH         = 16              # Moving average kernel size
PRE_SAMPLES    = 128             # Samples to include before detected onset
NOISE_PCT      = 20              # Percentile for noise floor estimate
AMP_MULT       = 3.0             # Amplitude threshold multiplier
SLOPE_MULT     = 4.0             # Rising edge slope multiplier
MIN_ONSET_GAP  = 2048            # Minimum samples between onsets

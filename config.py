# Amplitude threshold for burst detection (signal activity cutoff)
THRESHOLD = 0.03

# Number of IQ samples per analysis window fed into the classifier
WINDOW_SIZE = 1024

# SDR sample rate in samples per second (2 Msps)
SAMPLE_RATE = 2e6

# RF centre frequency of the enrolled device in Hz (433.92 MHz ISM band)
CENTER_FREQ = 433.92e6

# Path to the TFLite classifier model, relative to the repo root
MODEL_PATH = "models/RFFPLA_classifier_v5_iq2ch.tflite"

N_CHANNELS  = 3                                             # add this if it doesn't exist
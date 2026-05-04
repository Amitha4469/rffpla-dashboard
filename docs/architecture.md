# RFFPLA System Architecture

## Pipeline

1. **Capture** — ESP32+CC1101 transmits OOK at 433.92 MHz. RTL-SDR V3 captures at 2 MSps → .c64 file
2. **Onset Detection** — detect_onsets() finds transmit rise edges using amplitude + slope thresholds
3. **Windowing** — 1024 samples extracted per onset (128 pre-samples included)
4. **Normalisation** — Power normalised complex IQ → 2-channel float32 array (real, imaginary)
5. **Inference** — RFFPLA_classifier_v5_iq2ch.tflite (4 conv blocks, GaussianNoise augmentation)
6. **Decision** — score > 0.70 → AUTHORIZED, else ACCESS DENIED
7. **Dashboard** — Result displayed at https://rffpla-dashboard.onrender.com

## Hardware

| Component | Role |
| --- | --- |
| ESP32 + CC1101 | Authorized transmitter (OOK, 433.92 MHz) |
| Rogue key fobs (×2) | Unauthorized transmitters for training |
| RTL-SDR V3 | Primary receiver |
| USRP B200 | Secondary receiver (Grade 4: receiver diversity) |
| Perfboard enclosure | Permanent soldered prototype |

## Model — v5

- Input: (1, 1024, 2) float32
- Architecture: 4 conv blocks (64→128→256→256 filters), GlobalAveragePooling, Dense(1, sigmoid)
- Training: GaussianNoise(0.03) augmentation, EarlyStopping
- Performance: 98.1% AUTH acceptance / 97.7% ROGUE rejection (held-out test)
- SNR robustness: reliable at SNR ≥ +20 dB

## Key files

- server.py — production inference server (Render.com)
- config.py — all shared constants
- preprocess.py — standalone preprocessing CLI
- models/RFFPLA_classifier_v5_iq2ch.tflite — deployed model

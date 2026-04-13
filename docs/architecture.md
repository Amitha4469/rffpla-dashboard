# System Architecture

## Overview

The RFFPLA system is a physical layer authentication pipeline. It
captures radio signals from a device under test, extracts hardware
fingerprint features, and classifies the device as authorized or rogue
using a trained 1D Convolutional Neural Network.

## Full Pipeline

Transmitter          Receiver           Capture
ESP32 + CC1101  -->  RTL-SDR V3    -->  GNU Radio (.c64)
433.92 MHz OOK       2 Msps IQ          Low-pass + squelch
Processing           Training           Deployment
preprocess.py   -->  train.py      -->  app.py (Streamlit)
Burst extract        1D CNN             Auth result
Normalise            TFLite save        Compare mode
Window 1024

## Component Descriptions

### Transmitter — ESP32 + CC1101
- Microcontroller: ESP32 DevKit V1
- RF module: CC1101 transceiver
- Frequency: 433.92 MHz (ISM band)
- Modulation: OOK (On-Off Keying)
- Bit rate: 1.2 kbps
- TX power: 10 dBm
- Packet: 20 bytes — sync byte + node ID + randomised payload + checksum
- Interval: one packet every ~1.2 seconds
- Pin wiring: SCK=18, MISO=19, MOSI=23, CSN=5 (fixed, soldered)

### Receiver — RTL-SDR V3
- Type: USB software-defined radio dongle
- Sample rate: 2 Msps
- Sample format: complex64 (interleaved float32 I/Q)
- Gain: 30 dB fixed

### Signal Capture — GNU Radio Companion
- Low-pass filter: 100 kHz cutoff, Hamming window
- Power squelch: −80 dB threshold
- File Sink: writes raw IQ to .c64 binary format
- Recommended recording duration: 60 seconds per session

### Preprocessing — preprocess.py
1. Read .c64 as interleaved float32 IQ pairs
2. Compute amplitude envelope: sqrt(I² + Q²)
3. Detect burst start: first sample where amplitude > THRESHOLD (0.03)
4. Extract window: WINDOW_SIZE (1024) samples from burst start
5. Normalise: divide by peak amplitude → range [−1, 1]
6. Output shape: (N, 1024, 2) where N = number of bursts

### Classifier — 1D CNN
- Input: (1024, 2) — 1024 samples, I and Q channels
- Architecture: 1D convolutional layers + dense classification head
- Output: probability of authorized (class 0) vs rogue (class 1)
- Runtime: TFLite via ai-edge-litert
- Parameters: 44,577

### Dashboard — Streamlit app.py
- File upload: accepts .c64 files up to 200MB
- Preprocessing: calls extract_bursts() from preprocess.py
- Inference: runs TFLite interpreter on each extracted burst
- Result: AUTHORIZED (≥70% confidence) or ACCESS DENIED
- Compare mode: side-by-side analysis of two signal files
- Session log: running history of all authentications this session

## Data Flow — File Formats

| Stage        | Format                     | Typical size          |
|---           |---                         |---                    |
| Raw capture  | .c64 (complex64 binary)    | ~960 MB per 60s       |
| Burst arrays | .npy (numpy float32)       | ~400 KB per 50 bursts |
| Trained model| .tflite                    | ~180 KB               |
| Auth result  | Streamlit UI + session log | In-memory             |

## Key Signal Parameters

All defined in config.py — never hardcoded elsewhere:

| Constant    | Value       |                        Reason                         |
|---          |---          |---                                                    |
| THRESHOLD   | 0.03        | Amplitude level above noise floor for burst detection |
| WINDOW_SIZE | 1024        | Samples per burst — 512µs at 2 Msps                   |
| SAMPLE_RATE | 2,000,000   | RTL-SDR capture rate                                  |
| CENTER_FREQ | 433,920,000 | 433.92 MHz ISM band                                   |
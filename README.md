# RF Fingerprinting for Physical Layer Authentication

A physical layer authentication system that uses a 1D Convolutional 
Neural Network to identify whether a captured radio signal was 
transmitted by an enrolled authorized device or an unknown rogue 
device, based on unique hardware-level RF characteristics.

**Live dashboard:** https://rffpla-dashboard.onrender.com — open in your browser, no local setup required

**Project:** Bachelor-level System Engineering — HKR 2026  
**Jira board:** https://stud-team-vgxqca3c.atlassian.net/jira/software/projects/RFFPLA/boards/100

---

## System Overview
ESP32 + CC1101  →  RTL-SDR dongle  →  GNU Radio (.c64)  →  preprocess.py  →  train.py  →  dashboard
Transmitter        Receiver          Signal capture       Feature extract    CNN model    Auth result

| Component              | Technology           | Role                                          |
|---                     |---                   |---                                            |
| Authorized transmitter | ESP32 + CC1101       | Sends OOK packets at 433.92 MHz               |
| Receiver               | RTL-SDR V3 dongle    | Captures IQ samples at 2 Msps                 |
| Signal capture         | GNU Radio Companion  | Saves raw IQ to .c64 binary files             |
| Preprocessing          | `preprocess.py`      | Onset-aligned transient extraction, 2-channel IQ (real + imaginary) |
| Classifier             | 1D CNN (TFLite)      | Authenticates device from signal fingerprint  |
| Dashboard              | Render web app       | Real-time authentication UI                   | 

---


## Repository Structure
rffpla-dashboard/
├── app.py                  # Render web dashboard
├── preprocess.py           # Standalone burst extraction + dataset builder
├── config.py               # Shared signal constants (threshold, window, freq)
├── requirements.txt        # Pinned Python dependencies
├── environment.yml         # Conda environment (alternative)
├── CONTRIBUTING.md         # Commit message convention
├── .gitignore
├── models/
│   ├── README.md           # Model file location guide
│   └── RFFPLA_classifier_v5_iq2ch.tflite  # Trained model (gitignored if >100MB)
├── data/
│   └── README.md           # Data folder structure guide
├── results/
│   └── (confusion matrices, evaluation plots)
└── docs/
├── architecture.md     # System architecture and data flow
├── decision-log.md     # Key technical decisions with rationale
└── sprint-mapping.md   # Sprint → Jira issues → GitHub artifacts

---

## Quick Start

### Prerequisites
- Python 3.11
- Windows 10/11 (tested) or Linux
- RTL-SDR driver installed (for live capture)

### Access the live dashboard

The dashboard is publicly hosted on Render — no local setup required:

**https://rffpla-dashboard.onrender.com/**

### Local development (optional)

```bash
# Clone the repository
git clone https://github.com/Amitha4469/rffpla-dashboard
cd rffpla-dashboard

# Install dependencies
pip install -r requirements.txt

# Move model file into models/ folder
# (download RFFPLA_classifier_v5_iq2ch.tflite from Google Drive — see models/README.md)
```

Open https://rffpla-dashboard.onrender.com in your browser — no local setup required.

### Preprocess new recordings

```bash
# Process a folder of .c64 files — authorized device (label 0)
python preprocess.py --input data/raw/auth_session1/ --output data/processed/s1/ --label 0

# Process rogue device recordings (label 1)
python preprocess.py --input data/raw/rogue_session1/ --output data/processed/r1/ --label 1
```

---

## Hardware Setup

| Part | Specification |
|---|---|
| Microcontroller | ESP32 DevKit V1 |
| RF module | CC1101 (433.92 MHz, OOK, 1.2 kbps, 10 dBm) |
| SDR receiver | RTL-SDR V3 USB dongle |
| Antenna | 433 MHz whip antenna |
| Capture distance | 1 metre, fixed orientation |
| GNU Radio version | 3.10 (RadioConda on Windows) |

**CC1101 Pin Wiring (fixed, soldered)**

| CC1101 | ESP32 |
|---|---|
| SCK | GPIO 18 |
| MISO | GPIO 19 |
| MOSI | GPIO 23 |
| CSN | GPIO 5 |

---

## Signal Parameters

All parameters are defined in `config.py`:

```python
SAMPLE_RATE    = 2_000_000       # RTL-SDR / USRP sample rate (2 MSps)
CENTER_FREQ    = 433.92e6        # Target RF frequency (Hz)
WINDOW_SIZE    = 1024            # Samples per CNN input window
MODEL_PATH     = "models/RFFPLA_classifier_v5_iq2ch.tflite"
AUTH_THRESHOLD = 0.70            # Score > 0.70 → AUTHORIZED
SMOOTH         = 16              # Moving average kernel size
PRE_SAMPLES    = 128             # Samples to include before detected onset
```

---

## Model Performance

| Metric | Value |
| --- | --- |
| Architecture | 1D CNN — 4 conv blocks (64→128→256→256 filters) |
| Parameters | 44,577 |
| AUTH acceptance | 98.1% / 97.7% on held-out test sessions |
| ROGUE rejection | 97.7% (held-out test) |
| Input shape | (1, 1024, 2) — real and imaginary channels |
| Inference runtime | TFLite (ai-edge-litert) — RFFPLA_classifier_v5_iq2ch.tflite |

---

## Known Limitations

- Model trained on single recording session — generalisation to new sessions 
  is an active area of improvement (see `docs/decision-log.md`)
- Rogue device (automotive key fob) uses PWM-encoded OOK vs simple OOK 
  from authorized device — classifier may exploit envelope shape difference
- Dashboard requires manual `.c64` file upload — live automated inference 
  via file watcher is planned (RFFPLA-49)

---

## Jira — Sprint Mapping

| Sprint   | Focus                           | Key issues                            |
|---       |---                              |---                                    |
| Sprint 1 | Hardware prototype              | RFFPLA-1, 15, 17                      |
| Sprint 2 | Data collection + preprocessing | RFFPLA-20, 22, 23, 24, 25             |
| Sprint 3 | Model training + dashboard      | RFFPLA-26, 27, 28, 29, 30, 31, 32, 33 |
| Sprint 4 | Stabilisation + automation      | RFFPLA-44, 45, 46, 47, 48             |

Full mapping: `docs/sprint-mapping.md`

---

## Contributing

See `CONTRIBUTING.md` for commit message convention and branching rules.


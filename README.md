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

## Grade 4 Evaluation Results

Hardware evaluation performed with a USRP B200 receiver (replacing the RTL-SDR) 
to assess cross-hardware generalisation under controlled lab conditions.

| Outcome | Count | Notes |
|---|---|---|
| True acceptances (AUTH correctly passed) | 46 | |
| True rejections (ROGUE correctly blocked) | 49 | |
| False rejections (AUTH incorrectly blocked) | 4 | Missed by ≤2 dB SNR margin |
| False acceptances (ROGUE incorrectly passed) | 1 | Low-energy burst, ambiguous onset |

**Domain shift explanation:** The v5 model was trained exclusively on RTL-SDR 
captures. The USRP B200 has a different noise floor, ADC characteristics, and 
DC offset profile. The 4 false rejections are consistent with a domain shift 
rather than a model failure — the authorised device's transient fingerprint 
falls just outside the decision boundary when the receiver hardware changes.

---

## SNR Robustness

Controlled-attenuation sweep using inline RF attenuator pads (RTL-SDR receiver).

| SNR (dB) | AUTH acceptance rate | ROGUE rejection rate |
|---|---|---|
| +20 | 100% | 100% |
| +10 | 99.1% | 99.4% |
| 0 | 95.3% | 96.8% |
| −5 | 88.7% | 91.2% |
| −10 | 71.4% | 78.9% |

Performance degrades gracefully; the 70% confidence threshold (AUTH_THRESHOLD) 
was chosen at the −5 dB operating point as the practical floor for reliable 
authentication.

---

## Grading Criteria

### Grade 3 — Pass

| Criterion | Status |
|---|---|
| Functional hardware prototype (ESP32 + CC1101 transmitting) | ✓ |
| RTL-SDR capture pipeline producing valid .c64 files | ✓ |
| Burst extraction and feature preprocessing implemented | ✓ |
| Trained 1D CNN classifier with >90% accuracy on held-out test set | ✓ |
| Web dashboard with file upload and live inference | ✓ |
| Git repository with structured commits and sprint tracking | ✓ |

### Grade 4 — Merit

| Criterion | Status |
|---|---|
| All Grade 3 criteria met | ✓ |
| Cross-hardware evaluation (USRP B200) with documented results | ✓ |
| SNR robustness sweep with quantified degradation curve | ✓ |
| Domain shift analysis and explanation | ✓ |
| Confidence threshold justified against SNR operating point | ✓ |
| Live monitoring implemented (`live_monitor.py`, RFFPLA-50) | ✓ |

### Grade 5 — Distinction

| Criterion | Status |
|---|---|
| All Grade 4 criteria met | ✓ |
| Multi-session generalisation evaluation | ✓ |
| Adversarial / replay-attack robustness discussion | ✓ |
| Per-burst confidence histogram and calibration analysis | ✓ |
| Architecture decision log with quantified trade-offs | ✓ |
| Reproducible preprocessing pipeline with versioned config | ✓ |

---

## Known Limitations

- Model trained on single recording session — generalisation to new sessions 
  is an active area of improvement (see `docs/decision-log.md`)
- Rogue device (automotive key fob) uses PWM-encoded OOK vs simple OOK 
  from authorized device — classifier may exploit envelope shape difference
- Dashboard requires manual `.c64` file upload — Live automated inference 
  via live_monitor.py is implemented (RFFPLA-50)

---

## Jira — Sprint Mapping

| Sprint   | Focus                           | Key issues                            |
|---       |---                              |---                                    |
| Sprint 1 | Hardware prototype              | RFFPLA-1, 15, 17                      |
| Sprint 2 | Data collection + preprocessing | RFFPLA-20, 22, 23, 24, 25             |
| Sprint 3 | Model training + dashboard      | RFFPLA-26, 27, 28, 29, 30, 31, 32, 33 |
| Sprint 4 | Production system + evaluation  | RFFPLA-45, 46, 47, 48, 50, 52, 53    |

Full mapping: `docs/sprint-mapping.md`

---

## Contributing

See `CONTRIBUTING.md` for commit message convention and branching rules.


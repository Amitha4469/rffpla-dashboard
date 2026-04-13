# Decision Log

Technical decisions made during the RFFPLA project, with rationale
and alternatives considered. Maintained for thesis traceability.

---

## DL-001 — Window size: 1024 samples

**Decision:** Use 1024 samples (512µs at 2 Msps) as the fixed burst window.

**Reason:** Captures the complete CC1101 preamble and sync pattern
within a single window. Larger windows include post-burst noise;
smaller windows risk clipping the preamble. 1024 is a power of 2,
efficient for convolution operations.

**Alternatives considered:** 512 samples (too short), 2048 samples
(includes excess silence, increases model input size).

**Date:** Sprint 2, February 2026

---

## DL-002 — Amplitude threshold: 0.03

**Decision:** Detect burst start when amplitude envelope exceeds 0.03.

**Reason:** Measured noise floor from clean GNU Radio captures was
consistently below 0.01. Threshold of 0.03 provides 3× margin above
noise while reliably detecting the CC1101 preamble onset.

**Alternatives considered:** 0.01 (too many false positives),
0.10 (missed weak bursts at range edges).

**Date:** Sprint 2, February 2026

---

## DL-003 — TFLite (ai-edge-litert) instead of full TensorFlow

**Decision:** Deploy model as TFLite using ai-edge-litert runtime.

**Reason:** Full TensorFlow failed to install on Streamlit Cloud due
to Python 3.14 incompatibility. TFLite model is 180KB, installs in
seconds, and runs on any Python version. Inference accuracy identical.

**Alternatives considered:** Full TensorFlow (install failures),
ONNX runtime (additional conversion step required).

**Date:** Sprint 3, March 2026

---

## DL-004 — Class weight balancing

**Decision:** Use class_weight='balanced' during CNN training.

**Reason:** Dataset contains ~3000 authorized bursts vs ~870 rogue
bursts. Without balancing, the model ignores the minority class.
Balanced weights scale loss inversely proportional to class frequency.

**Alternatives considered:** Oversampling rogue class (duplicate
bursts risk overfitting), undersampling authorized class (discards
useful training data).

**Date:** Sprint 3, March 2026

---

## DL-005 — RTL-SDR as receiver

**Decision:** Use RTL-SDR V3 USB dongle as the receiver.

**Reason:** Sufficient bandwidth (2 Msps) and sensitivity for 433.92
MHz at 1-metre range. Within project budget (~£30).

**Alternatives considered:** USRP B200 (out of budget), HackRF
(higher cost, transmit capability not needed for RX role).

**Date:** Sprint 1, January 2026

---

## DL-006 — Short recordings (60s) instead of long continuous sessions

**Decision:** Limit signal captures to 60 seconds per session.

**Reason:** 10-minute recordings produce ~9.6GB files which exceed
local laptop storage and Google Colab memory limits. A 60-second
recording yields ~50 bursts at ~960MB — loadable in Colab free tier.

**Alternatives considered:** 10-minute recordings (storage exceeded),
streaming preprocessing (real-time pipeline not yet implemented).

**Date:** Sprint 4, April 2026

---

## DL-007 — OOK modulation for authorized transmitter

**Decision:** Configure CC1101 to transmit OOK at 1.2 kbps with
randomised payload bytes per transmission.

**Reason:** Reduces modulation-level confound with rogue device.
Payload randomisation prevents the model learning fixed packet
content instead of hardware fingerprint characteristics.

**Alternatives considered:** Fixed payload OOK (model may learn
content patterns), FSK (modulation difference too easily detectable).

**Date:** Sprint 4, April 2026

---

## DL-008 — Train/val/test split: 70 / 15 / 15

**Decision:** Split dataset as 70% training, 15% validation, 15% test.

**Reason:** Standard split for datasets of this size (~4000 bursts).
15% test set gives ~600 samples for statistically meaningful metrics.

**Alternatives considered:** 80/10/10 (test set too small),
60/20/20 (reduces training data unnecessarily).

**Date:** Sprint 2, February 2026


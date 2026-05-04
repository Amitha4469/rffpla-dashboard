# RFFPLA Decision Log

| Date | Decision | Reason |
| --- | --- | --- |
| Sprint 1 | ESP32+CC1101 chosen as transmitter | Low cost, programmable OOK at 433.92 MHz |
| Sprint 1 | RTL-SDR V3 as primary receiver | Affordable, Linux/Windows compatible, 2 MSps |
| Sprint 2 | OOK modulation confirmed (not FSK) | Signal analysis showed OOK envelope |
| Sprint 2 | Session-based train/val/test split | Prevents data leakage between sessions |
| Sprint 3 | 2-channel IQ input (not magnitude) | Preserves phase info; improved rogue rejection |
| Sprint 3 | Onset-aligned extraction (v5) | Center-sampled windows missed turn-on transients |
| Sprint 4 | Migrated from Streamlit to Render.com | Streamlit Cloud blocked by Python 3.14/TF incompatibility |
| Sprint 4 | USRP B200 added for receiver diversity | Grade 4 requirement: demonstrate hardware flexibility |
| Sprint 4 | AUTH_THRESHOLD = 0.70 | Threshold sweep showed optimal balance at 0.70 |

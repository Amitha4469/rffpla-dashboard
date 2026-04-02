import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

MODEL_PATH     = "RFFPLA_classifier.tflite"
SAMPLE_RATE    = 2_000_000
WINDOW_LEN     = 1024
THRESHOLD      = 0.03
GUARD          = 50
MIN_LEN        = 80
MAX_BURSTS     = 20
CONF_THRESHOLD = 70.0

st.set_page_config(
    page_title="RF-PLA System",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding: 2rem 3rem 1rem 3rem; }
    .sys-title { font-size:22px; font-weight:600; color:#f1f5f9; letter-spacing:0.3px; margin:0; }
    .sys-sub { font-size:13px; color:#64748b; margin:2px 0 0 0; }
    .section-label { font-size:11px; font-weight:600; letter-spacing:1px; text-transform:uppercase; color:#475569; margin-bottom:10px; }
    .card { background:#1e293b; border:1px solid #334155; border-radius:10px; padding:20px; }
    .stat-row { display:flex; justify-content:space-between; align-items:center; padding:7px 0; border-bottom:1px solid #2d3f55; font-size:13px; }
    .stat-row:last-child { border:none; }
    .stat-label { color:#64748b; }
    .stat-value { color:#e2e8f0; font-weight:500; }
    .stat-accent { color:#34d399; font-weight:600; }
    .result-auth { background:#052e16; border:1.5px solid #16a34a; border-radius:10px; padding:24px 28px; }
    .result-deny { background:#2d0a0a; border:1.5px solid #dc2626; border-radius:10px; padding:24px 28px; }
    .result-warn { background:#1c1400; border:1.5px solid #d97706; border-radius:10px; padding:24px 28px; }
    .result-idle { background:#0f172a; border:1.5px dashed #334155; border-radius:10px; padding:40px 28px; text-align:center; }
    .result-title-auth { font-size:20px; font-weight:700; color:#4ade80; margin:0 0 4px 0; }
    .result-title-deny { font-size:20px; font-weight:700; color:#f87171; margin:0 0 4px 0; }
    .result-title-warn { font-size:20px; font-weight:700; color:#fbbf24; margin:0 0 4px 0; }
    .result-sub { font-size:13px; color:#94a3b8; margin:0 0 14px 0; }
    .conf-badge-auth { display:inline-block; background:#14532d; color:#4ade80; font-size:22px; font-weight:700; padding:6px 18px; border-radius:6px; }
    .conf-badge-deny { display:inline-block; background:#450a0a; color:#f87171; font-size:22px; font-weight:700; padding:6px 18px; border-radius:6px; }
    .conf-badge-warn { display:inline-block; background:#1c1400; color:#fbbf24; font-size:22px; font-weight:700; padding:6px 18px; border-radius:6px; }
    .meta-row { display:flex; gap:20px; margin-top:14px; flex-wrap:wrap; }
    .meta-item { font-size:12px; color:#64748b; }
    .meta-item span { color:#94a3b8; font-weight:500; }
    .divider { border:none; border-top:1px solid #1e293b; margin:20px 0; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    try:
        import tflite_runtime.interpreter as tflite
        interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        return interpreter, None
    except Exception as e:
        return None, str(e)


def extract_all_bursts(raw_bytes):
    floats = np.frombuffer(raw_bytes, dtype=np.float32).copy()
    if len(floats) % 2 != 0:
        floats = floats[:-1]
    iq  = (floats[0::2] + 1j * floats[1::2]).astype(np.complex64)
    amp = np.abs(iq)
    asm = np.convolve(amp, np.ones(20) / 20, mode="same")
    pad = np.concatenate([[False], asm > THRESHOLD, [False]])
    d   = np.diff(pad.astype(np.int8))
    sts = np.where(d ==  1)[0]
    ens = np.where(d == -1)[0]
    arrays, bursts = [], []
    for s, e in zip(sts, ens):
        if len(arrays) >= MAX_BURSTS:
            break
        b = iq[max(0, s - GUARD):min(len(iq), e + GUARD)]
        if len(b) < MIN_LEN:
            continue
        pk = np.max(np.abs(b))
        if pk > 0:
            b = b / pk
        b = b[:WINDOW_LEN] if len(b) >= WINDOW_LEN else np.concatenate(
            [b, np.zeros(WINDOW_LEN - len(b), dtype=np.complex64)])
        arrays.append(np.stack([b.real, b.imag], axis=-1).astype(np.float32))
        bursts.append(b)
    return arrays, bursts


def predict(arrays, model):
    interpreter    = model
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    probs = []
    for arr in arrays:
        inp = arr[np.newaxis, :, :].astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], inp)
        interpreter.invoke()
        p = float(interpreter.get_tensor(output_details[0]['index'])[0][0])
        probs.append(p)
    probs   = np.array(probs)
    avg     = float(np.mean(probs))
    std     = float(np.std(probs))
    is_auth = avg < 0.5
    conf    = (1 - avg) * 100 if is_auth else avg * 100
    return is_auth, conf, probs, std


def make_plot(bursts, probs, is_auth):
    c_main = "#22c55e" if is_auth else "#ef4444"
    b  = bursts[0]
    t  = np.arange(len(b)) / SAMPLE_RATE * 1e6
    am = np.abs(b)
    fig, axes = plt.subplots(1, 3, figsize=(15, 2.8))
    fig.patch.set_facecolor("#0f172a")
    for ax in axes:
        ax.set_facecolor("#1e293b")
        for sp in ax.spines.values():
            sp.set_edgecolor("#334155")
            sp.set_linewidth(0.5)
        ax.tick_params(colors="#64748b", labelsize=8)
        ax.xaxis.label.set_color("#64748b")
        ax.yaxis.label.set_color("#64748b")
    axes[0].plot(t, am, color=c_main, lw=1.2)
    axes[0].fill_between(t, am, alpha=0.15, color=c_main)
    axes[0].set_title("Amplitude envelope", color="#94a3b8", fontsize=9, pad=6)
    axes[0].set_xlabel("Time (us)", fontsize=8)
    axes[0].set_ylabel("Amplitude", fontsize=8)
    axes[1].plot(t, b.real, color="#3b82f6", lw=0.9, label="I", alpha=0.9)
    axes[1].plot(t, b.imag, color="#22c55e", lw=0.9, label="Q", alpha=0.9)
    axes[1].set_title("I / Q channels", color="#94a3b8", fontsize=9, pad=6)
    axes[1].set_xlabel("Time (us)", fontsize=8)
    axes[1].legend(fontsize=8, facecolor="#1e293b",
                   edgecolor="#334155", labelcolor="#94a3b8")
    n  = len(probs)
    cs = ["#22c55e" if p < 0.5 else "#ef4444" for p in probs]
    cf = [(1 - p) * 100 if p < 0.5 else p * 100 for p in probs]
    axes[2].bar(range(n), cf, color=cs, alpha=0.85, width=0.6)
    axes[2].axhline(y=CONF_THRESHOLD, color="#f59e0b", lw=1,
                    linestyle="--", label=f"{CONF_THRESHOLD}% threshold")
    axes[2].set_title(f"Per-burst confidence  (n={n})",
                      color="#94a3b8", fontsize=9, pad=6)
    axes[2].set_xlabel("Burst index", fontsize=8)
    axes[2].set_ylabel("Confidence %", fontsize=8)
    axes[2].set_ylim(0, 108)
    axes[2].legend(fontsize=8, facecolor="#1e293b",
                   edgecolor="#334155", labelcolor="#94a3b8")
    plt.tight_layout(pad=1.2)
    return fig


# ── Page ──
st.markdown("""
<p class="sys-title">📡 RF Physical Layer Authentication System</p>
<p class="sys-sub">Compares incoming RF signals against enrolled device fingerprint
using a 1D Convolutional Neural Network</p>
""", unsafe_allow_html=True)
st.markdown("<hr class='divider'>", unsafe_allow_html=True)

model, err = load_model()
if err:
    st.error(f"Model failed to load — {err}")
    st.stop()

left, right = st.columns([1, 1.6], gap="large")

with left:
    st.markdown('<p class="section-label">Enrolled Device</p>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        <div class="stat-row">
            <span class="stat-label">Device</span>
            <span class="stat-value">CC1101 + ESP32</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Modulation</span>
            <span class="stat-value">FSK — 433.92 MHz</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Sample rate</span>
            <span class="stat-value">2,000,000 sps</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Classifier</span>
            <span class="stat-value">1D CNN · 44,577 params</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Test accuracy</span>
            <span class="stat-accent">99.69%</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Rogue recall</span>
            <span class="stat-accent">100.00%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-label">Upload Signal</p>',
                unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "GNU Radio .c64 capture file",
        type=["c64"],
        label_visibility="collapsed"
    )
    st.markdown(
        '<p style="font-size:12px;color:#475569;margin-top:6px">'
        '5 to 12 second recording recommended · Max 200 MB</p>',
        unsafe_allow_html=True
    )

with right:
    st.markdown('<p class="section-label">Authentication Result</p>',
                unsafe_allow_html=True)

    if uploaded is None:
        st.markdown("""
        <div class="result-idle">
            <p style="font-size:32px;margin:0 0 8px 0">📡</p>
            <p style="color:#475569;font-size:15px;margin:0 0 4px 0;font-weight:500">
                Awaiting signal input
            </p>
            <p style="color:#334155;font-size:13px;margin:0">
                Upload a .c64 capture file to begin
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        with st.spinner("Analysing signal..."):
            raw            = uploaded.read()
            arrays, bursts = extract_all_bursts(raw)

        if not arrays:
            st.markdown("""
            <div class="result-warn">
                <p class="result-title-warn">No Signal Found</p>
                <p class="result-sub">No valid bursts detected.
                Ensure the device was transmitting during recording.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            is_auth, conf, probs, std = predict(arrays, model)
            n   = len(arrays)
            dur = len(raw) / (SAMPLE_RATE * 8)

            if conf < CONF_THRESHOLD:
                st.markdown(f"""
                <div class="result-warn">
                    <p class="result-title-warn">Low Confidence</p>
                    <p class="result-sub">System could not make a reliable decision.
                    Try re-recording closer to the receiver.</p>
                    <span class="conf-badge-warn">{conf:.1f}% confidence</span>
                    <div class="meta-row">
                        <span class="meta-item">Bursts: <span>{n}</span></span>
                        <span class="meta-item">Duration: <span>~{dur:.1f}s</span></span>
                        <span class="meta-item">Std dev: <span>{std*100:.1f}%</span></span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            elif is_auth:
                st.markdown(f"""
                <div class="result-auth">
                    <p class="result-title-auth">Device Authenticated</p>
                    <p class="result-sub">Signal fingerprint matches enrolled device</p>
                    <span class="conf-badge-auth">{conf:.1f}% confidence</span>
                    <div class="meta-row">
                        <span class="meta-item">Enrolled: <span>CC1101+ESP32 FSK</span></span>
                        <span class="meta-item">Bursts: <span>{n}</span></span>
                        <span class="meta-item">Duration: <span>~{dur:.1f}s</span></span>
                        <span class="meta-item">Std dev: <span>{std*100:.1f}%</span></span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-deny">
                    <p class="result-title-deny">Unknown Device Detected</p>
                    <p class="result-sub">Signal does not match enrolled device
                    fingerprint — access denied</p>
                    <span class="conf-badge-deny">{conf:.1f}% confidence</span>
                    <div class="meta-row">
                        <span class="meta-item">Bursts: <span>{n}</span></span>
                        <span class="meta-item">Duration: <span>~{dur:.1f}s</span></span>
                        <span class="meta-item">Std dev: <span>{std*100:.1f}%</span></span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<p class="section-label">Signal Fingerprint Analysis</p>',
                        unsafe_allow_html=True)
            fig = make_plot(bursts, probs, is_auth)
            st.pyplot(fig)
            plt.close()

            with st.expander("Technical details"):
                active = int(np.sum(np.abs(bursts[0]) > 0.01))
                st.markdown(f"""
| | |
|---|---|
| File | {uploaded.name} |
| Size | {len(raw)/1e6:.1f} MB |
| Duration | ~{dur:.1f} seconds |
| Bursts extracted | {n} |
| Active samples | {active} ({active/SAMPLE_RATE*1e6:.1f} us) |
| Individual scores | {", ".join([f"{p:.3f}" for p in probs])} |
| Average score | {float(np.mean(probs)):.4f} |
| Threshold | 0.50 score / {CONF_THRESHOLD}% confidence |
                """)

st.markdown("<hr class='divider'>", unsafe_allow_html=True)
st.markdown("""
<p style="font-size:11px;color:#334155;text-align:center">
RF Fingerprinting for Physical Layer Authentication · HKR 2026 ·
RFFPLA_Classifier (1D CNN) · 99.69% test accuracy
</p>
""", unsafe_allow_html=True)
```

---

**What changed from the previous version:**
- `load_model` now uses `tflite_runtime` instead of TensorFlow
- `predict` now runs the TFLite interpreter loop instead of `model.predict()`
- Removed `from tensorflow import keras` import entirely
- All special characters like `µ` replaced with plain `us` to avoid encoding issues

After replacing `app.py` in GitHub, also make sure your `requirements.txt` contains exactly:
```
streamlit
numpy
matplotlib
tflite-runtime

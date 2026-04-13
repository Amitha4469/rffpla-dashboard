import hashlib
import os
import datetime
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from config import WINDOW_SIZE, SAMPLE_RATE, CENTER_FREQ, MODEL_PATH
from preprocess import extract_bursts

# ── Upload security constants ─────────────────────────────────────────────────
_MAX_UPLOAD_BYTES = 200 * 1_048_576   # 200 MiB hard cap — checked before .read()
_MAX_FILENAME_LEN = 255


def _safe_filename(name: str) -> str:
    """Sanitize an uploaded filename for safe display.

    Strips directory separators (prevents path-traversal strings like
    '../../etc/passwd' appearing in the session log or expanders), trims
    to a safe length, and removes non-printable characters.
    """
    name = os.path.basename(name.replace("\\", "/"))
    name = name[:_MAX_FILENAME_LEN].strip()
    name = "".join(c for c in name if c.isprintable())
    return name or "unnamed_file"


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RF-PLA System",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS (single block) ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Outfit:wght@400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }

[data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', monospace !important; }

.conf-num {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 48px;
    font-weight: 600;
    line-height: 1;
    margin: 8px 0 12px;
}

.card-pass { background:#0F3D2E; border:1.5px solid #1D9E75; border-radius:12px; padding:24px; margin-bottom:8px; }
.card-fail { background:#3D1A0C; border:1.5px solid #D85A30; border-radius:12px; padding:24px; margin-bottom:8px; }
.card-warn { background:#1c1400; border:1.5px solid #d97706;  border-radius:12px; padding:24px; margin-bottom:8px; }
.card-idle { background:#0f172a; border:1.5px dashed #334155; border-radius:12px; padding:40px 28px; text-align:center; }

.verdict { font-family:'Outfit',sans-serif; font-size:12px; letter-spacing:0.1em; text-transform:uppercase; font-weight:600; margin-bottom:6px; }
.c-pass  { color:#1D9E75; }
.c-fail  { color:#D85A30; }
.c-warn  { color:#d97706; }

.meta   { display:flex; gap:16px; margin-top:12px; flex-wrap:wrap; font-size:12px; color:#64748b; }
.meta .v { color:#94a3b8; font-weight:500; }

.stTabs [data-baseweb="tab-highlight"] { background-color:#1D9E75 !important; }
.stTabs [aria-selected="true"]         { color:#1D9E75 !important; }

.sec { font-size:11px; font-weight:600; letter-spacing:1px; text-transform:uppercase; color:#475569; margin-bottom:8px; }

.pill-ok  { display:inline-block; background:#0F3D2E; color:#1D9E75; font-size:11px; padding:3px 10px; border-radius:99px; font-weight:600; }
.pill-err { display:inline-block; background:#3D1A0C; color:#D85A30; font-size:11px; padding:3px 10px; border-radius:99px; font-weight:600; }
.tag      { display:inline-block; background:#1e293b; color:#94a3b8; font-size:11px; padding:3px 10px; border-radius:99px; border:1px solid #334155; margin-left:6px; }
</style>
""", unsafe_allow_html=True)

# ── Plotly chart helpers ──────────────────────────────────────────────────────
_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Outfit", size=12),
    margin=dict(l=40, r=20, t=30, b=40),
    template="plotly_dark",
)
_GRID = dict(gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.08)")


def _fig(title=""):
    f = go.Figure()
    f.update_layout(
        **_LAYOUT,
        title=dict(text=title, font=dict(family="Outfit", size=13), x=0),
        xaxis=dict(**_GRID),
        yaxis=dict(**_GRID),
    )
    return f


def chart_amp(burst, is_auth, title="Amplitude envelope", name="Signal"):
    t   = np.arange(len(burst)) / SAMPLE_RATE * 1e6
    amp = np.abs(burst)
    c   = "#1D9E75" if is_auth else "#D85A30"
    fc  = "rgba(29,158,117,0.10)" if is_auth else "rgba(216,90,48,0.10)"
    f   = _fig(title)
    f.add_trace(go.Scatter(
        x=t, y=amp, mode="lines", name=name,
        line=dict(color=c, width=2),
        fill="tozeroy", fillcolor=fc,
        hovertemplate="%{x:.2f} \u00b5s<br>Amp: %{y:.4f}<extra></extra>",
    ))
    f.update_layout(xaxis_title="Time (\u00b5s)", yaxis_title="Amplitude")
    return f


def chart_iq(burst, title="I / Q channels"):
    t = np.arange(len(burst)) / SAMPLE_RATE * 1e6
    f = _fig(title)
    f.add_trace(go.Scatter(
        x=t, y=burst.real, mode="lines", name="I",
        line=dict(color="#1D9E75", width=2),
        hovertemplate="%{x:.2f} \u00b5s<br>I: %{y:.4f}<extra></extra>",
    ))
    f.add_trace(go.Scatter(
        x=t, y=burst.imag, mode="lines", name="Q",
        line=dict(color="#D85A30", width=2),
        hovertemplate="%{x:.2f} \u00b5s<br>Q: %{y:.4f}<extra></extra>",
    ))
    f.update_layout(
        xaxis_title="Time (\u00b5s)", yaxis_title="Amplitude",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return f


def chart_conf(probs, threshold_pct):
    n  = len(probs)
    cf = [(1 - p) * 100 if p < 0.5 else p * 100 for p in probs]
    cs = ["#1D9E75" if p < 0.5 else "#D85A30" for p in probs]
    f  = _fig(f"Per-burst confidence  (n={n})")
    f.add_trace(go.Bar(
        x=list(range(n)), y=cf, marker_color=cs, opacity=0.85,
        hovertemplate="Burst %{x}<br>Confidence: %{y:.1f}%<extra></extra>",
        name="Confidence",
    ))
    f.add_hline(
        y=threshold_pct, line_dash="dash", line_color="#f59e0b",
        annotation_text=f"{threshold_pct:.0f}% threshold",
        annotation_position="top right",
    )
    f.update_layout(xaxis_title="Burst index", yaxis_title="Confidence %", yaxis_range=[0, 108])
    return f


def chart_overlay(burst_a, burst_b):
    ta = np.arange(len(burst_a)) / SAMPLE_RATE * 1e6
    tb = np.arange(len(burst_b)) / SAMPLE_RATE * 1e6
    f  = _fig("Amplitude envelope \u2014 Signal A vs Signal B")
    f.add_trace(go.Scatter(
        x=ta, y=np.abs(burst_a), mode="lines", name="Signal A",
        line=dict(color="#1D9E75", width=2),
        hovertemplate="%{x:.2f} \u00b5s<br>Amp: %{y:.4f}<extra></extra>",
    ))
    f.add_trace(go.Scatter(
        x=tb, y=np.abs(burst_b), mode="lines", name="Signal B",
        line=dict(color="#D85A30", width=2),
        hovertemplate="%{x:.2f} \u00b5s<br>Amp: %{y:.4f}<extra></extra>",
    ))
    f.update_layout(
        xaxis_title="Time (\u00b5s)", yaxis_title="Normalised amplitude",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return f


def _cmp_card(col, sig: str, confidence: float, auth: bool, low: bool,
              n_bursts: int, std: float) -> None:
    """Render a compare-mode result card and progress bar into a Streamlit column."""
    if low:
        html = (
            f'<div class="card-warn">'
            f'<div class="verdict c-warn">Low Confidence \u2014 {sig}</div>'
            f'<div class="conf-num" style="color:#d97706">{confidence:.1f}%</div>'
            f'<div class="meta">'
            f'<span>Bursts: <span class="v">{n_bursts}</span></span>'
            f'<span>Std dev: <span class="v">{std * 100:.1f}%</span></span>'
            f'</div></div>'
        )
    elif auth:
        html = (
            f'<div class="card-pass">'
            f'<div class="verdict c-pass">AUTHORIZED \u2014 {sig}</div>'
            f'<div class="conf-num" style="color:#1D9E75">{confidence:.1f}%</div>'
            f'<div class="meta">'
            f'<span>Bursts: <span class="v">{n_bursts}</span></span>'
            f'<span>Std dev: <span class="v">{std * 100:.1f}%</span></span>'
            f'</div></div>'
        )
    else:
        html = (
            f'<div class="card-fail">'
            f'<div class="verdict c-fail">ACCESS DENIED \u2014 {sig}</div>'
            f'<div class="conf-num" style="color:#D85A30">{confidence:.1f}%</div>'
            f'<div class="meta">'
            f'<span>Bursts: <span class="v">{n_bursts}</span></span>'
            f'<span>Std dev: <span class="v">{std * 100:.1f}%</span></span>'
            f'</div></div>'
        )
    with col:
        st.markdown(html, unsafe_allow_html=True)
        st.progress(min(confidence / 100, 1.0))


# ── Model loading & signal parsing ───────────────────────────────────────────

@st.cache_resource(show_spinner="Loading RFFPLA classifier…")
def load_model():
    # @st.cache_resource is correct here because TFLite Interpreter objects
    # hold native memory (allocated tensor buffers) that cannot be pickled.
    # The decorator keeps a single interpreter instance alive for the entire
    # server process and shares it across all Streamlit sessions — meaning
    # allocate_tensors() is called exactly once no matter how many users or
    # re-runs occur.  show_spinner gives visible feedback on cold start.
    if not os.path.exists(MODEL_PATH):
        return None, (
            f"Model file not found at {MODEL_PATH}. "
            "Check that RFFPLA_classifier.tflite is in the models/ folder."
        )
    try:
        from ai_edge_litert.interpreter import Interpreter
        interpreter = Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        return interpreter, None
    except Exception as e:
        return None, str(e)


def _hash_upload_bytes(b: bytes) -> int:
    """Partial-content hash for large binary uploads used as a @st.cache_data key.

    Streamlit's default bytes hasher walks the entire buffer — ~100–200 ms for
    a 200 MB file — on *every* re-run, even when returning a cache hit.
    This function hashes only: 8-byte little-endian file length + first 64 KiB
    + last 64 KiB, which takes ~0.2 ms and is effectively unique for real .c64
    captures (different recordings always differ in at least one of those regions).
    """
    chunk = 65_536  # 64 KiB
    h = hashlib.blake2b(digest_size=16)
    h.update(len(b).to_bytes(8, "little"))
    h.update(b[:chunk])
    if len(b) > chunk:
        h.update(b[-chunk:])
    return int.from_bytes(h.digest(), "little")


@st.cache_data(
    show_spinner=False,          # caller already owns a st.spinner context
    max_entries=10,              # bound cached outputs to ~3.2 MB total (see below)
    hash_funcs={bytes: _hash_upload_bytes},  # use fast partial hash, not full scan
)
def _parse_upload(raw: bytes):
    """Extract bursts from a .c64 upload once; serve the cached result on re-runs.

    Every Streamlit interaction — slider move, tab switch, widget click — triggers
    a full script re-run.  Without this cache, extract_bursts() repeats on every
    interaction: that means np.convolve() over up to 25 M samples (~1–2 s) each
    time the user adjusts the confidence slider.

    @st.cache_data serialises only the *outputs* (X, bursts, info) via pickle,
    not the raw bytes, so ``del raw`` in the caller still frees the ~200 MB
    upload buffer immediately after the first parse.

    Memory budget for max_entries=10:
        X arrays   : 10 × 20 bursts × 1 024 samples × 2 ch × 4 B ≈ 1.6 MB
        burst lists: 10 × 20 × 1 024 × 8 B (complex64)            ≈ 1.6 MB
        Total                                                       ≈ 3.2 MB
    — bounded regardless of the original 200 MB file sizes.
    """
    return extract_bursts(raw)


def predict(arrays, model):
    input_details  = model.get_input_details()
    output_details = model.get_output_details()
    probs = []
    for arr in arrays:
        inp = arr[np.newaxis, :, :].astype(np.float32)
        model.set_tensor(input_details[0]["index"], inp)
        model.invoke()
        p = float(model.get_tensor(output_details[0]["index"])[0][0])
        probs.append(p)
    probs   = np.array(probs)
    avg     = float(np.mean(probs))
    std     = float(np.std(probs))
    is_auth = avg < 0.5
    conf    = (1 - avg) * 100 if is_auth else avg * 100
    return is_auth, conf, probs, std


@st.cache_data(show_spinner=False, max_entries=20)
def _run_predict(X: np.ndarray) -> tuple:
    """Run TFLite inference once per unique burst set; cache across re-runs.

    Without this cache, all 20 inference passes repeat on every slider move
    or tab switch even though the signal data has not changed.  Combined with
    _parse_upload, the caching chain is: parse once → infer once → only
    threshold-dependent UI (the result card, progress bar, chart_conf) rebuilds
    on each interaction.

    load_model() is @st.cache_resource so the call here is a free no-op after
    the first load — it never re-allocates tensors or re-reads the .tflite file.
    """
    interp, _ = load_model()
    return predict(X, interp)


# ── Session state ─────────────────────────────────────────────────────────────
if "session_log" not in st.session_state:
    st.session_state.session_log = []

# ── Load model (global — both tabs need it) ───────────────────────────────────
model, model_err = load_model()
model_ok = model is not None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Enrolled Device")
    st.markdown(f"""
| Parameter | Value |
|---|---|
| Device | CC1101 + ESP32 |
| Modulation | FSK |
| Frequency | {CENTER_FREQ / 1e6:.2f} MHz |
| Sample rate | {int(SAMPLE_RATE / 1e6)} Msps |
| Classifier | 1D CNN |
| Parameters | 44,577 |
""")
    st.divider()
    st.markdown("### Model Performance")
    st.markdown("""
| Metric | Value |
|---|---|
| Test accuracy | 99.69% |
| Rogue recall | 100.00% |
""")
    st.divider()
    st.markdown("### About")
    st.caption("RF Fingerprinting for Physical Layer Authentication · HKR 2026.")

# ── Header ────────────────────────────────────────────────────────────────────
h_left, h_right = st.columns([3, 1])
with h_left:
    st.markdown("""
<p style="font-family:'Outfit',sans-serif;font-size:28px;font-weight:700;margin:0;color:#f1f5f9">
\U0001f6e1 RF Physical Layer Authentication
</p>
<p style="font-family:'Outfit',sans-serif;font-size:14px;color:#64748b;margin:4px 0 0 0">
Physical Layer Authentication System \u2014 1D Convolutional Neural Network
</p>
""", unsafe_allow_html=True)

with h_right:
    pill = (
        '<span class="pill-ok">\u25cf Model loaded</span>'
        if model_ok else
        '<span class="pill-err">\u25cf Model not found</span>'
    )
    st.markdown(
        f'<div style="text-align:right;padding-top:8px">'
        f'{pill}'
        f'<span class="tag">CC1101 \u00b7 FSK \u00b7 {CENTER_FREQ / 1e6:.2f} MHz</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

st.divider()

# ── Summary stats bar ─────────────────────────────────────────────────────────
_log     = st.session_state.session_log
_n_files = len(_log)
_n_pass  = sum(1 for e in _log if e["result"] == "PASS")
_pct     = f"{_n_pass / _n_files * 100:.0f}%" if _n_files else "\u2014"

s1, s2, s3, s4 = st.columns(4)
s1.metric("Test accuracy",      "99.69%")
s2.metric("Rogue recall",       "100.00%")
s3.metric("Files this session", str(_n_files))
s4.metric("Session pass rate",  _pct)

st.divider()

if not model_ok:
    st.error(model_err)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Authentication", "Compare mode"])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — Authentication
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    left, right = st.columns([1, 1.6], gap="large")

    with left:
        st.markdown('<p class="sec">Upload Signal</p>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "GNU Radio .c64 capture file",
            type=["c64"],
            label_visibility="collapsed",
            key="up1",
        )
        st.caption("5\u201312 second recording recommended \u00b7 Max 200 MB")
        st.divider()
        DISPLAY_THRESHOLD = st.slider(
            "Confidence threshold", 0.50, 0.95, 0.70, 0.05,
            help="Minimum confidence required for a PASS decision",
            key="thr1",
        )

    with right:
        st.markdown('<p class="sec">Authentication Result</p>', unsafe_allow_html=True)

        if uploaded is None:
            st.markdown("""
<div class="card-idle">
  <p style="font-size:32px;margin:0 0 8px 0">\U0001f6e1</p>
  <p style="color:#475569;font-size:15px;margin:0 0 4px 0;font-weight:500">Awaiting signal input</p>
  <p style="color:#334155;font-size:13px;margin:0">Upload a .c64 capture file to begin</p>
</div>
""", unsafe_allow_html=True)
        else:
            safe_name = _safe_filename(uploaded.name)

            # ── Size guard: check before .read() to prevent loading into memory ──
            if uploaded.size > _MAX_UPLOAD_BYTES:
                st.error(
                    f"File too large ({uploaded.size / 1_048_576:.1f} MiB). "
                    "Maximum is 200 MB — try a shorter recording."
                )
            else:
                raw        = uploaded.read()
                file_bytes = len(raw)            # store size before del
                n_samp     = file_bytes // 8     # complex64 = 8 bytes

                if n_samp < 1000:
                    st.error(
                        "File too small. Minimum recommended recording is 5 seconds at 2 Msps."
                    )
                elif not model_ok:
                    st.error(model_err)
                else:
                    with st.spinner("Analysing signal..."):
                        X, bursts, info = _parse_upload(raw)
                    del raw  # release raw bytes (~200 MB) — cache stores outputs only

                    if not len(X):
                        st.error(
                            "No signal burst detected. Try a longer recording or check "
                            "that the squelch threshold is not too high."
                        )
                    else:
                        is_auth, conf, probs, std = _run_predict(X)
                        n   = len(X)
                        dur = info["duration_s"]
                        thr = DISPLAY_THRESHOLD * 100
                        low = conf < thr

                        if low:
                            st.markdown(f"""
<div class="card-warn">
  <div class="verdict c-warn">Low Confidence</div>
  <div class="conf-num" style="color:#d97706">{conf:.1f}%</div>
  <p style="font-size:13px;color:#94a3b8;margin:0 0 8px">
    System could not make a reliable decision.
    Try re-recording closer to the receiver.
  </p>
  <div class="meta">
    <span>Bursts: <span class="v">{n}</span></span>
    <span>Duration: <span class="v">~{dur:.1f}s</span></span>
    <span>Std dev: <span class="v">{std * 100:.1f}%</span></span>
  </div>
</div>
""", unsafe_allow_html=True)
                        elif is_auth:
                            st.markdown(f"""
<div class="card-pass">
  <div class="verdict c-pass">AUTHORIZED</div>
  <div class="conf-num" style="color:#1D9E75">{conf:.1f}%</div>
  <p style="font-size:13px;color:#94a3b8;margin:0 0 8px">
    Signal fingerprint matches enrolled device
  </p>
  <div class="meta">
    <span>Enrolled: <span class="v">CC1101+ESP32 FSK</span></span>
    <span>Bursts: <span class="v">{n}</span></span>
    <span>Duration: <span class="v">~{dur:.1f}s</span></span>
    <span>Std dev: <span class="v">{std * 100:.1f}%</span></span>
  </div>
</div>
""", unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
<div class="card-fail">
  <div class="verdict c-fail">ACCESS DENIED</div>
  <div class="conf-num" style="color:#D85A30">{conf:.1f}%</div>
  <p style="font-size:13px;color:#94a3b8;margin:0 0 8px">
    Signal does not match enrolled device fingerprint
  </p>
  <div class="meta">
    <span>Bursts: <span class="v">{n}</span></span>
    <span>Duration: <span class="v">~{dur:.1f}s</span></span>
    <span>Std dev: <span class="v">{std * 100:.1f}%</span></span>
  </div>
</div>
""", unsafe_allow_html=True)

                        st.progress(min(conf / 100, 1.0))

                        # Append to session log (deduplicate same file across reruns)
                        result_str = "PASS" if (is_auth and not low) else "FAIL"
                        entry = {
                            "time":       datetime.datetime.now().strftime("%H:%M"),
                            "file":       safe_name,
                            "result":     result_str,
                            "confidence": round(conf, 1),
                            "bursts":     n,
                        }
                        log = st.session_state.session_log
                        if not log or log[0]["file"] != safe_name:
                            st.session_state.session_log.insert(0, entry)

                        st.divider()
                        st.markdown('<p class="sec">Signal Fingerprint Analysis</p>',
                                    unsafe_allow_html=True)

                        ca, cb, cc = st.columns(3)
                        with ca:
                            st.plotly_chart(chart_amp(bursts[0], is_auth),
                                            use_container_width=True)
                            st.caption(
                                f"Captured at {CENTER_FREQ / 1e6:.2f} MHz \u00b7 "
                                f"{int(SAMPLE_RATE / 1e6)} Msps \u00b7 "
                                f"{WINDOW_SIZE}-sample window"
                            )
                        with cb:
                            st.plotly_chart(chart_iq(bursts[0]), use_container_width=True)
                        with cc:
                            st.plotly_chart(chart_conf(probs, thr), use_container_width=True)

                        with st.expander("Technical details"):
                            active = int(np.sum(np.abs(bursts[0]) > 0.01))
                            st.markdown(f"""
| | |
|---|---|
| File | `{safe_name}` |
| Size | {file_bytes / 1e6:.1f} MB |
| Duration | ~{dur:.1f} s |
| Bursts extracted | {n} |
| Sample count | {n_samp:,} |
| Active samples | {active} ({active / SAMPLE_RATE * 1e6:.1f} \u00b5s) |
| Individual scores | {", ".join(f"{p:.3f}" for p in probs)} |
| Average score | {float(np.mean(probs)):.4f} |
| Threshold (display) | {thr:.0f}% confidence |
| Threshold (model) | 0.50 score |
""")

    # Session log (full width, below the two columns)
    st.divider()
    st.markdown('<p class="sec">Session Log</p>', unsafe_allow_html=True)
    if not st.session_state.session_log:
        st.caption("No files processed yet this session.")
    else:
        df = pd.DataFrame(
            st.session_state.session_log,
            columns=["time", "file", "result", "confidence", "bursts"],
        )
        df.columns = ["Time", "File", "Result", "Confidence%", "Bursts"]
        df["Result"] = df["Result"].apply(
            lambda r: "\u2705 PASS" if r == "PASS" else "\u274c FAIL"
        )
        st.dataframe(df, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — Compare mode
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<p class="sec">Compare two signal captures</p>',
                unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Signal A \u2014 authorized device**")
        file_a = st.file_uploader(
            "Signal A", type=["c64"], label_visibility="collapsed", key="cmp_a"
        )
    with col_b:
        st.markdown("**Signal B \u2014 rogue / unknown device**")
        file_b = st.file_uploader(
            "Signal B", type=["c64"], label_visibility="collapsed", key="cmp_b"
        )

    if file_a is None or file_b is None:
        st.info("Upload both Signal A and Signal B to run comparison.")
    elif not model_ok:
        st.error(model_err)
    else:
        size_err = False

        # ── Size guards: check before .read() to prevent loading into memory ──
        if file_a.size > _MAX_UPLOAD_BYTES:
            st.error(
                f"Signal A too large ({file_a.size / 1_048_576:.1f} MiB). "
                "Maximum is 200 MB."
            )
            size_err = True
        if file_b.size > _MAX_UPLOAD_BYTES:
            st.error(
                f"Signal B too large ({file_b.size / 1_048_576:.1f} MiB). "
                "Maximum is 200 MB."
            )
            size_err = True

        if not size_err:
            # Process sequentially so only one ~200 MB raw buffer is in memory
            # at a time — halves peak RAM vs. reading both files simultaneously.
            with st.spinner("Analysing both signals..."):
                raw_a = file_a.read()
                na    = len(raw_a) // 8
                if na < 1000:
                    st.error("Signal A: File too small. Minimum recommended recording is 5 seconds at 2 Msps.")
                    del raw_a
                    size_err = True
                else:
                    X_a, bursts_a, info_a = _parse_upload(raw_a)
                    del raw_a  # free Signal A buffer before loading Signal B

                if not size_err:
                    raw_b = file_b.read()
                    nb    = len(raw_b) // 8
                    if nb < 1000:
                        st.error("Signal B: File too small. Minimum recommended recording is 5 seconds at 2 Msps.")
                        del raw_b
                        size_err = True
                    else:
                        X_b, bursts_b, info_b = _parse_upload(raw_b)
                        del raw_b  # free Signal B buffer

            # Guard: X_a / X_b are only defined when their respective file passed
            # both the size guard and the min-sample check above.  If size_err was
            # set inside the spinner (na < 1000 or nb < 1000), referencing X_a/X_b
            # here would raise NameError.  Propagating size_err into burst_err
            # skips all downstream references safely.
            burst_err = size_err
            if not size_err:
                if not len(X_a):
                    st.error("Signal A: No signal burst detected. Try a longer recording or check that the squelch threshold is not too high.")
                    burst_err = True
                if not len(X_b):
                    st.error("Signal B: No signal burst detected. Try a longer recording or check that the squelch threshold is not too high.")
                    burst_err = True

            if not burst_err:
                auth_a, conf_a, probs_a, std_a = _run_predict(X_a)
                auth_b, conf_b, probs_b, std_b = _run_predict(X_b)
                dur_a = info_a["duration_s"]
                dur_b = info_b["duration_s"]

                CMP_THR = st.slider(
                    "Confidence threshold (compare)", 0.50, 0.95, 0.70, 0.05,
                    key="thr2",
                ) * 100
                low_a  = conf_a < CMP_THR
                low_b  = conf_b < CMP_THR
                pass_a = auth_a and not low_a
                pass_b = auth_b and not low_b

                rc_a, rc_b = st.columns(2)
                _cmp_card(rc_a, "Signal A", conf_a, auth_a, low_a, len(X_a), std_a)
                _cmp_card(rc_b, "Signal B", conf_b, auth_b, low_b, len(X_b), std_b)

                st.divider()

                # ── Amplitude overlay ─────────────────────────────────────────
                st.plotly_chart(
                    chart_overlay(bursts_a[0], bursts_b[0]),
                    use_container_width=True,
                )
                st.caption(
                    f"Captured at {CENTER_FREQ / 1e6:.2f} MHz \u00b7 "
                    f"{int(SAMPLE_RATE / 1e6)} Msps \u00b7 "
                    f"{WINDOW_SIZE}-sample window"
                )

                st.divider()

                # ── Side-by-side I/Q ──────────────────────────────────────────
                st.markdown('<p class="sec">I / Q Channels</p>',
                            unsafe_allow_html=True)
                iq_ca, iq_cb = st.columns(2)
                with iq_ca:
                    st.plotly_chart(
                        chart_iq(bursts_a[0], "Signal A \u2014 I / Q"),
                        use_container_width=True,
                    )
                with iq_cb:
                    st.plotly_chart(
                        chart_iq(bursts_b[0], "Signal B \u2014 I / Q"),
                        use_container_width=True,
                    )

                st.divider()

                # ── Difference summary table ───────────────────────────────────
                st.markdown('<p class="sec">Comparison Summary</p>',
                            unsafe_allow_html=True)
                v_a = "PASS" if pass_a else ("LOW CONF" if low_a else "FAIL")
                v_b = "PASS" if pass_b else ("LOW CONF" if low_b else "FAIL")
                st.table(pd.DataFrame({
                    "Metric": [
                        "Confidence", "Decision", "Bursts extracted",
                        "Burst std dev", "File size", "Duration",
                    ],
                    "Signal A": [
                        f"{conf_a:.1f}%", v_a, str(len(X_a)),
                        f"{std_a * 100:.1f}%",
                        f"{file_a.size / 1e6:.1f} MB",
                        f"~{dur_a:.1f}s",
                    ],
                    "Signal B": [
                        f"{conf_b:.1f}%", v_b, str(len(X_b)),
                        f"{std_b * 100:.1f}%",
                        f"{file_b.size / 1e6:.1f} MB",
                        f"~{dur_b:.1f}s",
                    ],
                }))

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<p style="font-size:11px;color:#334155;text-align:center;font-family:'Outfit',sans-serif">
RF Fingerprinting for Physical Layer Authentication \u00b7 HKR 2026 \u00b7
RFFPLA_Classifier (1D CNN) \u00b7 99.69% test accuracy
</p>
""", unsafe_allow_html=True)

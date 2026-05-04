import hashlib
import os
import datetime
import time

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from ai_edge_litert.interpreter import Interpreter

from config import WINDOW_SIZE, SAMPLE_RATE, CENTER_FREQ, MODEL_PATH
from preprocess import extract_bursts

# ── Upload security constants ─────────────────────────────────────────────────
_MAX_UPLOAD_BYTES = 200 * 1_048_576
_MAX_FILENAME_LEN = 255


def _safe_filename(name: str) -> str:
    name = os.path.basename(name.replace("\\", "/"))
    name = name[:_MAX_FILENAME_LEN].strip()
    name = "".join(c for c in name if c.isprintable())
    return name or "unnamed_file"


# ── 2-channel window helper (v5 — power-normalised IQ) ───────────────────────

def iq_to_2ch_window(iq_chunk: np.ndarray):
    p = np.mean(np.abs(iq_chunk)**2)
    if not np.isfinite(p) or p <= 1e-12:
        return None
    iq_norm = iq_chunk / (np.sqrt(p) + 1e-12)
    window = np.stack([iq_norm.real, iq_norm.imag], axis=-1)
    return window.astype(np.float32)  # shape (WINDOW_SIZE, 2)


# ── Theme definitions ─────────────────────────────────────────────────────────

_THEMES = {
    "green": dict(
        signal="#00e87a", bg_card="#0d1e15", bg_surface="#091510",
        bg_root="#030a06", bg_main="#0a1a0f", border="#163220",
        text="#a8e0bc", text_muted="#366048",
        signal_glow="rgba(0,232,122,0.35)",
        signal_scanline="rgba(0,232,122,0.15)",
        scanline_fill="rgba(0,232,122,0.25)",
    ),
    "amber": dict(
        signal="#f09c28", bg_card="#1a1205", bg_surface="#100c03",
        bg_root="#080601", bg_main="#120f04", border="#3a2a0a",
        text="#e0cfa8", text_muted="#6b4e10",
        signal_glow="rgba(240,156,40,0.35)",
        signal_scanline="rgba(240,156,40,0.15)",
        scanline_fill="rgba(240,156,40,0.25)",
    ),
    "red": dict(
        signal="#e04040", bg_card="#1a0808", bg_surface="#0f0505",
        bg_root="#080202", bg_main="#110404", border="#3a1212",
        text="#e0a8a8", text_muted="#6b2020",
        signal_glow="rgba(224,64,64,0.35)",
        signal_scanline="rgba(224,64,64,0.15)",
        scanline_fill="rgba(224,64,64,0.25)",
    ),
}


def _conf_to_theme(conf: float, is_auth: bool) -> str:
    if not is_auth:
        return "red"
    if conf >= 85.0:
        return "green"
    elif conf >= 50.0:
        return "amber"
    return "red"


def get_theme_css(theme_name: str) -> str:
    t = _THEMES.get(theme_name, _THEMES["green"])
    s     = t["signal"]
    bg_c  = t["bg_card"]
    bg_s  = t["bg_surface"]
    bg_r  = t["bg_root"]
    brd   = t["border"]
    txt   = t["text"]
    muted = t["text_muted"]

    red_extras = ""
    if theme_name == "red":
        red_extras = """
    body::after {
        content: '';
        position: fixed;
        inset: 0;
        pointer-events: none;
        background: radial-gradient(ellipse at center,
                    transparent 40%, rgba(224,64,64,0.08) 100%);
        z-index: 9999;
        animation: threat-pulse 2s ease-in-out infinite;
    }
    @keyframes threat-pulse { 0%,100% { opacity: 0.6 } 50% { opacity: 1.0 } }
    .rf-header { border-left: 3px solid #e04040 !important; }
    .rf-threat {
        color: #e04040 !important;
        font-size: 7px;
        letter-spacing: .3em;
        animation: blink-threat 0.8s step-end infinite;
        white-space: nowrap;
    }
    @keyframes blink-threat { 0%,100% { opacity: 1 } 50% { opacity: 0 } }
    """

    return f"""<style>
    @import url('https://cdn.jsdelivr.net/npm/@fontsource/space-mono/400.css');

    /* ── Theme override ──────────────────────────────────────────────────── */
    html, body {{ background: {bg_r} !important; transition: background 0.6s ease; }}
    [data-testid="stAppViewContainer"] {{ background: {bg_r} !important; transition: background 0.6s ease; }}
    [data-testid="stMain"] {{ background: {bg_r} !important; transition: background 0.6s ease; }}
    .main .block-container {{ padding-top: 0 !important; }}
    [data-testid="stSidebar"] {{
        background: {bg_s} !important; border-right: 1px solid {brd} !important;
        transition: background 0.6s ease, border-color 0.6s ease;
    }}
    section[data-testid="stSidebar"] > div {{ background: {bg_s} !important; transition: background 0.6s ease; }}
    header[data-testid="stHeader"] {{
        background: {bg_s} !important; border-bottom: 1px solid {brd} !important;
        transition: background 0.6s ease, border-color 0.6s ease;
    }}
    [data-testid="stToolbar"] {{ display: none !important; }}
    [data-testid="stDecoration"] {{ display: none !important; }}

    /* ── Global font ─────────────────────────────────────────────────────── */
    *, *::before, *::after {{
        font-family: 'Space Mono', 'Courier New', monospace !important;
        color: {txt};
        transition: color 0.5s ease;
    }}

    /* ── Scrollbar ───────────────────────────────────────────────────────── */
    ::-webkit-scrollbar {{ width: 4px; height: 4px; }}
    ::-webkit-scrollbar-track {{ background: {bg_r}; }}
    ::-webkit-scrollbar-thumb {{ background: {brd}; border-radius: 2px; }}

    /* ── Header ──────────────────────────────────────────────────────────── */
    .rf-header {{
        background: {bg_s};
        border-bottom: 1px solid {brd};
        padding: 10px 28px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin: -4rem -4rem 1.5rem -4rem;
        width: calc(100% + 8rem);
        transition: background 0.6s ease, border-color 0.6s ease, box-shadow 0.6s ease;
    }}
    .rf-logo {{
        color: {s} !important;
        font-size: 13px;
        letter-spacing: .45em;
        white-space: nowrap;
        transition: color 0.5s ease;
    }}
    .rf-center {{
        color: {muted} !important;
        font-size: 7px;
        letter-spacing: .35em;
        text-transform: uppercase;
        text-align: center;
        flex: 1;
        padding: 0 24px;
        transition: color 0.5s ease;
    }}
    .rf-status {{
        display: flex;
        align-items: center;
        gap: 8px;
        white-space: nowrap;
    }}
    .rf-status-text {{
        color: {txt} !important;
        font-size: 7px;
        letter-spacing: .3em;
        transition: color 0.5s ease;
    }}
    .rf-dot {{
        width: 6px; height: 6px;
        background: {s};
        border-radius: 50%;
        flex-shrink: 0;
        animation: rfpulse 1.6s ease-in-out infinite;
        transition: background 0.5s ease;
    }}
    @keyframes rfpulse {{ 0%, 100% {{ opacity: .35; }} 50% {{ opacity: 1; }} }}

    /* ── Panel headers ───────────────────────────────────────────────────── */
    .panel-hdr {{
        color: {muted} !important;
        font-size: 7px;
        letter-spacing: .42em;
        text-transform: uppercase;
        border-bottom: 1px solid {brd};
        padding-bottom: 8px;
        margin-bottom: 14px;
        transition: color 0.5s ease, border-color 0.6s ease;
    }}

    /* ── Result card ─────────────────────────────────────────────────────── */
    .result-card {{
        border-radius: 4px;
        padding: 20px 16px 16px;
        margin: 0 0 10px;
        text-align: center;
        transition: background 0.6s ease, border-color 0.6s ease;
    }}
    .result-badge {{
        display: inline-block;
        font-size: 10px;
        letter-spacing: .42em;
        text-transform: uppercase;
        border: 1px solid;
        border-radius: 2px;
        padding: 2px 10px;
        margin-bottom: 10px;
    }}
    .result-confval {{
        font-size: clamp(20px, 4vw, 34px);
        letter-spacing: .05em;
        line-height: 1;
        display: block;
        margin: 8px 0 4px;
    }}
    @keyframes blink {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0; }} }}
    .cursor {{ animation: blink step-end .9s infinite; display: inline-block; }}
    .result-label {{
        font-size: 10px !important;
        letter-spacing: .42em;
        text-transform: uppercase;
        display: block;
        margin-bottom: 12px;
    }}
    .conf-bar-wrap {{
        width: 100%;
        height: 3px;
        background: {brd};
        border-radius: 1px;
        margin-top: 12px;
        transition: background 0.6s ease;
    }}
    .conf-bar {{
        height: 3px;
        border-radius: 1px;
        transition: width 0.55s ease, background 0.6s ease;
    }}

    /* ── Metadata grid ───────────────────────────────────────────────────── */
    .meta-grid {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 8px 4px;
        margin-top: 12px;
    }}
    .meta-lbl {{
        font-size: 6.5px !important;
        letter-spacing: .3em;
        text-transform: uppercase;
        color: {muted} !important;
        transition: color 0.5s ease;
    }}
    .meta-val {{
        font-size: 10px !important;
        color: {txt} !important;
        text-align: right;
        transition: color 0.5s ease;
    }}

    /* ── Session log ─────────────────────────────────────────────────────── */
    .session-wrap {{
        max-height: 180px;
        overflow-y: auto;
        border: 1px solid {brd};
        border-radius: 2px;
        background: {bg_c};
        transition: background 0.6s ease, border-color 0.6s ease;
    }}
    .session-tbl {{ width: 100%; border-collapse: collapse; }}
    .session-tbl th {{
        font-size: 6.5px !important;
        letter-spacing: .3em;
        text-transform: uppercase;
        color: {muted} !important;
        padding: 6px 8px;
        background: {bg_s};
        border-bottom: 1px solid {brd};
        text-align: left;
        transition: color 0.5s ease, background 0.6s ease, border-color 0.6s ease;
    }}
    .session-tbl td {{
        padding: 5px 8px;
        font-size: 0.7rem !important;
        border-bottom: 1px solid {brd};
        color: {txt} !important;
        transition: color 0.5s ease, border-color 0.6s ease;
    }}
    .session-tbl tr:nth-child(even) td {{ background: {bg_s}; }}
    .session-tbl tr:nth-child(odd)  td {{ background: {bg_c}; }}
    .row-pass td {{ color: #00e87a !important; }}
    .row-fail td {{ color: #e04040 !important; }}

    /* ── Idle / Analyzing ────────────────────────────────────────────────── */
    .idle-card {{
        background: {bg_c};
        border: 1px solid {brd};
        border-radius: 4px;
        padding: 40px 24px;
        text-align: center;
        color: {muted} !important;
        font-size: 7px;
        letter-spacing: .42em;
        text-transform: uppercase;
        transition: background 0.6s ease, border-color 0.6s ease, color 0.5s ease;
    }}
    .analyzing-card {{
        background: {bg_c};
        border: 1px solid {brd};
        border-radius: 4px;
        padding: 32px;
        text-align: center;
        font-size: 7px;
        letter-spacing: .42em;
        color: {muted} !important;
        text-transform: uppercase;
        margin: 10px 0;
        transition: background 0.6s ease, border-color 0.6s ease, color 0.5s ease;
    }}
    @keyframes scan {{
        0%, 100% {{ opacity: .3; transform: scaleX(.3); }}
        50% {{ opacity: 1; transform: scaleX(1); }}
    }}
    .scan-bar {{
        width: 60%;
        height: 2px;
        background: linear-gradient(90deg, transparent, {s}, transparent);
        margin: 14px auto 0;
        animation: scan 1.2s ease-in-out infinite;
    }}

    /* ── Compare VS ──────────────────────────────────────────────────────── */
    .vs-wrap {{
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 100%;
        min-height: 200px;
        gap: 0;
    }}
    .vs-line {{ width: 1px; background: {brd}; flex: 1; min-height: 40px; transition: background 0.6s ease; }}
    .vs-txt {{
        color: {muted} !important;
        font-size: 10px;
        letter-spacing: .2em;
        padding: 8px 0;
        transition: color 0.5s ease;
    }}

    /* ── Sidebar ─────────────────────────────────────────────────────────── */
    .sb-hdr {{
        color: {muted} !important;
        font-size: 7px;
        letter-spacing: .42em;
        text-transform: uppercase;
        border-bottom: 1px solid {brd};
        padding-bottom: 6px;
        margin-bottom: 10px;
        display: block;
        transition: color 0.5s ease, border-color 0.6s ease;
    }}
    .sb-tbl {{ width: 100%; border-collapse: collapse; }}
    .sb-tbl td {{
        padding: 4px 2px;
        color: {txt} !important;
        border: none;
        font-size: 0.75rem !important;
        vertical-align: top;
        transition: color 0.5s ease;
    }}
    .sb-tbl td:first-child {{
        color: {muted} !important;
        font-size: 6.5px !important;
        letter-spacing: .2em;
        text-transform: uppercase;
        width: 50%;
        padding-top: 6px;
        transition: color 0.5s ease;
    }}
    .armed {{
        display: flex; align-items: center; gap: 8px;
        color: #00e87a !important; font-size: 8px; letter-spacing: .3em; margin-top: 6px;
    }}
    .offline {{
        display: flex; align-items: center; gap: 8px;
        color: #e04040 !important; font-size: 8px; letter-spacing: .3em; margin-top: 6px;
    }}
    .uncertain-status {{
        display: flex; align-items: center; gap: 8px;
        color: #f09c28 !important; font-size: 8px; letter-spacing: .3em; margin-top: 6px;
    }}
    .threat-status {{
        display: flex; align-items: center; gap: 8px;
        color: #e04040 !important; font-size: 8px; letter-spacing: .3em; margin-top: 6px;
    }}
    .armed-dot     {{ width:6px; height:6px; background:#00e87a; border-radius:50%; animation:rfpulse 1.6s ease-in-out infinite; }}
    .offline-dot   {{ width:6px; height:6px; background:#e04040; border-radius:50%; }}
    .uncertain-dot {{ width:6px; height:6px; background:#f09c28; border-radius:50%; animation:rfpulse 3s ease-in-out infinite; }}
    .threat-dot    {{ width:6px; height:6px; background:#e04040; border-radius:50%; animation:blink-threat 0.8s step-end infinite; }}

    /* ── Footer ──────────────────────────────────────────────────────────── */
    .rf-footer {{
        background: {bg_s};
        border-top: 1px solid {brd};
        padding: 10px;
        text-align: center;
        font-size: 7px;
        letter-spacing: .3em;
        color: {muted} !important;
        text-transform: uppercase;
        margin-top: 28px;
        transition: background 0.6s ease, border-color 0.6s ease, color 0.5s ease;
    }}

    /* ── Chart containers ────────────────────────────────────────────────── */
    .js-plotly-plot {{ border: 1px solid {brd}; border-radius: 4px; transition: border-color 0.6s ease; }}

    /* ── Chart section ───────────────────────────────────────────────────── */
    .chart-title-dot {{
        width: 5px; height: 5px; background: {s}; border-radius: 50%;
        display: inline-block; animation: rfpulse 1.6s ease-in-out infinite;
        flex-shrink: 0; margin-right: 6px; vertical-align: middle;
        transition: background 0.5s ease;
    }}
    .chart-lbl {{
        font-size: 8px !important; letter-spacing: .42em;
        text-transform: uppercase; color: {s} !important; vertical-align: middle;
        transition: color 0.5s ease;
    }}
    .chart-desc {{
        font-size: 7px !important; color: {muted} !important;
        letter-spacing: .2em; display: block; margin-top: 3px; margin-bottom: 0;
        transition: color 0.5s ease;
    }}
    .chart-card {{
        background: {bg_c}; border: 1px solid {brd}; border-radius: 4px;
        padding: 0 0 2px 0; overflow: hidden; margin-bottom: 4px;
        transition: background 0.6s ease, border-color 0.6s ease;
    }}
    .chart-accent {{ height: 3px; width: 100%; display: block; }}
    .chart-card-body {{ padding: 10px 12px 4px 12px; }}

    /* ── Streamlit widget overrides ──────────────────────────────────────── */
    .stFileUploader label,
    .stSlider label,
    .stExpander summary,
    p, span, div {{ color: {txt}; transition: color 0.5s ease; }}
    .stTabs [data-baseweb="tab-list"] {{
        border-bottom: 1px solid {brd} !important;
        gap: 0;
        background: transparent !important;
        transition: border-color 0.6s ease;
    }}
    .stTabs [data-baseweb="tab"] {{
        background: transparent !important;
        color: {muted} !important;
        font-size: 7px !important;
        letter-spacing: .35em !important;
        text-transform: uppercase;
        border: none !important;
        padding: 8px 20px !important;
        transition: color 0.5s ease;
    }}
    .stTabs [aria-selected="true"] {{
        color: {s} !important;
        border-bottom: 2px solid {s} !important;
        background: transparent !important;
        transition: color 0.5s ease, border-color 0.6s ease;
    }}
    [data-testid="stExpander"] {{
        border: 1px solid {brd} !important;
        border-radius: 4px !important;
        background: {bg_c} !important;
        transition: border-color 0.6s ease, background 0.6s ease;
    }}
    .stAlert {{ border-radius: 4px !important; }}

    {red_extras}
    </style>"""


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RF-PLA System",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state (must be before CSS injection) ──────────────────────────────
if "session_log" not in st.session_state:
    st.session_state.session_log = []
if "last_hash" not in st.session_state:
    st.session_state.last_hash = ""
if "theme" not in st.session_state:
    st.session_state.theme = "green"
st.cache_data.clear()

# ── Inject themed CSS ─────────────────────────────────────────────────────────
st.markdown(get_theme_css(st.session_state.theme), unsafe_allow_html=True)


# ── TFLite model loader ───────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    try:
        interpreter = Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        print("TFLite input shape:", input_details[0]["shape"])
        return interpreter, True, ""
    except Exception as e:
        return None, False, f"Model error: {e!s}"


# ── Prediction helpers ────────────────────────────────────────────────────────

def predict(X, interpreter):
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    probs = []
    for i in range(len(X)):
        inp = X[i:i+1].astype(np.float32)
        interpreter.set_tensor(input_details[0]["index"], inp)
        interpreter.invoke()
        out = interpreter.get_tensor(output_details[0]["index"])
        probs.append(float(out[0][0]))
    probs       = np.array(probs)
    mean_output = float(np.mean(probs))
    std_conf    = float(np.std(probs))
    is_auth     = mean_output > 0.70                    # v5: high score = AUTH
    # auth: confidence = raw auth score; rogue: confidence = 1 - score
    display_conf = (mean_output if is_auth else (1.0 - mean_output)) * 100.0
    return is_auth, display_conf, probs, std_conf


def _run_predict(X: np.ndarray):
    interpreter, ok, err = load_model()
    if not ok or interpreter is None:
        raise RuntimeError(err)
    return predict(X, interpreter)


# ── Plotly chart helpers ──────────────────────────────────────────────────────

def _fig(title: str = "", theme: str = "green"):
    t = _THEMES.get(theme, _THEMES["green"])
    f = go.Figure()
    f.update_layout(
        paper_bgcolor=t["bg_card"],
        plot_bgcolor=t["bg_root"],
        font=dict(family="Space Mono, monospace", size=9, color=t["text"]),
        margin=dict(l=40, r=20, t=20, b=40),
        template="plotly_dark",
        height=320,
        title=dict(
            text=title,
            font=dict(family="Space Mono, monospace", size=10, color=t["text_muted"]),
            x=0,
        ),
        xaxis=dict(
            showgrid=False, zeroline=False,
            color=t["text_muted"], tickfont=dict(size=8),
        ),
        yaxis=dict(
            gridcolor=t["border"], gridwidth=1,
            zerolinecolor=t["border"], zeroline=False,
            color=t["text_muted"], tickfont=dict(size=8),
        ),
    )
    return f


def chart_amp(burst, theme: str = "green", title="Amplitude envelope", name="Signal"):
    t = _THEMES.get(theme, _THEMES["green"])
    c = t["signal"]
    # compute rgba fill from signal hex
    r, g, b_ = int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)
    fc = f"rgba({r},{g},{b_},0.08)"
    amp = np.abs(burst)
    ts  = np.arange(len(burst)) / SAMPLE_RATE * 1e6
    f   = _fig(title, theme)
    f.add_trace(
        go.Scatter(
            x=ts, y=amp, mode="lines", name=name,
            line=dict(color=c, width=1.5),
            fill="tozeroy", fillcolor=fc,
            hovertemplate="%{x:.2f} µs<br>Amp: %{y:.4f}<extra></extra>",
        )
    )
    f.update_xaxes(title="Time (µs)")
    f.update_yaxes(title="Amplitude")
    return f


def chart_iq(burst, theme: str = "green", title="I / Q channels"):
    t = _THEMES.get(theme, _THEMES["green"])
    ts = np.arange(len(burst)) / SAMPLE_RATE * 1e6
    I  = burst.real
    Q  = burst.imag
    f  = _fig(title, theme)
    f.add_trace(go.Scatter(x=ts, y=I, mode="lines", name="I (in-phase)",   line=dict(color=t["signal"], width=1.5)))
    f.add_trace(go.Scatter(x=ts, y=Q, mode="lines", name="Q (quadrature)", line=dict(color="#f09c28",   width=1.5)))
    f.update_xaxes(title="Time (µs)")
    f.update_yaxes(title="Amplitude")
    return f


def chart_conf(probs, thr, theme: str = "green"):
    t = _THEMES.get(theme, _THEMES["green"])
    f = _fig("Per-burst confidence", theme)
    x = np.arange(len(probs))
    y = probs * 100.0
    f.add_trace(go.Bar(x=x, y=y, name="Burst confidence", marker_color=t["signal"]))
    f.add_hline(
        y=thr, line=dict(color="#e04040", width=1.5, dash="dash"),
        annotation_text=f"Threshold {thr:.0f}%",
        annotation_position="top right",
    )
    f.update_xaxes(title="Burst index")
    f.update_yaxes(title="Confidence (%)", range=[0, 100])
    return f


def chart_overlay(burst_a, burst_b, theme: str = "green"):
    t  = _THEMES.get(theme, _THEMES["green"])
    ts = np.arange(len(burst_a)) / SAMPLE_RATE * 1e6
    f  = _fig("Amplitude overlay", theme)
    f.update_layout(paper_bgcolor=t["bg_root"], plot_bgcolor=t["bg_root"])
    f.add_trace(go.Scatter(x=ts, y=np.abs(burst_a), mode="lines", name="Signal A (auth)",  line=dict(color="#00e87a", width=1.5)))
    f.add_trace(go.Scatter(x=ts, y=np.abs(burst_b), mode="lines", name="Signal B (rogue)", line=dict(color="#e04040", width=1.5)))
    f.update_xaxes(title="Time (µs)")
    f.update_yaxes(title="Amplitude")
    return f


# ── Signal Lab result card ────────────────────────────────────────────────────

def _show_confidence(conf: float, is_auth: bool):
    if not is_auth:
        color = "#e04040"; icon = "○"; label = "ACCESS DENIED"; badge = "THREAT DETECTED"
    elif conf >= 85.0:
        color = "#00e87a"; icon = "●"; label = "AUTHORIZED";    badge = "SIGNAL VERIFIED"
    elif conf >= 50.0:
        color = "#f09c28"; icon = "◑"; label = "UNCERTAIN";     badge = "LOW CONFIDENCE"
    else:
        color = "#e04040"; icon = "○"; label = "ACCESS DENIED"; badge = "THREAT DETECTED"

    bar_w = min(100.0, conf)
    st.markdown(
        f"""
        <div class="result-card" style="background:{color}0d; border:1px solid {color}33;">
            <span class="result-badge" style="color:{color}; border-color:{color}44; background:{color}0d;">{badge}</span>
            <span class="result-confval" style="color:{color};">
                {conf:.1f}%<span class="cursor" style="color:{color};">_</span>
            </span>
            <span class="result-label" style="color:{color};">{icon} {label}</span>
            <div class="conf-bar-wrap">
                <div class="conf-bar" style="width:{bar_w:.1f}%; background:{color};"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── OOK canvas waveform ───────────────────────────────────────────────────────

def _canvas_html(mode: str, color: str, bg_card: str, border: str, scanline_fill: str) -> str:
    scanline_tint = "rgba(0,0,0,0.12)" if mode != "denied" else f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.07)"
    return f"""<!DOCTYPE html>
<html><head>
<style>
  body {{ margin:0; padding:0; background:{bg_card}; overflow:hidden; }}
  canvas {{ display:block; width:100%; border:1px solid {border}; box-sizing:border-box; }}
</style></head>
<body>
<canvas id="c"></canvas>
<script>
var cvs = document.getElementById('c');
var ctx = cvs.getContext('2d');
var MODE  = '{mode}';
var COLOR = '{color}';
var SCANLINE_TINT = '{scanline_tint}';
var BITS  = [1,0,1,1,0,1,0,0,1,1,0,1];
var offset = 0;

function resize() {{
  cvs.width  = window.innerWidth || document.documentElement.clientWidth || 400;
  cvs.height = 160;
}}
resize();
window.addEventListener('resize', resize);

function draw() {{
  var W = cvs.width, H = cvs.height;
  ctx.fillStyle = '{bg_card}';
  ctx.fillRect(0, 0, W, H);

  for (var sy = 0; sy < H; sy += 3) {{
    ctx.fillStyle = SCANLINE_TINT;
    ctx.fillRect(0, sy, W, 1);
  }}

  ctx.beginPath();
  ctx.strokeStyle = COLOR;
  ctx.lineWidth = 1.5;
  ctx.shadowColor = COLOR;
  ctx.shadowBlur  = 5;

  var highY = H * 0.25, lowY = H * 0.75, midY = H * 0.5;

  if (MODE === 'denied') {{
    for (var x = 0; x <= W; x++) {{
      var t = (x + offset) * 0.013;
      var y = midY
            + H * 0.16 * Math.sin(t * 7.3)
            + H * 0.11 * Math.sin(t * 11.17)
            + H * 0.08 * Math.sin(t * 19.73);
      if (x === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }}
  }} else {{
    var segW  = W / BITS.length;
    var off   = offset % segW;
    var base  = Math.floor(offset / segW);
    var prevY = null;

    for (var i = -1; i <= BITS.length + 1; i++) {{
      var idx = ((base + i) % BITS.length + BITS.length) % BITS.length;
      var bit = BITS[idx];

      if (MODE === 'uncertain') {{
        var seed = Math.sin(idx * 127.1);
        if (seed > 0.52) bit = 0.5;
      }}

      var y = (bit === 1) ? highY : (bit === 0.5 ? midY : lowY);
      var x1 = i * segW - off;
      var x2 = x1 + segW;
      if (x2 < 0) continue;
      if (x1 > W) break;

      var dx1 = Math.max(0, x1);
      var dx2 = Math.min(W, x2);

      if (prevY === null) {{
        ctx.moveTo(dx1, y);
      }} else if (prevY !== y) {{
        ctx.lineTo(dx1, y);
      }}
      ctx.lineTo(dx2, y);
      prevY = y;
    }}
  }}

  ctx.stroke();
  offset += 1.5;
  requestAnimationFrame(draw);
}}
draw();
</script>
</body></html>"""


def _idle_canvas_html(bg_card: str, border: str, scanline_fill: str) -> str:
    return f"""<!DOCTYPE html>
<html><head>
<style>body{{margin:0;padding:0;background:{bg_card};overflow:hidden;}}canvas{{display:block;width:100%;border:1px solid {border};box-sizing:border-box;}}</style>
</head><body>
<canvas id="c"></canvas>
<script>
var cvs=document.getElementById('c');
var ctx=cvs.getContext('2d');
var t=0;
function resize(){{cvs.width=window.innerWidth||400;cvs.height=160;}}
resize();window.addEventListener('resize',resize);
function draw(){{
  var W=cvs.width,H=cvs.height;
  ctx.fillStyle='{bg_card}';ctx.fillRect(0,0,W,H);
  for(var sy=0;sy<H;sy+=3){{ctx.fillStyle='rgba(0,0,0,0.12)';ctx.fillRect(0,sy,W,1);}}
  ctx.beginPath();
  ctx.strokeStyle='{border}';ctx.lineWidth=1.5;ctx.shadowColor='{border}';ctx.shadowBlur=3;
  ctx.moveTo(0,H/2);ctx.lineTo(W,H/2);
  ctx.stroke();
  var cx=(t%W);
  ctx.beginPath();
  ctx.strokeStyle='{scanline_fill}';ctx.lineWidth=1;
  ctx.moveTo(cx,0);ctx.lineTo(cx,H);ctx.stroke();
  t+=0.8;requestAnimationFrame(draw);
}}
draw();
</script></body></html>"""


# ── PROJECT_RFFPLA animated title ────────────────────────────────────────────

def _title_html(theme: str = "green") -> str:
    t   = _THEMES.get(theme, _THEMES["green"])
    sig = t["signal"]
    txt = t["text"]
    bg  = t["bg_root"]
    glow = t["signal_glow"]
    scanline = t["signal_scanline"]
    r, g, b_ = int(sig[1:3], 16), int(sig[3:5], 16), int(sig[5:7], 16)
    return f"""<!DOCTYPE html>
<html><head>
<style>
  @import url('https://cdn.jsdelivr.net/npm/@fontsource/space-mono/400.css');
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    background: {bg};
    display: flex;
    align-items: center;
    justify-content: center;
    height: 110px;
    overflow: hidden;
    position: relative;
  }}
  @keyframes flicker {{
    0%, 100% {{ opacity: 1; }}
    92%       {{ opacity: 0.97; }}
    95%       {{ opacity: 0.92; }}
  }}
  @keyframes colorshift {{
    0%, 100% {{ color: {sig}; text-shadow: 0 0 18px {glow}; }}
    50%       {{ color: {txt}; text-shadow: 0 0 8px rgba({r},{g},{b_},0.15); }}
  }}
  @keyframes blink {{
    0%, 100% {{ opacity: 1; }}
    50%       {{ opacity: 0; }}
  }}
  @keyframes sweep {{
    0%   {{ top: -2px; opacity: 0; }}
    10%  {{ opacity: 1; }}
    90%  {{ opacity: 1; }}
    100% {{ top: 112px; opacity: 0; }}
  }}
  .container {{
    text-align: center;
    animation: flicker 4s infinite;
    position: relative;
    z-index: 1;
  }}
  .title {{
    font-family: 'Space Mono', 'Courier New', monospace;
    font-size: clamp(28px, 5vw, 52px);
    letter-spacing: 0.35em;
    color: {sig};
    animation: colorshift 3s ease-in-out infinite;
    text-transform: uppercase;
    display: inline;
  }}
  .cursor {{
    font-family: 'Space Mono', 'Courier New', monospace;
    font-size: clamp(28px, 5vw, 52px);
    color: {sig};
    animation: blink step-end 0.9s infinite, colorshift 3s ease-in-out infinite;
    display: inline;
    letter-spacing: 0;
  }}
  .scanline {{
    position: absolute;
    left: 0;
    width: 100%;
    height: 1px;
    background: {scanline};
    animation: sweep 3s linear infinite;
    pointer-events: none;
    z-index: 2;
  }}
</style>
</head>
<body>
<div class="scanline"></div>
<div class="container">
  <span class="title" id="t"></span><span class="cursor">_</span>
</div>
<script>
var FULL = "PROJECT_RFFPLA";
var el   = document.getElementById('t');
var idx  = 0;

function type() {{
  if (idx <= FULL.length) {{
    el.textContent = FULL.slice(0, idx);
    idx++;
    if (idx > FULL.length) {{
      setTimeout(function() {{ idx = 0; type(); }}, 2000);
    }} else {{
      setTimeout(type, 80);
    }}
  }}
}}
type();
</script>
</body></html>"""


# ── Session log HTML ──────────────────────────────────────────────────────────

def _session_log_html(log: list) -> str:
    if not log:
        return '<div class="idle-card" style="padding:12px 8px;">NO ENTRIES THIS SESSION</div>'
    rows = ""
    for entry in reversed(log):
        ts, fname, result, conf_, bursts_ = entry
        rc = "row-pass" if result == "PASS" else "row-fail"
        rows += (
            f'<tr class="{rc}">'
            f"<td>{ts}</td><td>{fname}</td><td>{result}</td>"
            f"<td>{conf_}%</td><td>{bursts_}</td>"
            f"</tr>"
        )
    return (
        '<div class="session-wrap">'
        '<table class="session-tbl"><thead><tr>'
        "<th>TIME</th><th>FILE</th><th>RESULT</th><th>CONF</th><th>BURSTS</th>"
        "</tr></thead>"
        f"<tbody>{rows}</tbody></table></div>"
    )


# ── Upload parsing ─────────────────────────────────────────────────────────────

def _parse_upload(raw_bytes: bytes):
    arr = np.frombuffer(raw_bytes, dtype=np.complex64)
    n_samp = len(arr)
    duration_s = n_samp / SAMPLE_RATE

    _, bursts, _ = extract_bursts(raw_bytes)
    windows = []

    for b in bursts:
        if len(b) < WINDOW_SIZE:
            continue
        start = (len(b) - WINDOW_SIZE) // 2
        iq_window = b[start:start + WINDOW_SIZE]
        X2 = iq_to_2ch_window(iq_window)
        if X2 is not None:
            windows.append(X2)

    if windows:
        X = np.stack(windows, axis=0).astype(np.float32)
    else:
        X = np.zeros((0, WINDOW_SIZE, 2), dtype=np.float32)

    info = {
        "duration_s": duration_s,
        "n_samples": int(n_samp),
        "n_bursts": len(windows),
    }
    return X, bursts, info


# ── Sidebar status helper ─────────────────────────────────────────────────────

def _sidebar_status_html(theme: str, model_ok: bool) -> str:
    if not model_ok:
        return '<div class="offline"><span class="offline-dot"></span>SYSTEM OFFLINE</div>'
    if theme == "red":
        return '<div class="threat-status"><span class="threat-dot"></span>THREAT DETECTED</div>'
    if theme == "amber":
        return '<div class="uncertain-status"><span class="uncertain-dot"></span>SYSTEM UNCERTAIN</div>'
    return '<div class="armed"><span class="armed-dot"></span>SYSTEM ARMED</div>'


# ── Load model (needed by header status + sidebar + tabs) ────────────────────

interp, model_ok, model_err = load_model()

# ── Animated project title ───────────────────────────────────────────────────

components.html(_title_html(st.session_state.theme), height=110, scrolling=False)

# ── Header ────────────────────────────────────────────────────────────────────

_current_time = datetime.datetime.now().strftime("%H:%M:%S")
_threat_html = '<span class="rf-threat">◈ THREAT DETECTED</span>' if st.session_state.theme == "red" else ""

st.markdown(
    f'<div class="rf-header">'
    f'<span class="rf-logo">◈ RFFPLA</span>'
    f'<span class="rf-center">RF Physical Layer Authentication</span>'
    f'<div class="rf-status">{_threat_html}'
    f'<span class="rf-dot"></span>'
    f'<span class="rf-status-text">ONLINE</span>'
    f'<span class="rf-status-text">{_current_time}</span>'
    f'</div></div>',
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<span class="sb-hdr">◈ System Status</span>', unsafe_allow_html=True)
    st.markdown(_sidebar_status_html(st.session_state.theme, model_ok), unsafe_allow_html=True)
    if not model_ok:
        st.caption(model_err)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<span class="sb-hdr">◈ Enrolled Device</span>', unsafe_allow_html=True)
    st.markdown(
        """
        <table class="sb-tbl">
            <tr><td>NODE</td><td>ESP32 + CC1101</td></tr>
            <tr><td>FREQUENCY</td><td>433.92 MHz</td></tr>
            <tr><td>MODULATION</td><td>OOK</td></tr>
            <tr><td>STATUS</td><td><span style="color:#00e87a;">● ACTIVE</span></td></tr>
        </table>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<span class="sb-hdr">◈ Model Performance</span>', unsafe_allow_html=True)
    st.markdown(
        """
        <table class="sb-tbl">
            <tr><td>ACCURACY</td><td>98.1% (held-out)</td></tr>
            <tr><td>PARAMETERS</td><td>≈230k</td></tr>
            <tr><td>CHANNELS</td><td>2 (I · Q)</td></tr>
            <tr><td>WINDOW</td><td>1024 samples</td></tr>
        </table>
        """,
        unsafe_allow_html=True,
    )

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2 = st.tabs(["◈  AUTHENTICATION", "◈  COMPARE MODE"])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — Authentication
# ═════════════════════════════════════════════════════════════════════════════

with tab1:
    up_col, thr_col = st.columns([2, 1])
    with up_col:
        uploaded = st.file_uploader(
            "GNU Radio .c64 capture file",
            type=["c64"],
            label_visibility="collapsed",
            key="up1",
        )
        st.caption("5–12 second recording recommended · Max 200 MB")
    with thr_col:
        DISPLAY_THRESHOLD = st.slider(
            "Confidence threshold",
            0.50, 0.95, 0.70, 0.05,
            help="Minimum confidence for PASS decision",
            key="thr1",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    _cur_theme = st.session_state.theme
    _t         = _THEMES[_cur_theme]

    # ── No file yet — idle two-panel ──────────────────────────────────────────
    if uploaded is None:
        lp, rp = st.columns([1, 1])
        with lp:
            st.markdown('<div class="panel-hdr">◈ SIGNAL MONITOR</div>', unsafe_allow_html=True)
            components.html(
                _idle_canvas_html(_t["bg_card"], _t["border"], _t["scanline_fill"]),
                height=190, scrolling=False,
            )
            st.markdown(
                """
                <div class="meta-grid">
                    <span class="meta-lbl">FREQUENCY</span><span class="meta-val">433.92 MHz</span>
                    <span class="meta-lbl">MODULATION</span><span class="meta-val">OOK</span>
                    <span class="meta-lbl">SAMPLE RATE</span><span class="meta-val">2.0 Msps</span>
                    <span class="meta-lbl">BURSTS</span><span class="meta-val">—</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with rp:
            st.markdown('<div class="panel-hdr">◈ AUTHENTICATION RESULT</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="idle-card">AWAITING SIGNAL INPUT<br>'
                f'<span style="color:{_t["border"]};font-size:32px;display:block;margin-top:12px;">○</span></div>',
                unsafe_allow_html=True,
            )

    if uploaded is not None and not model_ok:
        st.error(model_err)

    if uploaded is not None and model_ok:
        safe_name = _safe_filename(uploaded.name)

        if uploaded.size > _MAX_UPLOAD_BYTES:
            st.error(
                f"File too large ({uploaded.size / 1_048_576:.1f} MiB). Maximum is 200 MB."
            )
        else:
            raw = uploaded.read()
            n_samp = len(raw) // 8
            if n_samp < 1000:
                st.error("File too small. Minimum recommended recording is 5 seconds at 2 Msps.")
            else:
                file_hash = hashlib.md5(raw[:min(8192, len(raw))]).hexdigest()
                is_new = st.session_state.last_hash != file_hash

                if is_new:
                    analyzing_ph = st.empty()
                    analyzing_ph.markdown(
                        '<div class="analyzing-card">⟳ &nbsp; ANALYZING SIGNAL'
                        '<div class="scan-bar"></div></div>',
                        unsafe_allow_html=True,
                    )
                    time.sleep(0.9)
                    analyzing_ph.empty()
                    st.session_state.last_hash = file_hash

                X, bursts, info = _parse_upload(raw)

                if not len(X):
                    st.error(
                        "No signal burst detected. Try a longer recording or "
                        "check that the squelch threshold is not too high."
                    )
                else:
                    is_auth, conf, probs, std = _run_predict(X)
                    thr = DISPLAY_THRESHOLD * 100.0

                    # ── Update theme and rerun if it changed ───────────────
                    new_theme = _conf_to_theme(conf, is_auth)
                    if new_theme != st.session_state.theme:
                        st.session_state.theme = new_theme
                        st.rerun()

                    _cur_theme = st.session_state.theme
                    _t         = _THEMES[_cur_theme]

                    # Canvas mode
                    if not is_auth:
                        canvas_mode = "denied"
                    elif conf >= 85.0:
                        canvas_mode = "authorized"
                    elif conf >= 50.0:
                        canvas_mode = "uncertain"
                    else:
                        canvas_mode = "denied"

                    canvas_color = _t["signal"]

                    # ── Two-panel layout ───────────────────────────────────
                    lp, rp = st.columns([1, 1])

                    with lp:
                        st.markdown('<div class="panel-hdr">◈ SIGNAL MONITOR</div>', unsafe_allow_html=True)
                        components.html(
                            _canvas_html(canvas_mode, canvas_color, _t["bg_card"], _t["border"], _t["scanline_fill"]),
                            height=190,
                            scrolling=False,
                        )
                        st.markdown(
                            f"""
                            <div class="meta-grid">
                                <span class="meta-lbl">FREQUENCY</span><span class="meta-val">433.92 MHz</span>
                                <span class="meta-lbl">MODULATION</span><span class="meta-val">OOK</span>
                                <span class="meta-lbl">SAMPLE RATE</span><span class="meta-val">2.0 Msps</span>
                                <span class="meta-lbl">BURSTS</span><span class="meta-val">{info['n_bursts']}</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    with rp:
                        session_id = f"AUTH_S{len(st.session_state.session_log) + 1}"
                        st.markdown('<div class="panel-hdr">◈ AUTHENTICATION RESULT</div>', unsafe_allow_html=True)
                        st.write(f"DEBUG raw_mean={float(np.mean(probs)):.4f} | is_auth={is_auth} | conf={conf:.2f}")
                        _show_confidence(conf, is_auth)
                        st.markdown(
                            f"""
                            <div class="meta-grid">
                                <span class="meta-lbl">DEVICE</span><span class="meta-val">CC1101 + ESP32</span>
                                <span class="meta-lbl">SESSION</span><span class="meta-val">{session_id}</span>
                                <span class="meta-lbl">MODEL</span><span class="meta-val">v5 onset 2ch</span>
                                <span class="meta-lbl">ACCURACY</span><span class="meta-val">98.1%</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    if is_new:
                        st.session_state.session_log.append(
                            [
                                datetime.datetime.now().strftime("%H:%M:%S"),
                                safe_name,
                                "PASS" if conf >= thr and is_auth else "FAIL",
                                f"{conf:.1f}",
                                len(X),
                            ]
                        )

                    # ── Signal Fingerprint Analysis ────────────────────────
                    st.markdown("<br>", unsafe_allow_html=True)

                    st.markdown(
                        f"""
                        <div style="
                            border:1px solid {_t['border']};
                            border-radius:6px;
                            padding:20px 20px 4px 20px;
                            box-shadow:0 0 40px rgba(0,0,0,0.2) inset;
                            background:{_t['bg_main']};
                            transition: background 0.6s ease, border-color 0.6s ease;
                        ">
                        <div style="
                            display:flex; align-items:center; gap:8px;
                            border-bottom:1px solid {_t['border']};
                            padding-bottom:10px; margin-bottom:16px;
                        ">
                            <span style="
                                color:{_t['text_muted']};font-size:7px;letter-spacing:.42em;
                                text-transform:uppercase;
                            ">◈ SIGNAL FINGERPRINT ANALYSIS</span>
                        </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    ca, cb, cc = st.columns(3)
                    with ca:
                        st.markdown(
                            f"""
                            <div class="chart-card">
                                <span class="chart-accent" style="background:{canvas_color};"></span>
                                <div class="chart-card-body">
                                    <span class="chart-title-dot" style="background:{canvas_color};"></span>
                                    <span class="chart-lbl" style="color:{canvas_color} !important;">AMPLITUDE ENVELOPE</span>
                                    <span class="chart-desc">Burst power over time · {CENTER_FREQ / 1e6:.2f} MHz · {int(SAMPLE_RATE / 1e6)} Msps</span>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        st.plotly_chart(
                            chart_amp(bursts[0], _cur_theme),
                            use_container_width=True,
                            config={"displayModeBar": False},
                        )
                    with cb:
                        st.markdown(
                            f"""
                            <div class="chart-card">
                                <span class="chart-accent" style="background:#f09c28;"></span>
                                <div class="chart-card-body">
                                    <span class="chart-title-dot" style="background:#f09c28;"></span>
                                    <span class="chart-lbl" style="color:#f09c28 !important;">I / Q CHANNELS</span>
                                    <span class="chart-desc">In-phase vs quadrature component separation</span>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        st.plotly_chart(
                            chart_iq(bursts[0], _cur_theme),
                            use_container_width=True,
                            config={"displayModeBar": False},
                        )
                    with cc:
                        st.markdown(
                            f"""
                            <div class="chart-card">
                                <span class="chart-accent" style="background:{_t['text_muted']};"></span>
                                <div class="chart-card-body">
                                    <span class="chart-title-dot" style="background:{_t['text_muted']};"></span>
                                    <span class="chart-lbl" style="color:{_t['text_muted']} !important;">PER-BURST CONFIDENCE</span>
                                    <span class="chart-desc">Classifier score per extracted burst · threshold {thr:.0f}%</span>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        st.plotly_chart(
                            chart_conf(probs, thr, _cur_theme),
                            use_container_width=True,
                            config={"displayModeBar": False},
                        )

                    with st.expander("TECHNICAL DETAILS"):
                        dur = info["duration_s"]
                        active = int(np.sum(np.abs(bursts[0]) > 0.01))
                        st.markdown(
                            f"""
                            |                     |                           |
                            |---------------------|---------------------------|
                            | File                | `{safe_name}`             |
                            | Size                | {uploaded.size / 1e6:.1f} MB |
                            | Duration            | ~{dur:.1f} s              |
                            | Bursts extracted    | {len(X)}                  |
                            | Sample count        | {info["n_samples"]:,}     |
                            | Active samples      | {active} ({active / SAMPLE_RATE * 1e6:.1f} µs) |
                            | Individual scores   | {", ".join(f"{p:.3f}" for p in probs)} |
                            | Average score       | {float(np.mean(probs)):.4f} |
                            | Threshold (display) | {thr:.0f}% confidence      |
                            | Threshold (model)   | 0.70 score                |
                            """,
                        )

    # ── Session Log ───────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="panel-hdr">◈ SESSION LOG</div>', unsafe_allow_html=True)
    st.markdown(_session_log_html(st.session_state.session_log), unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — Compare mode
# ═════════════════════════════════════════════════════════════════════════════

with tab2:
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="panel-hdr">◈ SIGNAL A</div>', unsafe_allow_html=True)
        file_a = st.file_uploader(
            "Signal A — authorized device",
            type=["c64"],
            label_visibility="collapsed",
            key="cmp_a",
        )
    with col_b:
        st.markdown('<div class="panel-hdr">◈ SIGNAL B</div>', unsafe_allow_html=True)
        file_b = st.file_uploader(
            "Signal B — rogue / unknown device",
            type=["c64"],
            label_visibility="collapsed",
            key="cmp_b",
        )

    if file_a is None or file_b is None:
        st.info("Upload both Signal A and Signal B to run comparison.")
    elif not model_ok:
        st.error(model_err)
    else:
        size_err = False

        if file_a.size > _MAX_UPLOAD_BYTES:
            st.error(f"Signal A too large ({file_a.size / 1_048_576:.1f} MiB). Maximum is 200 MB.")
            size_err = True
        if file_b.size > _MAX_UPLOAD_BYTES:
            st.error(f"Signal B too large ({file_b.size / 1_048_576:.1f} MiB). Maximum is 200 MB.")
            size_err = True

        if not size_err:
            with st.spinner("Analysing both signals..."):
                raw_a = file_a.read()
                na = len(raw_a) // 8
                if na < 1000:
                    st.error("Signal A: File too small.")
                    del raw_a
                    size_err = True
                else:
                    X_a, bursts_a, info_a = _parse_upload(raw_a)
                    del raw_a

                if not size_err:
                    raw_b = file_b.read()
                    nb = len(raw_b) // 8
                    if nb < 1000:
                        st.error("Signal B: File too small.")
                        del raw_b
                        size_err = True
                    else:
                        X_b, bursts_b, info_b = _parse_upload(raw_b)
                        del raw_b

        burst_err = size_err
        if not size_err:
            if not len(X_a):
                st.error("Signal A: No burst detected.")
                burst_err = True
            if not len(X_b):
                st.error("Signal B: No burst detected.")
                burst_err = True

        if not burst_err:
            auth_a, conf_a, probs_a, std_a = _run_predict(X_a)
            auth_b, conf_b, probs_b, std_b = _run_predict(X_b)

            # Overall page theme follows Signal A (only if tab1 has no file)
            if uploaded is None:
                new_theme = _conf_to_theme(conf_a, auth_a)
                if new_theme != st.session_state.theme:
                    st.session_state.theme = new_theme
                    st.rerun()

            theme_a = _conf_to_theme(conf_a, auth_a)
            theme_b = _conf_to_theme(conf_b, auth_b)
            t_a     = _THEMES[theme_a]
            t_b     = _THEMES[theme_b]

            dur_a = info_a["duration_s"]
            dur_b = info_b["duration_s"]

            CMP_THR = (
                st.slider("Confidence threshold (compare)", 0.50, 0.95, 0.70, 0.05, key="thr2")
                * 100.0
            )

            low_a  = conf_a < CMP_THR
            low_b  = conf_b < CMP_THR
            pass_a = auth_a and not low_a
            pass_b = auth_b and not low_b

            # ── Side-by-side result cards ──────────────────────────────────
            rc_a, rc_vs, rc_b = st.columns([1, 0.12, 1])

            def _cmp_card(col, label, conf, is_auth, low_conf, n_bursts, std, card_theme):
                t_card = _THEMES.get(card_theme, _THEMES["green"])
                with col:
                    st.markdown(
                        f'<div class="panel-hdr">◈ {label.upper()}</div>',
                        unsafe_allow_html=True,
                    )
                    if not is_auth:
                        color = "#e04040"; icon = "○"; decision = "ACCESS DENIED"; badge = "THREAT DETECTED"
                    elif conf >= 85.0:
                        color = "#00e87a"; icon = "●"; decision = "AUTHORIZED";    badge = "SIGNAL VERIFIED"
                    elif conf >= 50.0:
                        color = "#f09c28"; icon = "◑"; decision = "UNCERTAIN";     badge = "LOW CONFIDENCE"
                    else:
                        color = "#e04040"; icon = "○"; decision = "ACCESS DENIED"; badge = "THREAT DETECTED"

                    st.write(f"DEBUG is_auth={is_auth} | conf={conf:.2f}")
                    bar_w = min(100.0, conf)
                    st.markdown(
                        f"""
                        <div class="result-card" style="background:{color}0d; border:1px solid {color}33;">
                            <span class="result-badge" style="color:{color}; border-color:{color}44; background:{color}0d;">{badge}</span>
                            <span class="result-confval" style="color:{color};">
                                {conf:.1f}%<span class="cursor" style="color:{color};">_</span>
                            </span>
                            <span class="result-label" style="color:{color};">{icon} {decision}</span>
                            <div class="conf-bar-wrap">
                                <div class="conf-bar" style="width:{bar_w:.1f}%; background:{color};"></div>
                            </div>
                        </div>
                        <div class="meta-grid">
                            <span class="meta-lbl">BURSTS</span><span class="meta-val">{n_bursts}</span>
                            <span class="meta-lbl">STD DEV</span><span class="meta-val">{std * 100:.1f}%</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            _cmp_card(rc_a, "Signal A", conf_a, auth_a, low_a, len(X_a), std_a, theme_a)

            with rc_vs:
                st.markdown(
                    '<div class="vs-wrap">'
                    '<div class="vs-line"></div>'
                    '<div class="vs-txt">VS</div>'
                    '<div class="vs-line"></div>'
                    '</div>',
                    unsafe_allow_html=True,
                )

            _cmp_card(rc_b, "Signal B", conf_b, auth_b, low_b, len(X_b), std_b, theme_b)

            # ── Overlay chart ──────────────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="panel-hdr">◈ AMPLITUDE OVERLAY</div>', unsafe_allow_html=True)
            st.plotly_chart(chart_overlay(bursts_a[0], bursts_b[0], theme_a), use_container_width=True)
            st.caption(
                f"Captured at {CENTER_FREQ / 1e6:.2f} MHz · "
                f"{int(SAMPLE_RATE / 1e6)} Msps · {WINDOW_SIZE}-sample window"
            )

            # ── IQ Channels ────────────────────────────────────────────────
            st.markdown('<div class="panel-hdr">◈ I / Q CHANNELS</div>', unsafe_allow_html=True)
            iq_ca, iq_cb = st.columns(2)
            with iq_ca:
                st.plotly_chart(chart_iq(bursts_a[0], theme_a, "Signal A — I / Q"), use_container_width=True)
            with iq_cb:
                st.plotly_chart(chart_iq(bursts_b[0], theme_b, "Signal B — I / Q"), use_container_width=True)

            # ── Comparison summary ─────────────────────────────────────────
            st.markdown('<div class="panel-hdr">◈ COMPARISON SUMMARY</div>', unsafe_allow_html=True)

            v_a = "PASS" if pass_a else ("LOW CONF" if low_a else "FAIL")
            v_b = "PASS" if pass_b else ("LOW CONF" if low_b else "FAIL")

            st.table(
                pd.DataFrame(
                    {
                        "Metric":   ["Confidence", "Decision", "Bursts extracted", "Burst std dev", "File size", "Duration"],
                        "Signal A": [f"{conf_a:.1f}%", v_a, str(len(X_a)), f"{std_a * 100:.1f}%", f"{file_a.size / 1e6:.1f} MB", f"~{dur_a:.1f}s"],
                        "Signal B": [f"{conf_b:.1f}%", v_b, str(len(X_b)), f"{std_b * 100:.1f}%", f"{file_b.size / 1e6:.1f} MB", f"~{dur_b:.1f}s"],
                    }
                )
            )

# ── Footer ────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="rf-footer">
        RFFPLA &nbsp;·&nbsp; HKR 2026 &nbsp;·&nbsp; v5 onset 2ch &nbsp;·&nbsp; 98.1% HELD-OUT ACCURACY
    </div>
    """,
    unsafe_allow_html=True,
)

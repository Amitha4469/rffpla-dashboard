import traceback
from datetime import datetime

import numpy as np
from flask import Flask, jsonify, render_template, request

from ai_edge_litert.interpreter import Interpreter
from config import MODEL_PATH, WINDOW_SIZE

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large — maximum is 500 MB'}), 413

# ── Model — loaded once at startup ───────────────────────────────────────────
interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print('Model input shape:', input_details[0]['shape'])

# ── Onset detection — identical to v5 training notebook ──────────────────────
_SMOOTH        = 16
_PRE_SAMPLES   = 128
_NOISE_PCT     = 20
_AMP_MULT      = 3.0
_SLOPE_MULT    = 4.0
_MIN_ONSET_GAP = 2048
_MAX_WINS      = 200   # cap per file for speed

def _moving_average(x, k=16):
    kernel = np.ones(k, dtype=np.float32) / k
    return np.convolve(x.astype(np.float32), kernel, mode='same')

def _detect_onsets(iq):
    amp = np.abs(iq).astype(np.float32)
    a   = _moving_average(amp, _SMOOTH)
    noise_floor = np.percentile(a, _NOISE_PCT)
    active = a > (noise_floor * _AMP_MULT)
    da = np.diff(a, prepend=a[0])
    slope_floor = np.percentile(np.abs(da), 60)
    rising = da > (slope_floor * _SLOPE_MULT)
    hits = np.where(active & rising)[0]
    if len(hits) == 0:
        return np.array([], dtype=np.int64)
    keep = [hits[0]]
    for h in hits[1:]:
        if h - keep[-1] >= _MIN_ONSET_GAP:
            keep.append(h)
    return np.array(keep, dtype=np.int64)

def _make_window(chunk):
    p = float(np.mean(np.abs(chunk)**2))
    if not np.isfinite(p) or p <= 1e-12:
        return None
    w = chunk / (np.sqrt(p) + 1e-12)
    return np.stack([w.real.astype(np.float32),
                     w.imag.astype(np.float32)], axis=-1)  # (1024, 2)

# ── Parse .c64 bytes → onset-aligned windows (matches training exactly) ───────
def parse_c64(raw_bytes):
    iq = np.frombuffer(raw_bytes, dtype=np.complex64)
    n_samples = len(iq)
    duration_s = n_samples / 2e6

    onsets  = _detect_onsets(iq)
    windows = []
    for s in onsets:
        start = int(max(0, s - _PRE_SAMPLES))
        end   = start + WINDOW_SIZE
        if end > len(iq):
            continue
        w = _make_window(iq[start:end])
        if w is not None:
            windows.append(w)
        if len(windows) >= _MAX_WINS:
            break

    # Fallback: if onset detector finds nothing, slide across the file
    if len(windows) == 0:
        print('[parse_c64] onset detector found 0 windows — using sliding fallback')
        step = WINDOW_SIZE // 2
        for start in range(0, len(iq) - WINDOW_SIZE, step):
            w = _make_window(iq[start:start + WINDOW_SIZE])
            if w is not None:
                windows.append(w)
            if len(windows) >= _MAX_WINS:
                break

    info = {
        'duration_s':   round(duration_s, 2),
        'sample_count': n_samples,
        'n_onsets':     len(onsets),
    }
    print(f'[parse_c64] {n_samples:,} samples | {len(onsets)} onsets | {len(windows)} windows')

    if not windows:
        return None, [], info

    X = np.array(windows, dtype=np.float32)   # (N, 1024, 2)
    # Return dummy burst list for waveform display (first WINDOW_SIZE samples as complex)
    burst0 = iq[:min(len(iq), 4096)]
    return X, [burst0], info

# ── Inference ─────────────────────────────────────────────────────────────────
def predict(X):
    probs = []
    for i in range(len(X)):
        inp = X[i:i+1].astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], inp)
        interpreter.invoke()
        out = interpreter.get_tensor(output_details[0]['index'])
        probs.append(float(out[0][0]))
    probs       = np.array(probs)
    mean_output = float(np.mean(probs))
    std_conf    = float(np.std(probs)) * 100
    is_auth     = mean_output > 0.70          # v5: high score = AUTH (label=1)
    # Send raw auth probability — frontend thresholds at 70%
    display_conf = mean_output * 100.0
    print(f'[predict] mean_output={mean_output:.4f} is_auth={is_auth} '
          f'display_conf={display_conf:.1f}% std={std_conf:.1f}% n={len(probs)}')
    return display_conf, is_auth, probs.tolist(), std_conf

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        raw = request.files['file'].read()
        print(f'[/predict] received {len(raw):,} bytes')
        if len(raw) < 8000:
            return jsonify({'error': 'File too small — minimum ~5 s at 2 Msps'}), 400

        X, bursts, info = parse_c64(raw)
        if X is None or len(X) == 0:
            return jsonify({'error': 'No valid windows detected — check recording length'}), 400

        conf, is_auth, probs, std = predict(X)
        b0   = bursts[0]
        step = max(1, len(b0) // 512)
        return jsonify({
            'confidence': round(conf, 2),
            'is_auth':    is_auth,
            'probs':      [round(p * 100, 2) for p in probs],
            'std':        round(std, 2),
            'n_bursts':   len(bursts),
            'n_windows':  int(len(X)),
            'amp':        np.abs(b0[::step]).tolist(),
            'i_ch':       b0[::step].real.tolist(),
            'q_ch':       b0[::step].imag.tolist(),
            'duration_s': info.get('duration_s', 0),
            'n_samples':  info.get('sample_count', 0),
            'timestamp':  datetime.now().strftime('%H:%M:%S'),
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        raw = request.files['file'].read()
        if len(raw) < 8000:
            return jsonify({'error': 'File too small'}), 400
        X, _, info = parse_c64(raw)
        if X is None or len(X) == 0:
            return jsonify({'error': 'No valid windows detected'}), 400
        conf, is_auth, _probs, _std = predict(X)
        result = 'AUTHORIZED' if is_auth else 'ACCESS DENIED'
        return jsonify({
            'result':           result,
            'confidence':       round(conf / 100.0, 2),
            'bursts_processed': int(len(X)),
        }), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/predict_compare', methods=['POST'])
def predict_compare():
    try:
        results = {}
        for key in ['file_a', 'file_b']:
            if key not in request.files:
                return jsonify({'error': f'{key} missing'}), 400
            raw = request.files[key].read()
            if len(raw) < 8000:
                results[key] = {'error': 'File too small'}
                continue
            X, bursts, info = parse_c64(raw)
            if X is None or len(X) == 0:
                results[key] = {'error': 'No windows detected'}
                continue
            conf, is_auth, probs, std = predict(X)
            b0   = bursts[0]
            step = max(1, len(b0) // 512)
            results[key] = {
                'confidence': round(conf, 2),
                'is_auth':    is_auth,
                'probs':      [round(p * 100, 2) for p in probs],
                'std':        round(std, 2),
                'n_bursts':   len(bursts),
                'n_windows':  int(len(X)),
                'amp':        np.abs(b0[::step]).tolist(),
                'i_ch':       b0[::step].real.tolist(),
                'q_ch':       b0[::step].imag.tolist(),
                'duration_s': info.get('duration_s', 0),
            }
        return jsonify(results)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

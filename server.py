import traceback
from datetime import datetime

import numpy as np
from flask import Flask, jsonify, render_template, request

from ai_edge_litert.interpreter import Interpreter
from config import MODEL_PATH, WINDOW_SIZE
from preprocess import extract_bursts

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


# ── Inference ─────────────────────────────────────────────────────────────────
def predict(X):
    probs = []
    for i in range(len(X)):
        inp = X[i:i+1].astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], inp)
        interpreter.invoke()
        out = interpreter.get_tensor(output_details[0]['index'])
        probs.append(float(out[0][0]))
    probs        = np.array(probs)
    mean_output  = float(np.mean(probs))
    std_conf     = float(np.std(probs)) * 100
    is_auth      = mean_output > 0.70              # v5: high score = AUTH
    display_conf = mean_output * 100.0  # raw auth prob — frontend thresholds at 70%
    return display_conf, is_auth, probs.tolist(), std_conf


# ── Burst extraction + 2-channel power-normalised window builder ──────────────
def parse_c64(raw_bytes):
    """Return (X, bursts, info) where X has shape (N, WINDOW_SIZE, 2)."""
    _, bursts, info = extract_bursts(raw_bytes)
    windows = []
    for b in bursts:
        if len(b) >= WINDOW_SIZE:
            centre = (len(b) - WINDOW_SIZE) // 2
            w = b[centre:centre + WINDOW_SIZE]
        else:
            w = np.concatenate([b, np.zeros(WINDOW_SIZE - len(b), dtype=np.complex64)])
        p = float(np.mean(np.abs(w)**2))
        if not np.isfinite(p) or p <= 1e-12:
            continue
        w_norm = w / (np.sqrt(p) + 1e-12)
        windows.append(np.stack([w_norm.real.astype(np.float32),
                                  w_norm.imag.astype(np.float32)], axis=-1))  # (WINDOW_SIZE, 2)
    if not windows:
        return None, bursts, info
    return np.array(windows, dtype=np.float32), bursts, info  # (N, WINDOW_SIZE, 2)


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
        print(f'[/predict] bursts={len(bursts)}  X={None if X is None else X.shape}  info={info}')

        if X is None or len(X) == 0:
            return jsonify({'error': 'No valid bursts detected — check squelch / recording length'}), 400

        conf, is_auth, probs, std = predict(X)
        print(f'[/predict] conf={conf:.2f}%  is_auth={is_auth}  std={std:.2f}%  windows={len(X)}')

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
            'duration_s': round(info.get('duration_s', 0), 2),
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
            return jsonify({'error': 'File too small — minimum ~5 s at 2 Msps'}), 400

        X, _bursts, info = parse_c64(raw)

        if X is None or len(X) == 0:
            return jsonify({'error': 'No valid bursts detected'}), 400

        conf, is_auth, _probs, _std = predict(X)

        if is_auth:
            result = 'AUTHORIZED'
        else:
            result = 'ACCESS DENIED'

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
            print(f'[/predict_compare] {key}: {len(raw):,} bytes')
            if len(raw) < 8000:
                results[key] = {'error': 'File too small'}
                continue
            X, bursts, info = parse_c64(raw)
            if X is None or len(X) == 0:
                results[key] = {'error': 'No bursts detected'}
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
                'duration_s': round(info.get('duration_s', 0), 2),
            }
        return jsonify(results)

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

"""
preprocess.py — RF burst extraction pipeline for RFFPLA.

Library use:
    from preprocess import extract_bursts
    X, bursts, info = extract_bursts("capture.c64")      # filepath
    X, bursts, info = extract_bursts(raw_bytes)           # or raw bytes

CLI use:
    python preprocess.py --input data/raw --output data/processed \
                         --label 0 --split 0.70,0.15,0.15
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from config import THRESHOLD, WINDOW_SIZE, SAMPLE_RATE

# ── Burst detector constants (implementation detail, not model hyper-params) ──
_SMOOTH_WIN = 20    # samples in the moving-average amplitude smoother
_GUARD      = 50    # guard samples prepended/appended around each burst edge
_MIN_LEN    = 80    # minimum burst length (samples) before windowing
_MAX_BURSTS = 20    # cap per file to avoid very long captures dominating


# ─────────────────────────────────────────────────────────────────────────────
# Core function
# ─────────────────────────────────────────────────────────────────────────────

def extract_bursts(source):
    """Extract and window signal bursts from a .c64 capture.

    Parameters
    ----------
    source : str | Path | bytes | bytearray
        Path to a .c64 file, or the raw bytes already read from one.

    Returns
    -------
    X : np.ndarray, shape (N, WINDOW_SIZE, 2), dtype float32
        Stacked model-input arrays — channel 0 is I, channel 1 is Q.
        N == 0 if no valid bursts were found.
    bursts : list of np.ndarray, each shape (WINDOW_SIZE,), dtype complex64
        Complex IQ snapshots (used for plotting).
    info : dict
        burst_count  – number of bursts returned in X
        dropped      – bursts rejected (too short or below threshold)
        sample_count – total IQ samples in the raw capture
        duration_s   – capture duration in seconds
    """
    # ── Load raw bytes ────────────────────────────────────────────────────────
    if isinstance(source, (str, os.PathLike)):
        with open(source, "rb") as fh:
            raw = fh.read()
    else:
        raw = bytes(source)

    # ── Decode complex64 ─────────────────────────────────────────────────────
    floats = np.frombuffer(raw, dtype=np.float32).copy()
    if len(floats) % 2 != 0:
        floats = floats[:-1]

    iq          = (floats[0::2] + 1j * floats[1::2]).astype(np.complex64)
    sample_count = len(iq)
    duration_s   = sample_count / SAMPLE_RATE

    # ── Amplitude envelope & burst detection ──────────────────────────────────
    amp = np.abs(iq)
    smoothed = np.convolve(amp, np.ones(_SMOOTH_WIN) / _SMOOTH_WIN, mode="same")

    # Rising / falling edges via diff on boolean mask
    active = np.concatenate([[False], smoothed > THRESHOLD, [False]])
    edges  = np.diff(active.astype(np.int8))
    starts = np.where(edges ==  1)[0]
    ends   = np.where(edges == -1)[0]

    arrays  = []
    bursts  = []
    dropped = 0

    for s, e in zip(starts, ends):
        if len(arrays) >= _MAX_BURSTS:
            break

        # Add guard samples around burst edges
        b = iq[max(0, s - _GUARD): min(len(iq), e + _GUARD)]

        if len(b) < _MIN_LEN:
            dropped += 1
            continue

        # Normalise to peak amplitude
        pk = np.max(np.abs(b))
        if pk > 0:
            b = b / pk

        # Window to exactly WINDOW_SIZE samples (truncate or zero-pad)
        if len(b) >= WINDOW_SIZE:
            b = b[:WINDOW_SIZE]
        else:
            b = np.concatenate([b, np.zeros(WINDOW_SIZE - len(b), dtype=np.complex64)])

        arrays.append(np.stack([b.real, b.imag], axis=-1).astype(np.float32))
        bursts.append(b)

    X = np.stack(arrays, axis=0) if arrays else np.empty((0, WINDOW_SIZE, 2), dtype=np.float32)

    info = {
        "burst_count":  len(bursts),
        "dropped":      dropped,
        "sample_count": sample_count,
        "duration_s":   duration_s,
    }
    return X, bursts, info


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_split(s):
    parts = [float(x) for x in s.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--split requires exactly three values, e.g. 0.70,0.15,0.15")
    if abs(sum(parts) - 1.0) > 1e-6:
        raise argparse.ArgumentTypeError(f"--split values must sum to 1.0 (got {sum(parts):.4f})")
    return parts


def _stratified_split(X, y, ratios, rng):
    """Simple random split (single label → no stratification needed)."""
    n      = len(X)
    idx    = rng.permutation(n)
    n_tr   = int(round(ratios[0] * n))
    n_va   = int(round(ratios[1] * n))
    # Test gets whatever is left so totals always add up exactly
    tr_idx = idx[:n_tr]
    va_idx = idx[n_tr: n_tr + n_va]
    te_idx = idx[n_tr + n_va:]
    return (
        X[tr_idx], X[va_idx], X[te_idx],
        y[tr_idx], y[va_idx], y[te_idx],
    )


def main():
    parser = argparse.ArgumentParser(
        description="Extract RF bursts from .c64 files and save train/val/test splits."
    )
    parser.add_argument("--input",  required=True,
                        help="Folder containing .c64 capture files")
    parser.add_argument("--output", required=True,
                        help="Folder where .npy output files are saved")
    parser.add_argument("--label",  required=True, type=int, choices=[0, 1],
                        help="Class label: 0=authorized, 1=rogue")
    parser.add_argument("--split",  default="0.70,0.15,0.15",
                        type=_parse_split,
                        help="Train/val/test ratios (default: 0.70,0.15,0.15)")
    parser.add_argument("--seed",   default=42, type=int,
                        help="Random seed for reproducible splits (default: 42)")
    args = parser.parse_args()

    input_dir  = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.is_dir():
        sys.exit(f"Error: --input '{input_dir}' is not a directory.")

    c64_files = sorted(input_dir.glob("*.c64"))
    if not c64_files:
        sys.exit(f"Error: No .c64 files found in '{input_dir}'.")

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Process each file ─────────────────────────────────────────────────────
    all_arrays  = []
    file_logs   = []
    total_drop  = 0
    n_processed = 0

    for filepath in c64_files:
        try:
            X, _bursts, info = extract_bursts(filepath)
        except Exception as exc:
            print(f"  [SKIP] {filepath.name}: {exc}", file=sys.stderr)
            continue

        n_processed += 1
        total_drop  += info["dropped"]

        file_logs.append({
            "file":         filepath.name,
            "burst_count":  info["burst_count"],
            "dropped":      info["dropped"],
            "sample_count": info["sample_count"],
            "duration_s":   round(info["duration_s"], 3),
        })

        if info["burst_count"] > 0:
            all_arrays.append(X)

    if not all_arrays:
        sys.exit("Error: No valid bursts extracted from any file.")

    # ── Combine & split ───────────────────────────────────────────────────────
    X_all = np.concatenate(all_arrays, axis=0)
    y_all = np.full(len(X_all), args.label, dtype=np.int32)

    rng = np.random.default_rng(args.seed)
    X_tr, X_va, X_te, y_tr, y_va, y_te = _stratified_split(
        X_all, y_all, args.split, rng
    )

    # ── Save arrays ───────────────────────────────────────────────────────────
    np.save(output_dir / "train_X.npy", X_tr)
    np.save(output_dir / "val_X.npy",   X_va)
    np.save(output_dir / "test_X.npy",  X_te)
    np.save(output_dir / "train_y.npy", y_tr)
    np.save(output_dir / "val_y.npy",   y_va)
    np.save(output_dir / "test_y.npy",  y_te)

    # ── Save JSON log ─────────────────────────────────────────────────────────
    log = {
        "timestamp":    datetime.now(timezone.utc).isoformat(),
        "params": {
            "threshold":   THRESHOLD,
            "window_size": WINDOW_SIZE,
            "sample_rate": SAMPLE_RATE,
            "smooth_win":  _SMOOTH_WIN,
            "guard":       _GUARD,
            "min_len":     _MIN_LEN,
            "max_bursts":  _MAX_BURSTS,
        },
        "label":         args.label,
        "split_ratios":  args.split,
        "seed":          args.seed,
        "files_processed": n_processed,
        "files_skipped":   len(c64_files) - n_processed,
        "total_bursts":    int(len(X_all)),
        "total_dropped":   total_drop,
        "split": {
            "train": int(len(X_tr)),
            "val":   int(len(X_va)),
            "test":  int(len(X_te)),
        },
        "files": file_logs,
    }
    log_path = output_dir / "preprocess_log.json"
    with open(log_path, "w") as fh:
        json.dump(log, fh, indent=2)

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"Files processed : {n_processed}")
    print(f"Total bursts    : {len(X_all)}")
    print(f"Dropped bursts  : {total_drop}")
    print(f"Split           : train={len(X_tr)}  val={len(X_va)}  test={len(X_te)}")
    print(f"Saved to        : {output_dir.resolve()}")


if __name__ == "__main__":
    main()

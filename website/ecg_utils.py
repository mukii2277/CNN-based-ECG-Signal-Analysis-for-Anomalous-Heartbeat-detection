"""
ecg_utils.py
============
Utilities for loading ECG files and extracting beats for prediction.

Supported input formats:
  * WFDB binary format (.dat + .hea + optional .atr)
  * WFDB .dat alone  -> auto-decoded using MIT-BIH format 212 (12-bit packed binary)
  * CSV / TXT / TSV  -> one amplitude value per line
  * NumPy .npy files -> 1-D arrays
"""

import os
import numpy as np


# ── Focal loss (required to deserialise the .keras model) ─────────────────────

def binary_focal_loss(gamma=2.0, alpha=0.25):
    """Focal loss factory — matches training custom loss."""
    import tensorflow as tf

    def focal_loss(y_true, y_pred):
        eps    = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        ce     = (-y_true * tf.math.log(y_pred)
                  - (1 - y_true) * tf.math.log(1 - y_pred))
        weight = (alpha * y_true * tf.pow(1 - y_pred, gamma)
                  + (1 - alpha) * (1 - y_true) * tf.pow(y_pred, gamma))
        return tf.reduce_mean(weight * ce)
    return focal_loss


# ── ECG file loading ───────────────────────────────────────────────────────────

def load_ecg_file(tmp_dir: str, saved_paths: list):
    """
    Load an uploaded ECG into a 1-D numpy array and return (signal, fs).

    Tries formats in order:
      1. WFDB (.dat + .hea together)
      2. WFDB .dat alone  -> MIT-BIH format-212 raw binary decode
      3. CSV / TXT / TSV
      4. NumPy .npy
    """
    # Only keep files that actually exist on disk
    saved_paths = [p for p in saved_paths if os.path.exists(p)]

    # Build extension map
    ext_map = {}
    for p in saved_paths:
        ext = os.path.splitext(p)[1].lower()
        ext_map.setdefault(ext, []).append(p)

    # ── 1. Full WFDB record (.dat + .hea) ─────────────────────────────────────
    if ".dat" in ext_map and ".hea" in ext_map:
        try:
            return _load_wfdb(tmp_dir, ext_map[".dat"][0])
        except Exception as e:
            print(f"[ecg_utils] WFDB full load failed ({e}), trying raw fallback …")

    # ── 2. .dat alone -> raw binary fallback ──────────────────────────────────
    if ".dat" in ext_map:
        try:
            sig, fs = _load_dat_raw(ext_map[".dat"][0])
            print(f"[ecg_utils] Decoded .dat via raw binary: {len(sig)} samples")
            return sig, fs
        except Exception as e:
            print(f"[ecg_utils] Raw .dat decode also failed: {e}")
            raise ValueError(
                "Could not decode the .dat file automatically.\n"
                "MIT-BIH .dat files work best when you also upload the "
                "matching .hea header file (e.g. 100.dat AND 100.hea together).\n"
                "Alternatively export your ECG as a CSV file (one value per line)."
            )

    # ── 3. CSV / TXT / TSV ────────────────────────────────────────────────────
    for ext in (".csv", ".txt", ".tsv"):
        if ext in ext_map:
            return _load_csv(ext_map[ext][0])

    # ── 4. NumPy .npy ─────────────────────────────────────────────────────────
    if ".npy" in ext_map:
        return _load_npy(ext_map[".npy"][0])

    # ── 5. Last resort: try every file as plaintext ───────────────────────────
    for p in saved_paths:
        try:
            return _load_csv(p)
        except Exception:
            pass

    raise ValueError(
        "Unsupported file format.\n"
        "Upload a WFDB record (.dat + .hea files together), "
        "or a CSV/TXT/TSV file with one ECG sample value per line."
    )


# ── WFDB loader ────────────────────────────────────────────────────────────────

def _load_wfdb(tmp_dir: str, dat_path: str):
    import wfdb
    rec_id = os.path.splitext(os.path.basename(dat_path))[0]
    rec    = wfdb.rdrecord(os.path.join(tmp_dir, rec_id))
    signal = rec.p_signal[:, 0].astype(np.float32)
    fs     = int(rec.fs)
    return signal, fs


# ── Raw MIT-BIH binary decoder ─────────────────────────────────────────────────

def _load_dat_raw(dat_path: str, fs: int = 360):
    """
    Try multiple binary interpretations of a .dat file in this order:
      A. MIT-BIH format 212 (12-bit packed, 3 bytes = 2 samples)
      B. Raw int16 little-endian (format 16, one sample per 2 bytes)
      C. Raw int16 big-endian    (format 16 big)

    Returns (signal_float32, fs).
    """
    with open(dat_path, "rb") as fh:
        raw_bytes = bytearray(fh.read())

    n = len(raw_bytes)
    if n < 4:
        raise ValueError("File is too small to be a valid .dat record.")

    best_signal = None
    best_std    = -1

    # ─── A: Format 212 (12-bit packed) ───────────────────────────────────────
    if n % 3 == 0 and n >= 6:
        try:
            samples = _decode_format212(raw_bytes)
            sig     = np.array(samples, dtype=np.float32) / 200.0  # ADC -> mV
            if np.std(sig) > best_std:
                best_signal = sig
                best_std    = float(np.std(sig))
        except Exception:
            pass

    # ─── B: int16 LE ─────────────────────────────────────────────────────────
    try:
        sig = np.frombuffer(bytes(raw_bytes), dtype="<i2").astype(np.float32) / 200.0
        if np.std(sig) > best_std:
            best_signal = sig
            best_std    = float(np.std(sig))
    except Exception:
        pass

    # ─── C: int16 BE ─────────────────────────────────────────────────────────
    try:
        sig = np.frombuffer(bytes(raw_bytes), dtype=">i2").astype(np.float32) / 200.0
        if np.std(sig) > best_std:
            best_signal = sig
            best_std    = float(np.std(sig))
    except Exception:
        pass

    if best_signal is None or len(best_signal) < 180:
        raise ValueError("Could not decode any valid signal from the .dat file.")

    # Take first channel if multi-channel (interleaved channels, step by 2)
    # MIT-BIH 2-channel records: channel 0 = samples 0,2,4,...
    if best_std < 0.001:
        # try every-other-sample (2-lead interleaved)
        sig_ch0 = best_signal[::2]
        if np.std(sig_ch0) > best_std:
            best_signal = sig_ch0

    return best_signal, fs


def _decode_format212(raw_bytes: bytearray):
    """Decode MIT-BIH format 212: 3 bytes -> 2 signed 12-bit samples."""
    samples = []
    n_triples = len(raw_bytes) // 3
    for i in range(n_triples):
        b0, b1, b2 = raw_bytes[3*i], raw_bytes[3*i+1], raw_bytes[3*i+2]

        # Sample A (first 12 bits)
        a = b0 | ((b1 & 0x0F) << 8)
        if a >= 2048:
            a -= 4096

        # Sample B (last 12 bits)
        b = ((b1 >> 4) & 0x0F) | (b2 << 4)
        if b >= 2048:
            b -= 4096

        samples.extend([a, b])
    return samples


# ── CSV / TXT loader ───────────────────────────────────────────────────────────

def _load_csv(path: str, fs: int = 360):
    """Load a plain-text file containing one ECG sample per line."""
    data = []
    with open(path, "r", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Try each whitespace/comma-separated token
            for tok in line.replace(",", " ").replace(";", " ").split():
                try:
                    data.append(float(tok))
                    break
                except ValueError:
                    continue

    if len(data) < 180:
        raise ValueError(
            f"Only {len(data)} numeric values found. "
            "At least 180 samples (one beat window) are required."
        )

    return np.array(data, dtype=np.float32), fs


# ── NumPy .npy loader ──────────────────────────────────────────────────────────

def _load_npy(path: str, fs: int = 360):
    arr = np.load(path).astype(np.float32).ravel()
    if len(arr) < 180:
        raise ValueError(f"NumPy array too short ({len(arr)} samples).")
    return arr, fs


# ── Beat extraction ────────────────────────────────────────────────────────────

WINDOW = 90          # ± samples around R-peak -> 180-sample beat


def _detect_r_peaks(signal: np.ndarray, fs: int = 360):
    """Pan-Tompkins-inspired R-peak detector."""
    from scipy.signal import butter, filtfilt, find_peaks

    nyq    = fs / 2.0
    lo, hi = max(0.001, 5 / nyq), min(0.999, 15 / nyq)
    b, a   = butter(2, [lo, hi], btype="band")

    try:
        filtered = filtfilt(b, a, signal)
    except Exception:
        filtered = signal.copy()

    diff       = np.diff(filtered)
    squared    = diff ** 2
    win        = max(1, int(0.15 * fs))
    integrated = np.convolve(squared, np.ones(win) / win, mode="same")
    min_dist   = int(0.3 * fs)

    peaks, _ = find_peaks(
        integrated,
        distance=min_dist,
        height=np.mean(integrated) * 0.5,
    )
    return peaks


def preprocess_beats(signal: np.ndarray, fs: int = 360) -> np.ndarray:
    """
    Extract and z-normalise 180-sample beats centred on R-peaks.
    Returns np.ndarray of shape (N, 180, 1).
    Falls back to sliding-window if no R-peaks found.
    """
    r_peaks = _detect_r_peaks(signal, fs)

    beats = []
    for r in r_peaks:
        if r - WINDOW < 0 or r + WINDOW >= len(signal):
            continue
        beat = signal[r - WINDOW: r + WINDOW]
        beat = (beat - beat.mean()) / (beat.std() + 1e-8)
        beats.append(beat[..., np.newaxis])

    if len(beats) == 0:
        print("[ecg_utils] No R-peaks detected — using sliding-window fallback.")
        for start in range(0, len(signal) - 180, 90):
            beat = signal[start: start + 180]
            beat = (beat - beat.mean()) / (beat.std() + 1e-8)
            beats.append(beat[..., np.newaxis])

    if len(beats) == 0:
        raise ValueError(
            "Could not extract beats from the uploaded signal. "
            "Verify that the file contains a valid ECG waveform."
        )

    return np.array(beats, dtype=np.float32)

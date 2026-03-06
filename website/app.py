"""
app.py
======
Flask backend for the ECG Arrhythmia Detection website.
Handles ECG file upload, prediction, and XAI visualisation.
"""

import os
import sys
import io
import base64
import traceback
import tempfile

import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

# ── Resolve project root so we can import model & utilities ───────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

# ── Model & threshold ─────────────────────────────────────────────────────────
MODEL_PATH  = os.path.join(PROJECT_ROOT, "BEST MODEL", "94_percent_model.keras")
THRESHOLD   = 0.93

app = Flask(__name__)
CORS(app)

# ── Lazy-load model once ───────────────────────────────────────────────────────
_model = None

def get_model():
    global _model
    if _model is None:
        import tensorflow as tf
        from ecg_utils import binary_focal_loss
        print("⏳  Loading model …", flush=True)
        _model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={
                "focal_loss":        binary_focal_loss(),
                "binary_focal_loss": binary_focal_loss,
            },
        )
        print("✅  Model loaded.", flush=True)
    return _model


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts a WFDB record (*.dat + *.hea + *.atr  OR  raw CSV/txt).
    Returns JSON with:
      - prediction   : "Normal" | "Arrhythmia"
      - confidence   : float  (0-100)
      - n_beats      : int
      - n_arrhythmia : int
      - plots        : dict of base64 PNG strings
    """
    from ecg_utils import load_ecg_file, preprocess_beats
    from xai_utils  import generate_xai_plots

    # ------------------------------------------------------------------
    # 1. Receive uploaded file(s)
    # ------------------------------------------------------------------
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    uploaded_files = request.files.getlist("file")
    if not uploaded_files:
        return jsonify({"error": "No file received."}), 400

    # Save to a temp directory
    tmp_dir = tempfile.mkdtemp()
    saved_paths = []
    for f in uploaded_files:
        dest = os.path.join(tmp_dir, f.filename)
        f.save(dest)
        saved_paths.append(dest)

    # ------------------------------------------------------------------
    # 2. Load ECG signal
    # ------------------------------------------------------------------
    try:
        signal, fs = load_ecg_file(tmp_dir, saved_paths)
    except Exception as exc:
        return jsonify({"error": f"Could not read ECG file: {exc}"}), 422

    # ------------------------------------------------------------------
    # 3. Extract beats & predict
    # ------------------------------------------------------------------
    try:
        beats = preprocess_beats(signal)                # (N, 180, 1)
        model = get_model()

        import tensorflow as tf
        probs  = model.predict(beats, verbose=0).ravel()  # (N,)
        preds  = (probs > THRESHOLD).astype(int)

        n_beats      = len(beats)
        n_arrhythmia = int(preds.sum())
        n_normal     = n_beats - n_arrhythmia

        arrhythmia_ratio = n_arrhythmia / max(n_beats, 1)
        overall_label    = "Arrhythmia" if arrhythmia_ratio > 0.1 else "Normal"
        overall_conf     = float(probs.mean() * 100)

        # take the most representative anomaly beat (highest prob) for XAI
        if n_arrhythmia > 0:
            xai_idx  = int(np.argmax(probs))
            xai_label = 1
        else:
            xai_idx  = int(np.argmin(probs))
            xai_label = 0

        xai_beat = beats[xai_idx]     # (180, 1)
        xai_prob  = float(probs[xai_idx])

    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": f"Prediction failed: {exc}"}), 500

    # ------------------------------------------------------------------
    # 4. Generate XAI plots
    # ------------------------------------------------------------------
    try:
        plots = generate_xai_plots(model, beats, preds, probs,
                                   xai_beat, xai_label, xai_prob)
    except Exception as exc:
        traceback.print_exc()
        plots = {}

    # ------------------------------------------------------------------
    # 5. Build ECG signal plot (first ~5 s worth)
    # ------------------------------------------------------------------
    signal_plot = _make_signal_plot(signal, fs, n_arrhythmia, n_normal, probs, beats)

    return jsonify({
        "prediction":    overall_label,
        "confidence":    round(overall_conf, 2),
        "n_beats":       n_beats,
        "n_arrhythmia":  n_arrhythmia,
        "n_normal":      n_normal,
        "threshold":     THRESHOLD,
        "signal_plot":   signal_plot,
        "plots":         plots,
    })


# ── Helper: raw ECG signal plot ────────────────────────────────────────────────

def _make_signal_plot(signal, fs, n_arrhythmia, n_normal, probs, beats):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    display_samples = min(len(signal), int(fs * 10))   # first 10 s
    t = np.arange(display_samples) / fs

    fig, ax = plt.subplots(figsize=(12, 3), facecolor="#0f1117")
    ax.set_facecolor("#0f1117")

    ax.plot(t, signal[:display_samples], color="#00d4ff", linewidth=0.9, alpha=0.9)
    ax.set_xlabel("Time (s)", color="#aab4be", fontsize=9)
    ax.set_ylabel("Amplitude (mV)", color="#aab4be", fontsize=9)
    ax.set_title("Uploaded ECG Signal", color="#ffffff", fontsize=11, fontweight="bold")
    ax.tick_params(colors="#aab4be", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#2d3748")

    ax.grid(True, color="#1e2a3a", linewidth=0.5, alpha=0.7)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110,
                facecolor="#0f1117")
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

"""
xai_utils.py
============
XAI helper functions adapted from explainable.py to work headlessly
(no plt.show()) and return base64-encoded PNG strings for the web UI.
"""

import io
import base64
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Colour palette (matches dark UI theme) ────────────────────────────────────
BG     = "#0f1117"
FG     = "#ffffff"
GRID   = "#1e2a3a"
AXIS   = "#aab4be"
BLUE   = "#00d4ff"
RED    = "#ff4b6e"
GREEN  = "#00ff88"
PURPLE = "#a855f7"
ORANGE = "#f59e0b"


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110, facecolor=BG)
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()


def _styled_ax(ax, title="", xlabel="Time (samples)", ylabel="Amplitude"):
    ax.set_facecolor(BG)
    ax.set_title(title, color=FG, fontsize=10, fontweight="bold")
    ax.set_xlabel(xlabel, color=AXIS, fontsize=8)
    ax.set_ylabel(ylabel, color=AXIS, fontsize=8)
    ax.tick_params(colors=AXIS, labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("#2d3748")
    ax.grid(True, color=GRID, linewidth=0.5, alpha=0.7)


# ── Grad-CAM ──────────────────────────────────────────────────────────────────

def compute_gradcam_1d(model, x_sample, layer_name):
    import tensorflow as tf
    from tensorflow.keras import backend as K

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output],
    )
    x_in = tf.cast(np.expand_dims(x_sample, 0), tf.float32)

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x_in)
        loss = preds[:, 0]

    grads    = tape.gradient(loss, conv_out)[0]  # (T, C)
    pooled   = tf.reduce_mean(grads, axis=0)     # (C,)
    conv_out = conv_out[0]                        # (T, C)

    heat = tf.reduce_sum(pooled * conv_out, axis=-1)
    heat = tf.nn.relu(heat)
    denom = tf.reduce_max(heat) + K.epsilon()
    heat = heat / denom
    return heat.numpy()


def gradcam_plot(model, beat, label_str, layer_name) -> str:
    """Return base64 Grad-CAM overlay for a single beat."""
    import tensorflow as tf

    sig  = beat.squeeze()
    heat = compute_gradcam_1d(model, beat, layer_name)
    # Resize heat to match signal length
    from scipy.ndimage import zoom
    if len(heat) != len(sig):
        heat = zoom(heat, len(sig) / len(heat))

    fig, ax = plt.subplots(figsize=(9, 2.8), facecolor=BG)
    ax.plot(sig, color=BLUE, linewidth=1.2, label="ECG Signal", alpha=0.9)
    ax.plot(heat * float(np.abs(sig).max()),
            color=RED, linewidth=1.5, label="Grad-CAM", alpha=0.8)
    _styled_ax(ax, title=f"Grad-CAM — {layer_name}  [{label_str}]")
    ax.legend(facecolor="#1a2535", labelcolor=FG, fontsize=8)
    return _fig_to_b64(fig)


# ── Integrated Gradients ───────────────────────────────────────────────────────

def compute_integrated_gradients(model, x_sample, steps=50):
    import tensorflow as tf

    x_sample = tf.cast(x_sample, tf.float32)
    baseline = tf.zeros_like(x_sample, dtype=tf.float32)
    alphas   = tf.cast(tf.linspace(0.0, 1.0, steps + 1), tf.float32)
    interp   = baseline + alphas[:, tf.newaxis, tf.newaxis] * (x_sample - baseline)

    with tf.GradientTape() as tape:
        tape.watch(interp)
        probs  = model(interp)
        target = probs[:, 0]

    grads     = tape.gradient(target, interp)        # (steps+1, T, 1)
    avg_grads = tf.reduce_mean(grads[:-1], axis=0)   # (T, 1)
    ig = ((x_sample - baseline) * avg_grads)[0].numpy().squeeze()
    return ig


def ig_plot(model, beat, label_str) -> str:
    """Return base64 Integrated Gradients plot for a single beat."""
    sig = beat.squeeze()
    ig  = compute_integrated_gradients(model, np.expand_dims(beat, 0))

    fig, axes = plt.subplots(2, 1, figsize=(9, 5), facecolor=BG, sharex=True)
    axes[0].plot(sig, color=BLUE, linewidth=1.2)
    _styled_ax(axes[0], title=f"ECG Signal  [{label_str}]", ylabel="Amplitude")

    axes[1].fill_between(range(len(ig)), ig, 0,
                          where=ig > 0, color=GREEN, alpha=0.7, label="Positive attr.")
    axes[1].fill_between(range(len(ig)), ig, 0,
                          where=ig < 0, color=RED,   alpha=0.7, label="Negative attr.")
    axes[1].plot(ig, color=FG, linewidth=0.6, alpha=0.5)
    _styled_ax(axes[1], title="Integrated Gradients Attribution",
               xlabel="Time (samples)", ylabel="Attribution")
    axes[1].legend(facecolor="#1a2535", labelcolor=FG, fontsize=8)

    plt.tight_layout()
    return _fig_to_b64(fig)


# ── Beat probability bar chart ─────────────────────────────────────────────────

def beat_probability_plot(probs, threshold=0.93) -> str:
    """Bar chart of per-beat arrhythmia probability."""
    n = len(probs)
    colors = [RED if p > threshold else GREEN for p in probs]

    # Downsample if too many beats for readability
    if n > 80:
        step = n // 80
        probs  = probs[::step]
        colors = colors[::step]
        n = len(probs)

    fig, ax = plt.subplots(figsize=(10, 2.8), facecolor=BG)
    ax.bar(range(n), probs, color=colors, alpha=0.85, width=0.8)
    ax.axhline(threshold, color=ORANGE, linewidth=1.5,
               linestyle="--", label=f"Threshold ({threshold})")
    _styled_ax(ax, title="Per-Beat Arrhythmia Probability",
               xlabel="Beat Index", ylabel="Probability")
    ax.set_ylim(0, 1.05)
    ax.legend(facecolor="#1a2535", labelcolor=FG, fontsize=8)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=RED,   label="Arrhythmia"),
        Patch(facecolor=GREEN, label="Normal"),
        plt.Line2D([0], [0], color=ORANGE, linestyle="--",
                   label=f"Threshold ({threshold})"),
    ]
    ax.legend(handles=legend_elements, facecolor="#1a2535",
              labelcolor=FG, fontsize=8)
    return _fig_to_b64(fig)


# ── Beat waveform comparison ───────────────────────────────────────────────────

def beat_waveform_plot(beats, probs, threshold=0.93) -> str:
    """Side-by-side overlay of normal vs arrhythmia beats."""
    anom_idx = np.where(probs > threshold)[0]
    norm_idx = np.where(probs <= threshold)[0]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3), facecolor=BG)

    for ax, indices, label, color in [
        (axes[0], norm_idx[:10],  "Normal Beats",     GREEN),
        (axes[1], anom_idx[:10],  "Arrhythmia Beats", RED),
    ]:
        if len(indices) == 0:
            ax.text(0.5, 0.5, "No beats", ha="center", va="center",
                    color=AXIS, transform=ax.transAxes)
        else:
            for i in indices:
                ax.plot(beats[i].squeeze(), color=color, alpha=0.4, linewidth=0.9)
            # mean waveform
            mean_beat = beats[indices].squeeze().mean(axis=0)
            ax.plot(mean_beat, color=FG, linewidth=1.8, label="Mean")
        _styled_ax(ax, title=label, xlabel="Sample", ylabel="Amplitude (norm.)")
        ax.legend(facecolor="#1a2535", labelcolor=FG, fontsize=8)

    plt.tight_layout()
    return _fig_to_b64(fig)


# ── Main entry point called from app.py ───────────────────────────────────────

def generate_xai_plots(model, beats, preds, probs, xai_beat, xai_label, xai_prob) -> dict:
    """
    Generate all XAI plots for the web UI.
    Returns a dict mapping plot_name → base64 PNG data URI.
    """
    import tensorflow as tf

    label_str = f"{'Arrhythmia' if xai_label == 1 else 'Normal'}  (p={xai_prob:.3f})"
    plots = {}

    # 1. Per-beat probability bar chart
    try:
        plots["beat_prob"] = beat_probability_plot(probs)
    except Exception as e:
        print(f"beat_prob failed: {e}")

    # 2. Normal vs Arrhythmia waveform comparison
    try:
        plots["waveforms"] = beat_waveform_plot(beats, probs)
    except Exception as e:
        print(f"waveforms failed: {e}")

    # 3. Grad-CAM plots for each Conv1D layer
    try:
        conv_layers = [l.name for l in model.layers
                       if isinstance(l, tf.keras.layers.Conv1D)]
        if conv_layers:
            # Last layer
            plots["gradcam_last"] = gradcam_plot(
                model, xai_beat, label_str, conv_layers[-1])
            # Second-last (if available)
            if len(conv_layers) >= 2:
                plots["gradcam_second"] = gradcam_plot(
                    model, xai_beat, label_str, conv_layers[-2])
    except Exception as e:
        print(f"Grad-CAM failed: {e}")

    # 4. Integrated Gradients
    try:
        plots["integrated_gradients"] = ig_plot(model, xai_beat, label_str)
    except Exception as e:
        print(f"Integrated Gradients failed: {e}")

    return plots

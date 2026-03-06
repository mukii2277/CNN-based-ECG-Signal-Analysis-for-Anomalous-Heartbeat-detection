"""
explainable.py
==============
Explainable AI (XAI) analysis for the ECG Arrhythmia detection model.

Techniques implemented (same as the notebook ECG_ML_Part2_XAi_6_best_model_XAI):
  1. Grad-CAM (last Conv1D layer)
  2. Grad-CAM (second-last Conv1D layer — sharper gradients)
  3. All-layer Grad-CAM + raw activation heat-maps
  4. Integrated Gradients
  5. Dead Filter Analysis
  6. Feature Importance via Input Gradients
  7. t-SNE of penultimate-layer embeddings
  8. UMAP visualization
  9. KMeans Cluster Purity & Silhouette Score
 10. Per-class t-SNE (class A vs N, class V vs N)
 11. Intra- vs Inter-class Euclidean Distance Histogram

Usage:
  python explainable.py
"""

# ===========================================================================
# 0. Imports
# ===========================================================================
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import wfdb

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from scipy.spatial.distance import pdist, squareform
from tensorflow.keras import backend as K

try:
    import umap.umap_ as umap
    UMAP_AVAILABLE = True
except ImportError:
    print("⚠️  umap-learn not installed. UMAP plots will be skipped. "
          "Install with: pip install umap-learn")
    UMAP_AVAILABLE = False

# ===========================================================================
# 1. Configuration — update these paths to match your system
# ===========================================================================
TEST_DIR   = r"D:\Journel paper project\WORKFLOW\Dataset\Test"
MODEL_PATH = r"D:\Journel paper project\WORKFLOW\BEST MODEL\94_percent_model.keras"

NORMAL_CLASSES  = ['N', 'L', 'R', 'e', 'j']
ANOMALY_CLASSES = ['A', 'V', 'E', 'F', '/', 'f', 'a', 'S', 'J']

THRESHOLD      = 0.93    # classification threshold (best found during training)
EMBED_LIMIT    = 4000    # max beats to load for embedding analysis
IG_STEPS       = 50      # Integrated Gradients interpolation steps
TSNE_PERPLEXITY = 30

# Reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# ===========================================================================
# 2. Custom Focal Loss (required to deserialize the .keras model)
# ===========================================================================
def binary_focal_loss(gamma=2.0, alpha=0.25):
    """Focal loss factory — keeps class-imbalance handling from training."""
    def focal_loss(y_true, y_pred):
        eps    = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        ce     = (-y_true * tf.math.log(y_pred)
                  - (1 - y_true) * tf.math.log(1 - y_pred))
        weight = (alpha * y_true * tf.pow(1 - y_pred, gamma)
                  + (1 - alpha) * (1 - y_true) * tf.pow(y_pred, gamma))
        return tf.reduce_mean(weight * ce)
    return focal_loss

# ===========================================================================
# 3. Data loading
# ===========================================================================
def extract_beats(folder, limit=None):
    """
    Read all MIT-BIH records from *folder*, extract 180-sample beat windows
    centred on each annotated R-peak, z-normalise each beat, and return arrays.

    Returns
    -------
    beats   : np.ndarray  shape (N, 180, 1)
    y_bin   : np.ndarray  binary labels  0=Normal, 1=Anomaly
    y_sym   : np.ndarray  original annotation symbols  ('N', 'A', 'V', …)
    """
    beats, y_bin, y_sym = [], [], []
    all_classes = NORMAL_CLASSES + ANOMALY_CLASSES

    for filename in os.listdir(folder):
        if not filename.endswith(".dat"):
            continue
        rec_id = filename[:-4]
        try:
            rec = wfdb.rdrecord(os.path.join(folder, rec_id))
            ann = wfdb.rdann(os.path.join(folder, rec_id), 'atr')
            sig = rec.p_signal[:, 0]

            for i, r in enumerate(ann.sample):
                if r - 90 < 0 or r + 90 >= len(sig):
                    continue
                sym = ann.symbol[i]
                if sym not in all_classes:
                    continue

                beat = sig[r - 90 : r + 90]
                beat = (beat - beat.mean()) / (beat.std() + 1e-8)   # z-norm
                beats.append(beat[..., np.newaxis])
                y_bin.append(0 if sym in NORMAL_CLASSES else 1)
                y_sym.append(sym)

                if limit and len(beats) >= limit:
                    break

        except Exception as exc:
            print(f"⚠️  Skipped {rec_id}: {exc}")

        if limit and len(beats) >= limit:
            break

    return np.array(beats), np.array(y_bin), np.array(y_sym)

# ===========================================================================
# 4. Model loading
# ===========================================================================
def load_model(model_path):
    print("⏳ Loading model …")
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            "focal_loss":        binary_focal_loss(),   # instance (for inference)
            "binary_focal_loss": binary_focal_loss,     # factory  (for serialization)
        },
    )
    print(f"✅ Model loaded — {model.count_params():,} parameters")
    return model

# ===========================================================================
# 5. Grad-CAM (1-D signals)
# ===========================================================================
def compute_gradcam_1d(model, x_sample, layer_name):
    """
    Compute Grad-CAM heat-map for a single ECG sample.

    Parameters
    ----------
    model      : tf.keras.Model
    x_sample   : np.ndarray  shape (180, 1)  — one beat
    layer_name : str         name of the target Conv1D layer

    Returns
    -------
    heat : np.ndarray  shape (time_steps,)  values in [0, 1]
    """
    grad_model = tf.keras.models.Model(
        inputs  = model.inputs,
        outputs = [model.get_layer(layer_name).output, model.output],
    )
    x_in = tf.cast(np.expand_dims(x_sample, 0), tf.float32)

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x_in)
        loss = preds[:, 0]          # class-1 probability

    grads   = tape.gradient(loss, conv_out)[0]   # shape: (T, C)
    pooled  = tf.reduce_mean(grads, axis=0)      # shape: (C,)
    conv_out = conv_out[0]                        # shape: (T, C)

    heat = tf.reduce_sum(pooled * conv_out, axis=-1)
    heat = tf.nn.relu(heat)
    heat = heat / (tf.reduce_max(heat) + K.epsilon())
    return heat.numpy()


def show_gradcam(model, X_test, y_test, layer_name, sample_idx=0, label=1):
    """Plot ECG signal overlaid with Grad-CAM heat-map."""
    idx  = np.where(y_test == label)[0][sample_idx]
    sig  = X_test[idx].squeeze()
    heat = compute_gradcam_1d(model, X_test[idx], layer_name)

    plt.figure(figsize=(10, 3))
    plt.plot(sig, label='ECG', color='black', linewidth=1)
    plt.plot(heat * sig.max(), label='Grad-CAM', color='red', alpha=0.7)
    label_str = 'Anomaly' if label else 'Normal'
    plt.title(f"Grad-CAM [{layer_name}] – {label_str} sample #{sample_idx}")
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def run_gradcam_analysis(model, X_test, y_test):
    """Run Grad-CAM on the last and second-last Conv1D layers."""
    conv_layers = [l.name for l in model.layers
                   if isinstance(l, tf.keras.layers.Conv1D)]
    print(f"\n🔍 Conv1D layers found: {conv_layers}")

    # --- Last Conv1D (broad context) ---
    last_layer = conv_layers[-1]
    print(f"\n=== Grad-CAM: Last Conv1D layer ({last_layer}) ===")
    show_gradcam(model, X_test, y_test, last_layer, sample_idx=0, label=1)
    show_gradcam(model, X_test, y_test, last_layer, sample_idx=1, label=1)
    show_gradcam(model, X_test, y_test, last_layer, sample_idx=0, label=0)

    # --- Second-last Conv1D (sharper / earlier features) ---
    if len(conv_layers) >= 2:
        second_last = conv_layers[-2]
        print(f"\n=== Grad-CAM: Second-last Conv1D layer ({second_last}) ===")
        show_gradcam(model, X_test, y_test, second_last, sample_idx=0, label=1)
        show_gradcam(model, X_test, y_test, second_last, sample_idx=1, label=1)
        show_gradcam(model, X_test, y_test, second_last, sample_idx=0, label=0)


# ===========================================================================
# 6. All-layer Grad-CAM + Activation heat-maps
# ===========================================================================
def plot_layer_gradcam_and_activations(model, sample, label_name, layer_name,
                                       show_activation=True):
    """Plot (1) Grad-CAM overlay and (2) raw activation heat-map for one layer."""
    sig = sample.squeeze()
    cam = compute_gradcam_1d(model, sample, layer_name)

    # (1) ECG + Grad-CAM overlay
    plt.figure(figsize=(10, 2.8))
    plt.plot(sig, color='black', linewidth=1, label='ECG')
    plt.plot(cam * sig.max(), color='red', alpha=0.7, label='Grad-CAM')
    plt.title(f"{label_name} – {layer_name}")
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # (2) Raw activation heat-map
    if show_activation:
        act_model = tf.keras.Model(model.inputs,
                                   model.get_layer(layer_name).output)
        acts = act_model(np.expand_dims(sample, 0))[0].numpy().T   # (C, T)
        plt.figure(figsize=(10, 2.5))
        plt.imshow(acts, aspect='auto', origin='lower', interpolation='nearest')
        plt.colorbar(label='Activation')
        plt.title(f"Activations – {label_name} – {layer_name}")
        plt.ylabel("Channels")
        plt.xlabel("Time (samples)")
        plt.tight_layout()
        plt.show()


def run_all_layer_analysis(model, X_test, y_test, show_activation=True):
    """Visualise every Conv1D layer for one anomaly and one normal beat."""
    conv_layers = [l.name for l in model.layers
                   if isinstance(l, tf.keras.layers.Conv1D)]

    idx_anom = np.where(y_test == 1)[0][0]
    idx_norm = np.where(y_test == 0)[0][0]

    print("\n=== All-layer Grad-CAM: Anomaly ===")
    for ln in conv_layers:
        plot_layer_gradcam_and_activations(model, X_test[idx_anom],
                                           "Anomaly", ln, show_activation)

    print("\n=== All-layer Grad-CAM: Normal ===")
    for ln in conv_layers:
        plot_layer_gradcam_and_activations(model, X_test[idx_norm],
                                           "Normal", ln, show_activation)


# ===========================================================================
# 7. Integrated Gradients
# ===========================================================================
def compute_integrated_gradients(model, x_sample, baseline=None, steps=IG_STEPS):
    """
    Compute Integrated Gradients attribution for a single sample.

    Parameters
    ----------
    model    : tf.keras.Model
    x_sample : np.ndarray  shape (1, 180, 1)
    baseline : np.ndarray or None — default is all-zeros (silence)
    steps    : int  — number of Riemann-sum steps

    Returns
    -------
    ig : np.ndarray  shape (180,)  — per-timestep attribution
    """
    x_sample = tf.cast(x_sample, tf.float32)
    if baseline is None:
        baseline = tf.zeros_like(x_sample, dtype=tf.float32)
    else:
        baseline = tf.cast(baseline, tf.float32)

    alphas = tf.cast(tf.linspace(0.0, 1.0, steps + 1), tf.float32)
    interpolated = baseline + alphas[:, tf.newaxis, tf.newaxis] * (x_sample - baseline)

    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        probs  = model(interpolated)
        target = probs[:, 0]          # class 1

    grads     = tape.gradient(target, interpolated)   # (steps+1, T, 1)
    avg_grads = tf.reduce_mean(grads[:-1], axis=0)    # (T, 1)
    ig = ((x_sample - baseline) * avg_grads)[0].numpy().squeeze()
    return ig


def run_integrated_gradients(model, X_test, y_test):
    """Plot Integrated Gradients for one anomaly and one normal beat."""
    idx_anom = np.where(y_test == 1)[0][0]
    idx_norm = np.where(y_test == 0)[0][0]

    x_anom = X_test[idx_anom : idx_anom + 1]
    x_norm = X_test[idx_norm : idx_norm + 1]

    ig_anom = compute_integrated_gradients(model, x_anom)
    ig_norm = compute_integrated_gradients(model, x_norm)

    print("\n=== Integrated Gradients ===")
    for (signal, ig, title) in [
        (x_anom[0].squeeze(), ig_anom, "Integrated Gradients – Anomaly sample"),
        (x_norm[0].squeeze(), ig_norm, "Integrated Gradients – Normal sample"),
    ]:
        plt.figure(figsize=(10, 3))
        plt.plot(signal, label='ECG Signal', color='black')
        plt.plot(ig,     label='Integrated Gradients', color='red', alpha=0.7)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# ===========================================================================
# 8. Dead Filter Analysis
# ===========================================================================
def run_dead_filter_analysis(model, X_test, n_samples=500):
    """Identify Conv1D filters that are always zero (ReLU dead neurons)."""
    print("\n=== Dead Filter Analysis ===")
    dead_total, total_filters, dead_map = 0, 0, {}

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv1D):
            act_model = tf.keras.Model(model.inputs, layer.output)
            act = act_model.predict(X_test[:n_samples], verbose=0)
            flat = act.reshape(-1, act.shape[-1])
            dead = int(np.sum(np.all(np.isclose(flat, 0), axis=0)))
            dead_total        += dead
            total_filters     += act.shape[-1]
            dead_map[layer.name] = (dead, act.shape[-1])

    print(f"Dead filters: {dead_total}/{total_filters} "
          f"({dead_total / max(total_filters, 1) * 100:.1f}%)")
    for layer_name, (dead, total) in dead_map.items():
        print(f"  {layer_name}: {dead}/{total}")


# ===========================================================================
# 9. Feature Importance via Input Gradients
# ===========================================================================
def run_feature_importance(model, X_test, n_samples=500):
    """Plot mean |gradient| w.r.t. input across the test set."""
    print("\n=== Feature Importance (Mean |Input Gradient|) ===")
    X_batch = tf.cast(X_test[:n_samples], tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(X_batch)
        preds = model(X_batch)[:, 0]

    grads = tape.gradient(preds, X_batch).numpy()
    importance = np.mean(np.abs(grads), axis=0).squeeze()   # (180,)

    plt.figure(figsize=(10, 3))
    plt.plot(importance)
    plt.title("Average |Gradient| over Time – Feature Importance")
    plt.xlabel("Time (samples)")
    plt.ylabel("Importance")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ===========================================================================
# 10. Embedding extraction helpers
# ===========================================================================
def get_embedding_model(model):
    """Return a sub-model that outputs the penultimate Dense layer."""
    dense_layers = [l for l in model.layers
                    if isinstance(l, tf.keras.layers.Dense)]
    # penultimate Dense (second-to-last)
    pen_layer = dense_layers[-2]
    print(f"🔗 Using embedding layer: {pen_layer.name}  "
          f"(output dim: {pen_layer.output_shape[-1]})")
    return tf.keras.Model(model.input, pen_layer.output)


# ===========================================================================
# 11. t-SNE visualisation
# ===========================================================================
def run_tsne(embeddings, y_bin, n_samples=1000):
    """Plot 2-D t-SNE of penultimate-layer embeddings (binary labels)."""
    print("\n=== t-SNE of Embeddings ===")
    idx = np.random.choice(len(embeddings), size=min(n_samples, len(embeddings)),
                            replace=False)
    emb_sub = embeddings[idx]
    lab_sub = y_bin[idx]

    tsne  = TSNE(n_components=2, perplexity=TSNE_PERPLEXITY, random_state=42)
    emb_2d = tsne.fit_transform(emb_sub)

    plt.figure(figsize=(6, 6))
    plt.scatter(emb_2d[lab_sub == 0, 0], emb_2d[lab_sub == 0, 1],
                s=10, alpha=0.6, label='Normal (0)')
    plt.scatter(emb_2d[lab_sub == 1, 0], emb_2d[lab_sub == 1, 1],
                s=10, alpha=0.6, label='Anomaly (1)')
    plt.title("t-SNE of Penultimate-layer Embeddings")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ===========================================================================
# 12. UMAP visualisation
# ===========================================================================
def run_umap(embeddings, y_bin):
    """Plot 2-D UMAP of penultimate-layer embeddings."""
    if not UMAP_AVAILABLE:
        print("⚠️  Skipping UMAP (umap-learn not installed).")
        return
    print("\n=== UMAP of Embeddings ===")
    reducer = umap.UMAP(n_components=2, random_state=42)
    emb_2d  = reducer.fit_transform(embeddings)

    plt.figure(figsize=(6, 6))
    plt.scatter(emb_2d[y_bin == 0, 0], emb_2d[y_bin == 0, 1],
                s=10, alpha=0.6, label='Normal')
    plt.scatter(emb_2d[y_bin == 1, 0], emb_2d[y_bin == 1, 1],
                s=10, alpha=0.6, label='Anomaly')
    plt.title("UMAP – Embedding Space")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ===========================================================================
# 13. KMeans Cluster Purity & Silhouette Score
# ===========================================================================
def run_clustering_metrics(embeddings, y_bin):
    """Fit KMeans (k=2) and report cluster purity + silhouette score."""
    print("\n=== KMeans Clustering Metrics ===")
    kmeans   = KMeans(n_clusters=2, random_state=42, n_init='auto').fit(embeddings)
    clusters = kmeans.labels_

    # purity: pick majority-class assignment per cluster
    purity = max(np.mean(clusters == y_bin), np.mean(clusters != y_bin))
    sil    = silhouette_score(embeddings, y_bin)

    print(f"  Cluster purity (k=2) : {purity:.4f}")
    print(f"  Silhouette score     : {sil:.4f}")


# ===========================================================================
# 14. Per-class t-SNE (symbol-level)
# ===========================================================================
def run_per_class_tsne(embeddings, y_sym):
    """Per-class t-SNE for specific anomaly types vs Normal (N)."""
    print("\n=== Per-class t-SNE (A vs N, V vs N) ===")

    for sym in ['A', 'V']:
        idx_sym  = np.where(y_sym == sym)[0]
        idx_norm = np.where(y_sym == 'N')[0][:len(idx_sym)]

        if len(idx_sym) < 10:
            print(f"⚠️  Skipping '{sym}': only {len(idx_sym)} beats")
            continue

        idx_combined = np.concatenate([idx_sym, idx_norm])
        perp = max(5, min(30, len(idx_combined) // 3))

        tsne   = TSNE(n_components=2, perplexity=perp, random_state=42)
        emb_2d = tsne.fit_transform(embeddings[idx_combined])

        plt.figure(figsize=(6, 6))
        plt.scatter(emb_2d[:len(idx_sym), 0], emb_2d[:len(idx_sym), 1],
                    s=14, alpha=0.7, label=f'{sym} beats')
        plt.scatter(emb_2d[len(idx_sym):, 0], emb_2d[len(idx_sym):, 1],
                    s=14, alpha=0.7, label='Normal (N)')
        plt.title(f"t-SNE – Class {sym} vs Normal  (perplexity={perp})")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# ===========================================================================
# 15. Intra- vs Inter-class Distance Histogram
# ===========================================================================
def run_distance_histogram(embeddings, y_bin, n_samples=1000):
    """Histogram comparing intra-class vs inter-class Euclidean distances."""
    print("\n=== Intra- vs Inter-class Distance Histogram ===")
    sub_idx = np.random.choice(len(embeddings),
                                size=min(n_samples, len(embeddings)),
                                replace=False)
    sub_emb = embeddings[sub_idx]
    sub_lab = y_bin[sub_idx]

    dists = squareform(pdist(sub_emb, metric='euclidean'))
    intra = dists[sub_lab[:, None] == sub_lab[None, :]]
    inter = dists[sub_lab[:, None] != sub_lab[None, :]]

    plt.figure(figsize=(7, 4))
    plt.hist(intra, bins=40, alpha=0.7, label='Intra-class')
    plt.hist(inter, bins=40, alpha=0.7, label='Inter-class')
    plt.title("Euclidean Distance Histogram\n(Intra- vs Inter-class)")
    plt.xlabel("Distance")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ===========================================================================
# 16. Confusion Matrix + Classification Report
# ===========================================================================
def run_evaluation(model, X_test, y_test):
    """Evaluate model performance and display confusion matrix."""
    print("\n=== Model Evaluation ===")
    y_prob = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_prob > THRESHOLD).astype(int)

    print(f"\n--- Classification Report (threshold = {THRESHOLD:.2f}) ---")
    print(classification_report(y_test, y_pred, digits=4,
                                 target_names=["Normal (0)", "Anomaly (1)"]))

    cm   = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Normal (0)", "Anomaly (1)"])
    fig, ax = plt.subplots(figsize=(4, 4))
    disp.plot(ax=ax, colorbar=False)
    plt.title("Confusion Matrix – Test Set")
    plt.tight_layout()
    plt.show()


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    print("=" * 60)
    print("  ECG Arrhythmia — Explainable AI Analysis")
    print("=" * 60)

    # ---- Load model ----
    model = load_model(MODEL_PATH)

    # ---- Load test data (small set for Grad-CAM, etc.) ----
    print("\n⏳ Loading test data (all beats) …")
    X_test, y_test, y_sym_test = extract_beats(TEST_DIR)
    print(f"✅ X_test: {X_test.shape}  |  y_test: {y_test.shape}")

    # ---- Evaluate ----
    run_evaluation(model, X_test, y_test)

    # ---- Grad-CAM (last + second-last Conv1D layer) ----
    run_gradcam_analysis(model, X_test, y_test)

    # ---- All-layer Grad-CAM + Activations ----
    # Set show_activation=False to skip activation heat-maps (faster)
    run_all_layer_analysis(model, X_test, y_test, show_activation=True)

    # ---- Integrated Gradients ----
    run_integrated_gradients(model, X_test, y_test)

    # ---- Dead Filter Analysis ----
    run_dead_filter_analysis(model, X_test, n_samples=500)

    # ---- Feature Importance ----
    run_feature_importance(model, X_test, n_samples=500)

    # ---- Load embedding data (limited to EMBED_LIMIT for speed) ----
    print(f"\n⏳ Loading {EMBED_LIMIT} beats for embedding analysis …")
    X_emb, y_emb, y_sym_emb = extract_beats(TEST_DIR, limit=EMBED_LIMIT)
    print(f"✅ X_emb: {X_emb.shape}")

    # ---- Get embeddings ----
    embed_model = get_embedding_model(model)
    embeddings  = embed_model.predict(X_emb, verbose=0)
    print(f"✅ Embeddings: {embeddings.shape}")

    # ---- t-SNE ----
    run_tsne(embeddings, y_emb)

    # ---- UMAP ----
    run_umap(embeddings, y_emb)

    # ---- KMeans Clustering Metrics ----
    run_clustering_metrics(embeddings, y_emb)

    # ---- Per-class t-SNE ----
    run_per_class_tsne(embeddings, y_sym_emb)

    # ---- Distance Histogram ----
    run_distance_histogram(embeddings, y_emb)

    print("\n✅ All XAI analyses complete.")


if __name__ == "__main__":
    main()

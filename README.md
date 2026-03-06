# ECG Arrhythmia Detection with Explainable AI

> A deep-learning system for detecting cardiac arrhythmias from ECG signals, featuring a 94 % accurate 1-D CNN model, a Flask web interface for real-time prediction, and a comprehensive suite of Explainable AI (XAI) visualisations.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Features](#features)
- [Model](#model)
- [Explainable AI Techniques](#explainable-ai-techniques)
- [Installation](#installation)
- [Running the Web App](#running-the-web-app)
- [Running the Standalone XAI Analysis](#running-the-standalone-xai-analysis)
- [Supported ECG File Formats](#supported-ecg-file-formats)
- [API Reference](#api-reference)
- [Dependencies](#dependencies)

---

## Overview

This project trains and deploys a **1-D Convolutional Neural Network (CNN)** to classify ECG heartbeats as **Normal** or **Arrhythmia** using the MIT-BIH Arrhythmia Database. The model uses a custom **Binary Focal Loss** to handle the severe class imbalance inherent in cardiac data.

Beyond prediction, the system provides full **explainability** — helping clinicians and researchers understand *why* the model made a particular decision through multiple XAI techniques rendered directly in the browser or as matplotlib plots.

---

## Project Structure

```
PSG/
├── BEST MODEL/
│   └── 94_percent_model.keras      # Pre-trained Keras model (94 % accuracy)
├── explainable.py                  # Standalone XAI analysis script
└── website/
    ├── app.py                      # Flask application (backend)
    ├── ecg_utils.py                # ECG loading & beat extraction utilities
    ├── xai_utils.py                # Headless XAI plot generators (web UI)
    ├── requirements.txt            # Python dependencies
    ├── templates/
    │   └── index.html              # Web frontend (single-page UI)
    └── static/
        ├── css/style.css           # Dark-themed stylesheet
        └── js/main.js              # Frontend JavaScript logic
```

---

## Features

### 🩺 Web Interface (`website/`)
- **File upload** — supports WFDB binary records (`.dat` + `.hea`), CSV/TXT/TSV, and NumPy `.npy` files
- **Real-time prediction** — extracts R-peaks with a Pan-Tompkins-inspired detector, classifies each beat, returns an overall label with confidence score
- **Interactive visualisations** — all plots are rendered server-side and streamed as base64 PNGs into the dark-themed UI:
  - Raw ECG signal (first 10 s)
  - Per-beat arrhythmia probability bar chart
  - Normal vs. arrhythmia waveform comparison
  - Grad-CAM overlays (last and second-last Conv1D layers)
  - Integrated Gradients attribution
- **REST API** — JSON endpoint `/predict` for programmatic access

### 🔬 Standalone XAI Analysis (`explainable.py`)
Full, batch XAI analysis pipeline for research and model validation (see [Explainable AI Techniques](#explainable-ai-techniques)).

---

## Model

| Property | Detail |
|---|---|
| Architecture | 1-D CNN with multiple Conv1D blocks |
| Input | 180-sample z-normalised beat windows (±90 samples around R-peak) |
| Output | Binary probability: 0 = Normal, 1 = Arrhythmia |
| Loss function | Binary Focal Loss (γ=2.0, α=0.25) |
| Classification threshold | 0.93 |
| Test accuracy | **~94 %** |
| Dataset | MIT-BIH Arrhythmia Database |

**Normal classes:** `N, L, R, e, j`  
**Arrhythmia classes:** `A, V, E, F, /, f, a, S, J`

The overall record-level label is determined by the **arrhythmia ratio**: if more than 10 % of beats are classified as arrhythmia, the recording is labelled _Arrhythmia_.

---

## Explainable AI Techniques

Both `explainable.py` (standalone) and `xai_utils.py` (web) implement the following:

| # | Technique | Description |
|---|---|---|
| 1 | **Grad-CAM** (last Conv1D) | Highlights time-steps the final convolutional layer focuses on |
| 2 | **Grad-CAM** (2nd-last Conv1D) | Sharper, earlier-feature localisation |
| 3 | **All-layer Grad-CAM + Activation heat-maps** | Per-layer visualisation of both saliency and raw activations |
| 4 | **Integrated Gradients** | Attribute model output to each input time-step via path integrals |
| 5 | **Dead Filter Analysis** | Identifies always-zero (ReLU-dead) Conv1D filters |
| 6 | **Feature Importance (Mean \|Input Gradient\|)** | Average absolute gradient per time sample across the test set |
| 7 | **t-SNE** | 2-D embedding projection of the penultimate Dense layer |
| 8 | **UMAP** | Faster, topology-preserving embedding visualisation |
| 9 | **KMeans Cluster Purity & Silhouette Score** | Quantifies embedding separability |
| 10 | **Per-class t-SNE** | Class A vs N and V vs N scatter plots |
| 11 | **Intra- vs Inter-class Distance Histogram** | Euclidean embedding distance distributions |

---

## Installation

### Prerequisites
- Python ≥ 3.9
- (Recommended) a virtual environment

```bash
# 1. Clone / download the repository
# 2. Install dependencies
pip install -r website/requirements.txt

# Optional — needed for UMAP visualisations in explainable.py
pip install umap-learn
```

---

## Running the Web App

```bash
cd PSG/website
python app.py
```

The server starts on `http://0.0.0.0:5000`. Open `http://localhost:5000` in your browser.

> **Note:** The model is loaded lazily on the **first** `/predict` request; expect a short warm-up delay (~10–20 s on CPU).

### Quick test with curl

```bash
# Upload a WFDB record (both .dat and .hea required)
curl -X POST http://localhost:5000/predict \
     -F "file=@100.dat" \
     -F "file=@100.hea"
```

---

## Running the Standalone XAI Analysis

1. Open `explainable.py` and update the path constants at the top:

```python
TEST_DIR   = r"path/to/your/MIT-BIH/Test"
MODEL_PATH = r"path/to/BEST MODEL/94_percent_model.keras"
```

2. Run:

```bash
python explainable.py
```

All 11 XAI analyses will execute sequentially and display matplotlib windows.

---

## Supported ECG File Formats

| Format | Extension(s) | Notes |
|---|---|---|
| WFDB binary | `.dat` + `.hea` (+ optional `.atr`) | Best quality; upload both files together |
| WFDB `.dat` only | `.dat` | Auto-decoded via MIT-BIH format 212, int16 LE/BE |
| Plain text | `.csv`, `.txt`, `.tsv` | One amplitude value per line |
| NumPy array | `.npy` | Must be a 1-D float array |

---

## API Reference

### `GET /health`
Returns `{"status": "ok"}` — use for readiness checks.

### `POST /predict`
**Body:** `multipart/form-data` with one or more `file` fields.

**Response:**
```json
{
  "prediction":   "Normal | Arrhythmia",
  "confidence":   85.42,
  "n_beats":      312,
  "n_arrhythmia": 5,
  "n_normal":     307,
  "threshold":    0.93,
  "signal_plot":  "data:image/png;base64,...",
  "plots": {
    "beat_prob":             "data:image/png;base64,...",
    "waveforms":             "data:image/png;base64,...",
    "gradcam_last":          "data:image/png;base64,...",
    "gradcam_second":        "data:image/png;base64,...",
    "integrated_gradients":  "data:image/png;base64,..."
  }
}
```

---

## Dependencies

| Package | Version |
|---|---|
| Flask | 3.1.0 |
| flask-cors | 5.0.0 |
| TensorFlow | ≥ 2.15.0 |
| NumPy | ≥ 1.24.0 |
| Matplotlib | ≥ 3.7.0 |
| SciPy | ≥ 1.11.0 |
| wfdb | ≥ 4.1.0 |
| scikit-learn | ≥ 1.3.0 |
| umap-learn | optional |

---

## License

This project is intended for **academic and research purposes**. Please cite the MIT-BIH Arrhythmia Database if you use it in publications:

> Moody GB, Mark RG. The impact of the MIT-BIH Arrhythmia Database. *IEEE Eng in Med and Biol* 20(3):45-50 (May-June 2001).

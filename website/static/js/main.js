/**
 * main.js
 * =======
 * CardioAI – ECG Arrhythmia Detection frontend logic.
 */

"use strict";

// ── Animated ECG canvas (hero) ────────────────────────────────────────────────

(function heroECG() {
  const canvas  = document.getElementById("ecgCanvas");
  if (!canvas) return;
  const ctx     = canvas.getContext("2d");
  canvas.width  = canvas.offsetWidth  || 500;
  canvas.height = 140;

  const ecgShape = [
    0,0,0,0,0,0,0,0,0,0,
    0.1,0.15,0.1,0,           // P wave
    0,-0.1,-0.1,0,            // PQ segment
    -0.2,-0.5,-1,-1.5,        // Q dip
    0.5,2.5,4,2.5,0.5,        // QRS spike
    -1.5,-1,-0.5,-0.2,0,      // S wave
    0,0,0.1,0.2,0.3,0.35,0.3,0.2,0.1,0,  // T wave
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,     // baseline
  ];
  const pat = [...ecgShape, ...ecgShape];

  let offset = 0;

  function draw() {
    canvas.width = canvas.offsetWidth || 500;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const W = canvas.width, H = canvas.height;
    const midY = H / 2, amp  = H * 0.35;
    const step = W / (pat.length - 1);

    ctx.strokeStyle = "#00d4ff";
    ctx.lineWidth   = 2.2;
    ctx.shadowColor = "#00d4ff";
    ctx.shadowBlur  = 8;

    ctx.beginPath();
    for (let i = 0; i < pat.length; i++) {
      const idx = (i + Math.floor(offset)) % pat.length;
      const frac = offset - Math.floor(offset);
      const nextIdx = (idx + 1) % pat.length;
      const y = midY - (pat[idx] * (1 - frac) + pat[nextIdx] * frac) * amp;
      i === 0 ? ctx.moveTo(i * step, y) : ctx.lineTo(i * step, y);
    }
    ctx.stroke();

    // glow trail
    const grad = ctx.createLinearGradient(0, 0, W, 0);
    grad.addColorStop(0,   "rgba(0,212,255,0)");
    grad.addColorStop(0.7, "rgba(0,212,255,0.1)");
    grad.addColorStop(1,   "rgba(0,212,255,0.03)");
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, W, H);

    offset = (offset + 0.35) % pat.length;
    requestAnimationFrame(draw);
  }
  draw();
})();


// ── Loading ECG canvas ────────────────────────────────────────────────────────

(function loadingECG() {
  const canvas = document.getElementById("loadingCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  canvas.width  = 300;
  canvas.height = 60;

  const shape = [0,0,0,0.1,0.15,0.1,0,-0.2,-1,-2,4,-2,-0.5,0,0,0.3,0.3,0.1,0,0,0,0,0];
  let offset = 0;
  function drawLoading() {
    ctx.clearRect(0, 0, 300, 60);
    ctx.strokeStyle = "#00d4ff";
    ctx.lineWidth   = 1.8;
    ctx.shadowColor = "#00d4ff";
    ctx.shadowBlur  = 6;
    const step = 300 / (shape.length - 1);
    ctx.beginPath();
    for (let i = 0; i < shape.length; i++) {
      const idx  = (i + Math.floor(offset)) % shape.length;
      const y    = 30 - shape[idx] * 22;
      i === 0 ? ctx.moveTo(i * step, y) : ctx.lineTo(i * step, y);
    }
    ctx.stroke();
    offset = (offset + 0.5) % shape.length;
    requestAnimationFrame(drawLoading);
  }
  drawLoading();
})();


// ── File upload logic ─────────────────────────────────────────────────────────

const dropZone   = document.getElementById("dropZone");
const fileInput  = document.getElementById("fileInput");
const fileList   = document.getElementById("fileList");
const analyseBtn = document.getElementById("analyseBtn");

let uploadedFiles = [];

function formatBytes(bytes) {
  if (bytes < 1024)       return bytes + " B";
  if (bytes < 1024*1024)  return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / 1024 / 1024).toFixed(2) + " MB";
}

function renderFileList(files) {
  fileList.innerHTML = "";
  if (!files.length) { fileList.classList.add("hidden"); return; }
  fileList.classList.remove("hidden");

  files.forEach(f => {
    const div = document.createElement("div");
    div.className = "file-item";
    div.innerHTML = `
      <span class="file-item-icon">📄</span>
      <span class="file-item-name">${f.name}</span>
      <span class="file-item-size">${formatBytes(f.size)}</span>`;
    fileList.appendChild(div);
  });
}

function onFilesSelected(files) {
  uploadedFiles = Array.from(files);
  renderFileList(uploadedFiles);
  analyseBtn.disabled = uploadedFiles.length === 0;
}

fileInput.addEventListener("change", () => onFilesSelected(fileInput.files));

dropZone.addEventListener("dragover",  e => { e.preventDefault(); dropZone.classList.add("drag-over"); });
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
dropZone.addEventListener("drop", e => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  onFilesSelected(e.dataTransfer.files);
});


// ── Loading overlay steps ─────────────────────────────────────────────────────

function setLoadStep(step) {
  const ids = ["ls1", "ls2", "ls3", "ls4"];
  ids.forEach((id, i) => {
    const el = document.getElementById(id);
    if (i < step)       el.className = "load-step done";
    else if (i === step) el.className = "load-step active";
    else                 el.className = "load-step";
  });
}

function showLoading(show) {
  const overlay = document.getElementById("loadingOverlay");
  show ? overlay.classList.remove("hidden") : overlay.classList.add("hidden");
}


// ── Analyse button ────────────────────────────────────────────────────────────

analyseBtn.addEventListener("click", async () => {
  if (!uploadedFiles.length) return;

  showLoading(true);
  setLoadStep(0);

  const formData = new FormData();
  uploadedFiles.forEach(f => formData.append("file", f));

  // Simulate step progression
  const stepDelays = [600, 900, 1200];
  for (let i = 0; i < stepDelays.length; i++) {
    await delay(stepDelays[i]);
    setLoadStep(i + 1);
  }

  try {
    const resp = await fetch("/predict", { method: "POST", body: formData });
    const data = await resp.json();

    setLoadStep(3);
    await delay(400);
    showLoading(false);

    if (data.error) {
      showError(data.error);
      return;
    }

    renderResults(data);

  } catch (err) {
    showLoading(false);
    showError("Network error: " + err.message);
  }
});


// ── Render results ────────────────────────────────────────────────────────────

function renderResults(data) {
  // Scroll into view
  const resultsSection = document.getElementById("results");
  const xaiSection     = document.getElementById("xai");

  resultsSection.classList.remove("hidden");
  xaiSection.classList.remove("hidden");

  // ─ Verdict card ─
  const isArrhythmia = data.prediction === "Arrhythmia";
  const verdictCard  = document.getElementById("verdictCard");
  verdictCard.className = "verdict-card " + (isArrhythmia ? "arrhythmia" : "normal");

  document.getElementById("verdictLabel").textContent = data.prediction;
  document.getElementById("verdictSub").textContent   = isArrhythmia
    ? `${data.n_arrhythmia} of ${data.n_beats} beats classified as arrhythmic.`
    : `All ${data.n_beats} beats appear normal.`;

  // Verdict icon paths
  const icon = document.getElementById("verdictIcon");
  const path = document.getElementById("verdictPath");
  if (isArrhythmia) {
    icon.setAttribute("viewBox", "0 0 60 60");
    path.setAttribute("d", "M20 20 L40 40 M40 20 L20 40");
  } else {
    path.setAttribute("d", "M18 30 L26 38 L42 22");
  }

  // Confidence ring
  const conf      = Math.max(0, Math.min(100, data.confidence));
  const circumf   = 213.6;
  const offset    = circumf - (conf / 100) * circumf;
  document.getElementById("confArc").setAttribute("stroke-dashoffset", offset);
  document.getElementById("confValue").textContent = conf.toFixed(1) + "%";

  // Summary cards
  document.getElementById("scBeats").textContent  = data.n_beats;
  document.getElementById("scNormal").textContent = data.n_normal;
  document.getElementById("scArrhy").textContent  = data.n_arrhythmia;
  document.getElementById("scThresh").textContent = data.threshold;

  // Signal plot
  if (data.signal_plot) {
    document.getElementById("signalImg").src = data.signal_plot;
  }

  // ─ XAI plots ─
  const plotKeys = {
    "beat_prob":           "img-beat_prob",
    "waveforms":           "img-waveforms",
    "gradcam_last":        "img-gradcam_last",
    "gradcam_second":      "img-gradcam_second",
    "integrated_gradients":"img-integrated_gradients",
  };

  const plots = data.plots || {};
  Object.entries(plotKeys).forEach(([key, imgId]) => {
    const img  = document.getElementById(imgId);
    const naEl = document.getElementById("na-" + key);

    if (plots[key]) {
      img.src = plots[key];
      img.classList.remove("hidden");
      if (naEl) naEl.classList.add("hidden");
    } else {
      img.classList.add("hidden");
      if (naEl) naEl.classList.remove("hidden");
    }
  });

  // Smooth scroll
  setTimeout(() => {
    resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
  }, 100);
}


// ── XAI tab switching ─────────────────────────────────────────────────────────

document.querySelectorAll(".xai-tab").forEach(tab => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".xai-tab").forEach(t => t.classList.remove("active"));
    document.querySelectorAll(".xai-panel").forEach(p => p.classList.remove("active"));
    tab.classList.add("active");
    document.getElementById("panel-" + tab.dataset.tab).classList.add("active");
  });
});


// ── Error helper ──────────────────────────────────────────────────────────────

function showError(msg) {
  // Simple toast notification
  const toast = document.createElement("div");
  toast.style.cssText = `
    position:fixed; bottom:24px; right:24px; z-index:999;
    background:#ff4b6e22; border:1px solid #ff4b6e;
    color:#ff4b6e; padding:14px 20px; border-radius:12px;
    font-size:0.88rem; max-width:360px; line-height:1.5;
    box-shadow:0 4px 24px rgba(255,75,110,0.2);
    animation: slideIn 0.3s ease;
  `;
  toast.innerHTML = `<strong>⚠ Error</strong><br>${msg}`;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 7000);
}

function delay(ms) { return new Promise(res => setTimeout(res, ms)); }

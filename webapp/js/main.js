/**
 * main.js — Camera loop, UI, orchestration
 */

import { initFaceLandmarker, detectFace } from "./face.js";
import {
  pushFrame, bufferLengthSec, resetBuffers,
  computeHeartRate, computeRespRate,
  FOREHEAD_IDX, LEFT_CHEEK_IDX, RIGHT_CHEEK_IDX,
} from "./rppg.js";

// ─── DOM refs ─────────────────────────────────────────────────────────────────
const video         = document.getElementById("cam");
const canvas        = document.getElementById("overlay");
const ctx           = canvas.getContext("2d");
const bpmEl         = document.getElementById("bpm");
const rpmEl         = document.getElementById("rpm");
const snrEl         = document.getElementById("snr");
const statusEl      = document.getElementById("status");
const startBtn      = document.getElementById("start-btn");
const stopBtn       = document.getElementById("stop-btn");
const loadingEl     = document.getElementById("loading");
const qualFill      = document.getElementById("quality-fill");
const summaryEl     = document.getElementById("summary");
const summaryItems  = document.getElementById("summary-items");
const restartBtn    = document.getElementById("restart-btn");

// ─── State ───────────────────────────────────────────────────────────────────
let running         = false;
let lastHeartResult = null;
let lastRespResult  = null;
let currentStream   = null;

// EMA smoothing
let smoothBpm = null;
let smoothRpm = null;
const EMA_ALPHA_HEART = 0.15;
const EMA_ALPHA_RESP  = 0.10;
const SNR_MIN_DB      = 2.0;

// Frontal flash (low-light boost)
const BRIGHTNESS_THRESHOLD = 80;  // 0-255, below this triggers flash
const BRIGHTNESS_EMA_ALPHA = 0.1;
let smoothBrightness = null;
let flashActive      = false;      // one-way latch per session

// History for summary (sampled every ~5s when valid)
let heartHistory = [];   // [{bpm, snrDb}]
let respHistory  = [];   // [{rpm}]
let sessionStart = null;

// Offscreen canvas for pixel extraction (video native resolution)
const offscreen = new OffscreenCanvas(1, 1);
const offCtx    = offscreen.getContext("2d", { willReadFrequently: true });

// ─── object-fit: cover mapping ────────────────────────────────────────────────
// The video is CSS object-fit:cover inside the viewport.
// We need to know the scale + offset used by the browser to display the video,
// so we can map landmark coords (0..1 in video space) → canvas pixel coords.
//
// object-fit: cover uses scale = max(dispW/vidW, dispH/vidH), centered.

function getCoverTransform() {
  const dispW = canvas.width;   // = window.innerWidth
  const dispH = canvas.height;  // = window.innerHeight
  const vidW  = video.videoWidth;
  const vidH  = video.videoHeight;
  if (!vidW || !vidH) return null;

  const scale  = Math.max(dispW / vidW, dispH / vidH);
  const scaledW = vidW * scale;
  const scaledH = vidH * scale;
  const offsetX = (dispW - scaledW) / 2;
  const offsetY = (dispH - scaledH) / 2;
  return { scale, offsetX, offsetY, vidW, vidH };
}

// Map a landmark (lm.x, lm.y in [0,1]) to canvas display coords.
// The canvas has CSS scaleX(-1) (mirror), so we mirror the x coord
// BEFORE drawing — so that the drawn content aligns with the mirrored video.
function lmToCanvas(lm, t) {
  // First map to display coords (non-mirrored)
  const x = lm.x * t.vidW * t.scale + t.offsetX;
  const y = lm.y * t.vidH * t.scale + t.offsetY;
  // Mirror x to match the CSS-mirrored display
  return { x: canvas.width - x, y };
}

// ─── ROI extraction ───────────────────────────────────────────────────────────
// Pixel extraction uses the offscreen canvas which has the video at native
// resolution (no CSS transforms). Landmarks coords are [0,1] × vidW/H.

function extractRoiMean(landmarks, idxList) {
  const vw = video.videoWidth, vh = video.videoHeight;
  let xMin = Infinity, yMin = Infinity, xMax = 0, yMax = 0;
  for (const idx of idxList) {
    const lm = landmarks[idx];
    const x = Math.round(Math.max(0, Math.min(lm.x * vw, vw - 1)));
    const y = Math.round(Math.max(0, Math.min(lm.y * vh, vh - 1)));
    if (x < xMin) xMin = x; if (x > xMax) xMax = x;
    if (y < yMin) yMin = y; if (y > yMax) yMax = y;
  }

  const rw = xMax - xMin + 1, rh = yMax - yMin + 1;
  if (rw < 1 || rh < 1) return null;

  const imgData = offCtx.getImageData(xMin, yMin, rw, rh);
  const data = imgData.data;
  let sumR = 0, sumG = 0, sumB = 0;
  const total = rw * rh;
  for (let i = 0; i < total; i++) {
    sumR += data[i * 4];
    sumG += data[i * 4 + 1];
    sumB += data[i * 4 + 2];
  }
  return { r: sumR / total, g: sumG / total, b: sumB / total };
}

function extractCombinedRoi(landmarks) {
  const fore  = extractRoiMean(landmarks, FOREHEAD_IDX);
  const left  = extractRoiMean(landmarks, LEFT_CHEEK_IDX);
  const right = extractRoiMean(landmarks, RIGHT_CHEEK_IDX);

  const rois = [fore, left, right].filter(Boolean);
  if (rois.length === 0) return null;

  let r = 0, g = 0, b = 0;
  for (const roi of rois) { r += roi.r; g += roi.g; b += roi.b; }
  return { r: r / rois.length, g: g / rois.length, b: b / rois.length };
}

// ─── Brightness detection ────────────────────────────────────────────────────

function measureBrightness() {
  const vw = video.videoWidth, vh = video.videoHeight;
  if (!vw || !vh) return null;

  // Sample a small center region for speed (100x100 or less)
  const sw = Math.min(100, vw), sh = Math.min(100, vh);
  const sx = Math.floor((vw - sw) / 2), sy = Math.floor((vh - sh) / 2);
  const imgData = offCtx.getImageData(sx, sy, sw, sh);
  const d = imgData.data;
  let sum = 0;
  const total = sw * sh;
  for (let i = 0; i < total; i++) {
    sum += 0.299 * d[i * 4] + 0.587 * d[i * 4 + 1] + 0.114 * d[i * 4 + 2];
  }
  return sum / total;
}

// ─── Flash overlay (vertical ellipse with white surround) ───────────────────

function drawFlashOverlay() {
  const w = canvas.width, h = canvas.height;
  const cx = w / 2, cy = h / 2;
  const rx = w * 0.30;  // ellipse horizontal radius (~60% width)
  const ry = h * 0.40;  // ellipse vertical radius (~80% height)

  // Fill entire canvas white
  ctx.save();
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, w, h);

  // Cut out the ellipse (make it transparent to show camera feed)
  ctx.globalCompositeOperation = "destination-out";
  ctx.beginPath();
  ctx.ellipse(cx, cy, rx, ry, 0, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

// ─── Canvas overlay drawing ───────────────────────────────────────────────────

function drawRoiOverlay(landmarks, t, idxList, color) {
  // Bounding box in canvas display coords
  let xMin = Infinity, yMin = Infinity, xMax = -Infinity, yMax = -Infinity;
  for (const idx of idxList) {
    const p = lmToCanvas(landmarks[idx], t);
    if (p.x < xMin) xMin = p.x; if (p.x > xMax) xMax = p.x;
    if (p.y < yMin) yMin = p.y; if (p.y > yMax) yMax = p.y;
  }

  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.strokeRect(xMin, yMin, xMax - xMin, yMax - yMin);

  ctx.fillStyle = color;
  for (const idx of idxList) {
    const p = lmToCanvas(landmarks[idx], t);
    ctx.beginPath();
    ctx.arc(p.x, p.y, 3, 0, Math.PI * 2);
    ctx.fill();
  }
}

function drawOverlay(landmarks, snrDb) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!landmarks) return;

  const t = getCoverTransform();
  if (!t) return;

  drawRoiOverlay(landmarks, t, FOREHEAD_IDX,   "rgba(255,230,0,0.85)");
  drawRoiOverlay(landmarks, t, LEFT_CHEEK_IDX,  "rgba(0,255,120,0.85)");
  drawRoiOverlay(landmarks, t, RIGHT_CHEEK_IDX, "rgba(0,220,255,0.85)");

  // Quality dot (top-right)
  const qualColor = snrDb === null ? "#888"
    : snrDb > 3 ? "#4ade80"
    : snrDb > 0 ? "#facc15"
    : "#f87171";
  ctx.fillStyle = qualColor;
  ctx.beginPath();
  ctx.arc(canvas.width - 20, 20, 8, 0, Math.PI * 2);
  ctx.fill();
}

// ─── Resize canvas to match viewport ─────────────────────────────────────────

function resizeCanvas() {
  canvas.width  = window.innerWidth;
  canvas.height = window.innerHeight;
}

window.addEventListener("resize", resizeCanvas);

// ─── UI updates ───────────────────────────────────────────────────────────────

function updateQualityBar(snrDb) {
  if (snrDb === null) { qualFill.style.width = "0%"; return; }
  const pct = Math.max(0, Math.min(100, (snrDb + 5) / 20 * 100));
  qualFill.style.width = pct + "%";
  qualFill.style.background = snrDb > 3 ? "#4ade80" : snrDb > 0 ? "#facc15" : "#f87171";
}

function updateMetrics() {
  const bufSec = bufferLengthSec();

  if (smoothBpm !== null && lastHeartResult) {
    bpmEl.textContent = Math.round(smoothBpm);
    snrEl.textContent = lastHeartResult.snrDb.toFixed(1) + " dB";
    updateQualityBar(lastHeartResult.snrDb);
  } else {
    bpmEl.textContent = "--";
    snrEl.textContent = "--";
    updateQualityBar(null);
  }

  rpmEl.textContent = smoothRpm !== null ? Math.round(smoothRpm) : "--";

  if (bufSec < 2) {
    statusEl.textContent = "Posicione o rosto na câmera...";
  } else if (bufSec < 6) {
    statusEl.textContent = `Coletando dados... ${Math.ceil(6 - bufSec)}s`;
  } else if (!lastHeartResult) {
    statusEl.textContent = "Calculando...";
  } else {
    const q = lastHeartResult.snrDb > 3 ? "Boa" : lastHeartResult.snrDb > 0 ? "Regular" : "Fraca";
    statusEl.textContent = `Qualidade: ${q} | ${bufSec.toFixed(0)}s coletados`;
  }
}

// ─── Main loop ────────────────────────────────────────────────────────────────

let lastComputeMs  = 0;
let lastSampleMs   = 0;
const COMPUTE_INTERVAL_MS = 1000;
const SAMPLE_INTERVAL_MS  = 5000;

async function processFrame(timestampMs) {
  if (!running) return;

  const vw = video.videoWidth, vh = video.videoHeight;
  if (!vw || !vh) { requestAnimationFrame(processFrame); return; }

  // Sync offscreen canvas to video native resolution
  if (offscreen.width !== vw || offscreen.height !== vh) {
    offscreen.width = vw; offscreen.height = vh;
  }

  // Draw video to offscreen for pixel extraction (no CSS transforms)
  offCtx.drawImage(video, 0, 0, vw, vh);

  // Check brightness and latch flash if too dark
  if (!flashActive) {
    const brightness = measureBrightness();
    if (brightness !== null) {
      smoothBrightness = smoothBrightness === null
        ? brightness
        : BRIGHTNESS_EMA_ALPHA * brightness + (1 - BRIGHTNESS_EMA_ALPHA) * smoothBrightness;
      if (smoothBrightness < BRIGHTNESS_THRESHOLD) {
        flashActive = true;
      }
    }
  }

  // Detect face (pass video element; MediaPipe reads at native resolution)
  const landmarks = detectFace(video, timestampMs);

  if (landmarks) {
    const rgb = extractCombinedRoi(landmarks);
    if (rgb) pushFrame(rgb.r, rgb.g, rgb.b, timestampMs);
  }

  drawOverlay(landmarks, lastHeartResult?.snrDb ?? null);

  // Draw flash overlay on top if latched
  if (flashActive) {
    drawFlashOverlay();
  }

  if (timestampMs - lastComputeMs > COMPUTE_INTERVAL_MS) {
    lastComputeMs = timestampMs;

    const heart = computeHeartRate();
    if (heart) {
      lastHeartResult = heart;
      if (heart.snrDb >= SNR_MIN_DB) {
        smoothBpm = smoothBpm === null
          ? heart.bpm
          : EMA_ALPHA_HEART * heart.bpm + (1 - EMA_ALPHA_HEART) * smoothBpm;
      }
    }

    const resp = computeRespRate();
    if (resp) {
      lastRespResult = resp;
      if (resp.snrDb >= SNR_MIN_DB) {
        smoothRpm = smoothRpm === null
          ? resp.rpm
          : EMA_ALPHA_RESP * resp.rpm + (1 - EMA_ALPHA_RESP) * smoothRpm;
      }
    }

    updateMetrics();
  }

  // Sample history every 5s for summary
  if (timestampMs - lastSampleMs > SAMPLE_INTERVAL_MS) {
    lastSampleMs = timestampMs;
    if (lastHeartResult) heartHistory.push({ bpm: lastHeartResult.bpm, snrDb: lastHeartResult.snrDb });
    if (lastRespResult)  respHistory.push({ rpm: lastRespResult.rpm });
  }

  requestAnimationFrame(processFrame);
}

// ─── Stop + Summary ───────────────────────────────────────────────────────────

function stopCamera() {
  if (currentStream) {
    currentStream.getTracks().forEach(t => t.stop());
    currentStream = null;
  }
  video.srcObject = null;
}

function showSummary() {
  const sessionSec = sessionStart ? Math.round((Date.now() - sessionStart) / 1000) : 0;
  const mins = Math.floor(sessionSec / 60), secs = sessionSec % 60;
  const durationStr = mins > 0 ? `${mins}m ${secs}s` : `${secs}s`;

  // Compute averages
  const avgBpm = heartHistory.length
    ? Math.round(heartHistory.reduce((a, b) => a + b.bpm, 0) / heartHistory.length)
    : null;
  const avgSnr = heartHistory.length
    ? (heartHistory.reduce((a, b) => a + b.snrDb, 0) / heartHistory.length).toFixed(1)
    : null;
  const avgRpm = respHistory.length
    ? Math.round(respHistory.reduce((a, b) => a + b.rpm, 0) / respHistory.length)
    : null;

  // Use last known values as fallback if history is short
  const bpmVal  = avgBpm  ?? (lastHeartResult ? Math.round(lastHeartResult.bpm) : null);
  const snrVal  = avgSnr  ?? (lastHeartResult ? lastHeartResult.snrDb.toFixed(1) : null);
  const rpmVal  = avgRpm  ?? (lastRespResult  ? Math.round(lastRespResult.rpm)   : null);
  const quality = snrVal !== null
    ? (parseFloat(snrVal) > 3 ? "Boa" : parseFloat(snrVal) > 0 ? "Regular" : "Fraca")
    : "—";

  summaryItems.innerHTML = `
    <div class="summary-row">
      <div>
        <div class="s-label">Duração</div>
      </div>
      <div style="text-align:right">
        <span class="s-value" style="font-size:24px">${durationStr}</span>
      </div>
    </div>
    <div class="summary-row">
      <div>
        <div class="s-label">Frequência Cardíaca</div>
        <div class="s-sub">Média da sessão</div>
      </div>
      <div style="text-align:right">
        <span class="s-value">${bpmVal ?? "—"}</span>
        ${bpmVal ? '<span class="s-unit">bpm</span>' : ""}
      </div>
    </div>
    <div class="summary-row">
      <div>
        <div class="s-label">Respiração</div>
        <div class="s-sub">Média da sessão</div>
      </div>
      <div style="text-align:right">
        <span class="s-value">${rpmVal ?? "—"}</span>
        ${rpmVal ? '<span class="s-unit">rpm</span>' : ""}
      </div>
    </div>
    <div class="summary-row">
      <div>
        <div class="s-label">Qualidade do Sinal</div>
        <div class="s-sub">${snrVal !== null ? `SNR médio: ${snrVal} dB` : "Dados insuficientes"}</div>
      </div>
      <div style="text-align:right">
        <span class="s-value" style="font-size:20px">${quality}</span>
      </div>
    </div>
  `;

  // Hide live UI, show summary
  summaryEl.hidden = false;
  stopBtn.hidden = true;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function doStop() {
  running = false;
  stopCamera();
  showSummary();
}

stopBtn.addEventListener("click", doStop);

restartBtn.addEventListener("click", () => {
  summaryEl.hidden = true;
  bpmEl.textContent = "--";
  rpmEl.textContent = "--";
  snrEl.textContent = "--";
  qualFill.style.width = "0%";
  statusEl.textContent = "Toque em Iniciar para começar";
  startBtn.hidden = false;
});

// ─── Init ─────────────────────────────────────────────────────────────────────

resizeCanvas();

startBtn.addEventListener("click", async () => {
  startBtn.hidden = true;
  loadingEl.hidden = false;
  statusEl.textContent = "Inicializando...";

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
    });
    currentStream = stream;
    video.srcObject = stream;
    await new Promise((res) => { video.onloadedmetadata = res; });
    await video.play();

    await initFaceLandmarker((msg) => { statusEl.textContent = msg; });

    loadingEl.hidden = true;
    running = true;
    lastHeartResult = null;
    lastRespResult  = null;
    heartHistory    = [];
    respHistory     = [];
    lastComputeMs   = 0;
    lastSampleMs    = 0;
    sessionStart    = Date.now();
    smoothBpm        = null;
    smoothRpm        = null;
    smoothBrightness = null;
    flashActive      = false;
    resetBuffers();
    stopBtn.hidden = false;
    requestAnimationFrame(processFrame);
  } catch (err) {
    loadingEl.hidden = true;
    startBtn.hidden = false;
    statusEl.textContent = "Erro: " + (err.message || err);
    console.error(err);
  }
});

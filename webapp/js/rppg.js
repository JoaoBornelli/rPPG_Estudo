/**
 * rppg.js — CHROM signal processing + rPPG pipeline
 * Implements CHROM as per de Haan & Jeanne 2013.
 */

import { normalize, detrendLinear, bandpassFft, welchPsd, estimateRate, mean, std } from "./dsp.js";

// ─── ROI landmark indices (same as Python source) ────────────────────────────
export const FOREHEAD_IDX  = [54, 10, 67, 103, 109, 338, 297, 332, 284];
export const LEFT_CHEEK_IDX  = [117, 118, 50, 205, 187, 147, 213, 192];
export const RIGHT_CHEEK_IDX = [346, 347, 280, 425, 411, 376, 433, 416];

// ─── Constants ───────────────────────────────────────────────────────────────
const HEART_WINDOW_SEC     = 12.0;
const MIN_HEART_WINDOW_SEC = 6.0;
const RESP_WINDOW_SEC      = 30.0;
const MIN_RESP_WINDOW_SEC  = 12.0;
const HEART_BAND_HZ        = [0.8, 3.2];
const RESP_BAND_HZ         = [0.1, 0.5];
const WELCH_SEG_SEC_HEART  = 5.0;
const WELCH_OVERLAP_HEART  = 0.5;
const WELCH_SEG_SEC_RESP   = 20.0;
const WELCH_OVERLAP_RESP   = 0.75;
const ALPHA_WINDOW_SEC     = 1.6;
const MAX_BUFFER_SEC       = 35.0;

// ─── Signal buffers ──────────────────────────────────────────────────────────
let rBuf = [], gBuf = [], bBuf = [], tsBuf = [];

export function pushFrame(meanR, meanG, meanB, timestampMs) {
  rBuf.push(meanR); gBuf.push(meanG); bBuf.push(meanB); tsBuf.push(timestampMs);
  // trim to max window
  while (tsBuf.length > 1 && (tsBuf[tsBuf.length - 1] - tsBuf[0]) > MAX_BUFFER_SEC * 1000) {
    rBuf.shift(); gBuf.shift(); bBuf.shift(); tsBuf.shift();
  }
}

export function bufferLengthSec() {
  if (tsBuf.length < 2) return 0;
  return (tsBuf[tsBuf.length - 1] - tsBuf[0]) / 1000;
}

export function resetBuffers() {
  rBuf = []; gBuf = []; bBuf = []; tsBuf = [];
}

// ─── Sampling rate ───────────────────────────────────────────────────────────
function samplingRate(tsArr) {
  if (tsArr.length < 2) return null;
  let sum = 0;
  for (let i = 1; i < tsArr.length; i++) sum += tsArr[i] - tsArr[i - 1];
  const avgDt = sum / (tsArr.length - 1) / 1000; // seconds
  return avgDt > 0 ? 1 / avgDt : null;
}

// ─── CHROM — de Haan & Jeanne 2013 ──────────────────────────────────────────
/**
 * Corrected CHROM with per-frame normalization (div by mean) and
 * alpha computed in sub-windows of ALPHA_WINDOW_SEC.
 */
function chromSignal(r, g, b, fs) {
  const n = r.length;
  const alphaWin = Math.max(4, Math.round(ALPHA_WINDOW_SEC * fs));
  const signal = new Float64Array(n);

  for (let start = 0; start < n; start += alphaWin) {
    const end = Math.min(start + alphaWin, n);
    const len = end - start;
    if (len < 2) break;

    // DC component of sub-window
    let muR = 0, muG = 0, muB = 0;
    for (let i = start; i < end; i++) { muR += r[i]; muG += g[i]; muB += b[i]; }
    muR /= len; muG /= len; muB /= len;
    if (muR < 1e-6) muR = 1e-6;
    if (muG < 1e-6) muG = 1e-6;
    if (muB < 1e-6) muB = 1e-6;

    // Normalize by dividing by mean (remove DC illumination)
    const rn = new Float64Array(len), gn = new Float64Array(len), bn = new Float64Array(len);
    for (let i = 0; i < len; i++) {
      rn[i] = r[start + i] / muR - 1;
      gn[i] = g[start + i] / muG - 1;
      bn[i] = b[start + i] / muB - 1;
    }

    // CHROM projection
    const xs = new Float64Array(len), ys = new Float64Array(len);
    for (let i = 0; i < len; i++) {
      xs[i] = 3 * rn[i] - 2 * gn[i];
      ys[i] = 1.5 * rn[i] + gn[i] - 1.5 * bn[i];
    }

    // Alpha for this sub-window
    const alpha = std(xs) / (std(ys) + 1e-8);

    for (let i = 0; i < len; i++) {
      signal[start + i] = xs[i] - alpha * ys[i];
    }
  }

  return normalize(signal);
}

// ─── Window extraction ───────────────────────────────────────────────────────
function extractWindow(windowSec) {
  const n = tsBuf.length;
  if (n < 2) return null;
  const cutMs = tsBuf[n - 1] - windowSec * 1000;
  let startIdx = 0;
  while (startIdx < n - 1 && tsBuf[startIdx] < cutMs) startIdx++;
  const ts = tsBuf.slice(startIdx);
  if (ts.length < 2) return null;
  return {
    r: Float64Array.from(rBuf.slice(startIdx)),
    g: Float64Array.from(gBuf.slice(startIdx)),
    b: Float64Array.from(bBuf.slice(startIdx)),
    ts,
  };
}

// ─── Public API ──────────────────────────────────────────────────────────────

export function computeHeartRate() {
  const win = extractWindow(HEART_WINDOW_SEC);
  if (!win) return null;

  const durSec = (win.ts[win.ts.length - 1] - win.ts[0]) / 1000;
  if (durSec < MIN_HEART_WINDOW_SEC) return null;

  const fs = samplingRate(win.ts);
  if (!fs || fs < 1) return null;

  let sig = chromSignal(win.r, win.g, win.b, fs);
  sig = detrendLinear(sig);
  sig = bandpassFft(sig, fs, HEART_BAND_HZ[0], HEART_BAND_HZ[1]);
  sig = normalize(sig);

  const psdResult = welchPsd(sig, fs, WELCH_SEG_SEC_HEART, WELCH_OVERLAP_HEART);
  const est = estimateRate(psdResult, HEART_BAND_HZ);
  if (!est) return null;
  return { bpm: est.rate, snrDb: est.snrDb, peakHz: est.peakHz, fs };
}

export function computeRespRate() {
  const win = extractWindow(RESP_WINDOW_SEC);
  if (!win) return null;

  const durSec = (win.ts[win.ts.length - 1] - win.ts[0]) / 1000;
  if (durSec < MIN_RESP_WINDOW_SEC) return null;

  const fs = samplingRate(win.ts);
  if (!fs || fs < 1) return null;

  let sig = chromSignal(win.r, win.g, win.b, fs);
  sig = detrendLinear(sig);
  sig = bandpassFft(sig, fs, RESP_BAND_HZ[0], RESP_BAND_HZ[1]);
  sig = normalize(sig);

  const psdResult = welchPsd(sig, fs, WELCH_SEG_SEC_RESP, WELCH_OVERLAP_RESP);
  const est = estimateRate(psdResult, RESP_BAND_HZ);
  if (!est) return null;
  return { rpm: est.rate, snrDb: est.snrDb, peakHz: est.peakHz };
}

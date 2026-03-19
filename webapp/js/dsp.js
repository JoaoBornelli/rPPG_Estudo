/**
 * dsp.js — Pure JS DSP engine for rPPG
 * All functions use Float64Array for precision.
 */

// ─── Math helpers ────────────────────────────────────────────────────────────

export function mean(arr) {
  let s = 0;
  for (let i = 0; i < arr.length; i++) s += arr[i];
  return s / arr.length;
}

export function std(arr) {
  const m = mean(arr);
  let v = 0;
  for (let i = 0; i < arr.length; i++) v += (arr[i] - m) ** 2;
  return Math.sqrt(v / arr.length);
}

// ─── Normalize (z-score) ─────────────────────────────────────────────────────

export function normalize(x) {
  const s = std(x);
  if (s < 1e-8) return new Float64Array(x.length);
  const m = mean(x);
  const out = new Float64Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = (x[i] - m) / s;
  return out;
}

// ─── Linear detrend ──────────────────────────────────────────────────────────

export function detrendLinear(signal) {
  const n = signal.length;
  if (n < 3) return signal;
  // least-squares fit of degree-1 polynomial
  let sx = 0, sy = 0, sxy = 0, sx2 = 0;
  for (let i = 0; i < n; i++) {
    sx += i; sy += signal[i]; sxy += i * signal[i]; sx2 += i * i;
  }
  const denom = n * sx2 - sx * sx;
  if (Math.abs(denom) < 1e-12) return signal;
  const slope = (n * sxy - sx * sy) / denom;
  const intercept = (sy - slope * sx) / n;
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) out[i] = signal[i] - (slope * i + intercept);
  return out;
}

// ─── FFT (radix-2 Cooley-Tukey, in-place) ────────────────────────────────────

function nextPow2(n) {
  let p = 1;
  while (p < n) p <<= 1;
  return p;
}

/**
 * In-place FFT on re[] and im[] of length N (must be power of 2).
 * inverse=false → forward FFT.
 */
function fftInPlace(re, im, inverse) {
  const N = re.length;
  // bit-reversal permutation
  let j = 0;
  for (let i = 1; i < N; i++) {
    let bit = N >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [re[i], re[j]] = [re[j], re[i]];
      [im[i], im[j]] = [im[j], im[i]];
    }
  }
  // butterfly
  const sign = inverse ? 1 : -1;
  for (let len = 2; len <= N; len <<= 1) {
    const ang = sign * 2 * Math.PI / len;
    const wRe = Math.cos(ang), wIm = Math.sin(ang);
    for (let i = 0; i < N; i += len) {
      let uRe = 1, uIm = 0;
      for (let k = 0; k < len / 2; k++) {
        const aRe = re[i + k], aIm = im[i + k];
        const bRe = re[i + k + len/2] * uRe - im[i + k + len/2] * uIm;
        const bIm = re[i + k + len/2] * uIm + im[i + k + len/2] * uRe;
        re[i + k] = aRe + bRe; im[i + k] = aIm + bIm;
        re[i + k + len/2] = aRe - bRe; im[i + k + len/2] = aIm - bIm;
        const nuRe = uRe * wRe - uIm * wIm;
        uIm = uRe * wIm + uIm * wRe;
        uRe = nuRe;
      }
    }
  }
  if (inverse) {
    for (let i = 0; i < N; i++) { re[i] /= N; im[i] /= N; }
  }
}

/**
 * rfft — real FFT. Returns {re, im} of length N/2+1.
 * Input is zero-padded to next power of 2.
 */
export function rfft(signal) {
  const N = nextPow2(signal.length);
  const re = new Float64Array(N);
  const im = new Float64Array(N);
  for (let i = 0; i < signal.length; i++) re[i] = signal[i];
  fftInPlace(re, im, false);
  // return only first N/2+1 bins
  const half = N / 2 + 1;
  return { re: re.slice(0, half), im: im.slice(0, half), N };
}

/**
 * irfft — inverse real FFT. Returns real signal of length n.
 * spec = {re, im, N} from rfft.
 */
export function irfft(spec, n) {
  const N = spec.N;
  const re = new Float64Array(N);
  const im = new Float64Array(N);
  // reconstruct full spectrum (conjugate symmetry)
  const half = N / 2 + 1;
  for (let i = 0; i < half; i++) {
    re[i] = spec.re[i]; im[i] = spec.im[i];
  }
  for (let i = half; i < N; i++) {
    re[i] = spec.re[N - i]; im[i] = -spec.im[N - i];
  }
  fftInPlace(re, im, true);
  return re.slice(0, n);
}

/**
 * rfftfreq — frequency bins matching rfft output.
 * d = 1/fs (sample spacing in seconds).
 */
export function rfftfreq(n, d) {
  const N = nextPow2(n);
  const half = N / 2 + 1;
  const freqs = new Float64Array(half);
  for (let i = 0; i < half; i++) freqs[i] = i / (N * d);
  return freqs;
}

// ─── Bandpass filter (ideal FFT-based) ───────────────────────────────────────

export function bandpassFft(signal, fs, lowHz, highHz) {
  if (signal.length < 4) return signal;
  const n = signal.length;
  const spec = rfft(signal);
  const freqs = rfftfreq(n, 1 / fs);
  for (let i = 0; i < freqs.length; i++) {
    if (freqs[i] < lowHz || freqs[i] > highHz) {
      spec.re[i] = 0; spec.im[i] = 0;
    }
  }
  return irfft(spec, n);
}

// ─── Welch PSD ───────────────────────────────────────────────────────────────

/**
 * Returns { freqs: Float64Array, psd: Float64Array } or null.
 */
export function welchPsd(signal, fs, segSec, overlap) {
  const n = signal.length;
  if (n < 16) return null;

  let nperseg = Math.round(segSec * fs);
  nperseg = Math.max(32, Math.min(nperseg, n));
  // ensure power of 2 for FFT efficiency
  nperseg = nextPow2(nperseg);

  const step = Math.max(1, Math.round(nperseg * (1 - overlap)));
  if (n < nperseg) return null;

  // Hann window
  const win = new Float64Array(nperseg);
  for (let i = 0; i < nperseg; i++) win[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / (nperseg - 1)));
  const winPow = win.reduce((a, b) => a + b * b, 0) + 1e-12;

  const freqs = rfftfreq(nperseg, 1 / fs);
  const acc = new Float64Array(freqs.length);
  let count = 0;

  for (let start = 0; start + nperseg <= n; start += step) {
    const seg = new Float64Array(nperseg);
    let segMean = 0;
    for (let i = 0; i < nperseg; i++) segMean += signal[start + i];
    segMean /= nperseg;
    for (let i = 0; i < nperseg; i++) seg[i] = (signal[start + i] - segMean) * win[i];

    const spec = rfft(seg);
    for (let i = 0; i < freqs.length; i++) {
      acc[i] += (spec.re[i] ** 2 + spec.im[i] ** 2) / winPow;
    }
    count++;
  }

  if (count === 0) return null;
  const psd = new Float64Array(acc.length);
  for (let i = 0; i < acc.length; i++) psd[i] = acc[i] / count;
  return { freqs, psd };
}

// ─── Rate estimation ─────────────────────────────────────────────────────────

/**
 * Returns { rate (per min), snrDb, peakHz } or null.
 */
export function estimateRate(welchResult, bandHz) {
  if (!welchResult) return null;
  const { freqs, psd } = welchResult;
  const [low, high] = bandHz;

  let peakPow = -Infinity, peakHz = null, totalPow = 0, count = 0;
  for (let i = 0; i < freqs.length; i++) {
    if (freqs[i] < low || freqs[i] > high) continue;
    totalPow += psd[i];
    count++;
    if (psd[i] > peakPow) { peakPow = psd[i]; peakHz = freqs[i]; }
  }

  if (peakHz === null || count === 0) return null;
  const noisePow = (totalPow - peakPow) / Math.max(count - 1, 1);
  const snrDb = 10 * Math.log10((peakPow + 1e-12) / (noisePow + 1e-12));
  return { rate: peakHz * 60, snrDb, peakHz };
}

"""
rPPG via CHROM (de Haan & Jeanne 2013) sobre ROI testa+bochechas.

Pipeline:
  feed_frame(rgb_frame, landmarks, timestamp_ms)
    → acumula R/G/B médios em buffers temporais
    → recompute() periódico (a cada 1s):
        - CHROM signal
        - bandpass (FFT ideal — TODO: Butterworth zero-phase)
        - Welch PSD → BPM (banda 0.8-3.2 Hz)
        - filtro EMA por SNR
        - peaks no domínio do tempo → IBI buffer (NOVO, para HRV)
"""
import cv2 as cv
import numpy as np
from scipy.signal import find_peaks

from modules.capture import landmark_points

# ROI landmarks (testa + bochechas bilaterais)
FOREHEAD_IDX = [54, 10, 67, 103, 109, 338, 297, 332, 284]
LEFT_CHEEK_IDX = [117, 118, 50, 205, 187, 147, 213, 192]
RIGHT_CHEEK_IDX = [346, 347, 280, 425, 411, 376, 433, 416]

HEART_WINDOW_SEC = 12.0
MIN_HEART_WINDOW_SEC = 6.0
HEART_BAND_HZ = (0.8, 3.2)
WELCH_SEG_SEC_HEART = 5.0
WELCH_OVERLAP_HEART = 0.5
ALPHA_WINDOW_SEC = 1.6
MAX_RPPG_BUFFER_SEC = 65.0  # estende para 65s pra acomodar janela HRV de 60s
EMA_ALPHA_HEART = 0.15
SNR_MIN_DB = 2.0

RESP_WINDOW_SEC = 30.0
MIN_RESP_WINDOW_SEC = 12.0
RESP_BAND_HZ = (0.1, 0.5)
WELCH_SEG_SEC_RESP = 20.0
WELCH_OVERLAP_RESP = 0.75
EMA_ALPHA_RESP = 0.10

# IBI / HRV
IBI_WINDOW_SEC = 60.0
PEAK_MIN_DISTANCE_SEC = 0.4   # equivalente a 150bpm
PEAK_PROMINENCE_REL = 0.5     # fração do desvio padrão do sinal


# ============== Construção da ROI e extração RGB ==============
def build_roi_mask(rgb_frame, face_landmarks):
    h, w = rgb_frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    forehead = landmark_points(face_landmarks, FOREHEAD_IDX, w, h)
    c_x = np.mean(forehead[:, 0])
    c_y = np.mean(forehead[:, 1])
    forehead = np.stack(
        [c_x + (forehead[:, 0] - c_x) * 1.12,
         c_y + (forehead[:, 1] - c_y) * 1.18],
        axis=1,
    )
    forehead[:, 0] = np.clip(forehead[:, 0], 0, w - 1)
    forehead[:, 1] = np.clip(forehead[:, 1], 0, h - 1)
    forehead = forehead.astype(np.int32)

    left_cheek = landmark_points(face_landmarks, LEFT_CHEEK_IDX, w, h)
    right_cheek = landmark_points(face_landmarks, RIGHT_CHEEK_IDX, w, h)

    cv.fillConvexPoly(mask, cv.convexHull(forehead), 255)
    cv.fillConvexPoly(mask, cv.convexHull(left_cheek), 255)
    cv.fillConvexPoly(mask, cv.convexHull(right_cheek), 255)
    return mask


def extract_combined_roi(rgb_frame, face_landmarks):
    mask = build_roi_mask(rgb_frame, face_landmarks)
    px = rgb_frame[mask == 255]
    if px.size == 0:
        return None
    return np.array([np.mean(px[:, 0]), np.mean(px[:, 1]), np.mean(px[:, 2])])


# ============== DSP ==============
def chrom_signal(r, g, b, fs):
    n = len(r)
    alpha_win = max(4, round(ALPHA_WINDOW_SEC * fs))
    signal = np.zeros(n)
    for start in range(0, n, alpha_win):
        end = min(start + alpha_win, n)
        if end - start < 2:
            break
        mu_r = max(np.mean(r[start:end]), 1e-6)
        mu_g = max(np.mean(g[start:end]), 1e-6)
        mu_b = max(np.mean(b[start:end]), 1e-6)
        rn = r[start:end] / mu_r - 1
        gn = g[start:end] / mu_g - 1
        bn = b[start:end] / mu_b - 1
        xs = 3 * rn - 2 * gn
        ys = 1.5 * rn + gn - 1.5 * bn
        alpha = np.std(xs) / (np.std(ys) + 1e-8)
        signal[start:end] = xs - alpha * ys
    s = np.std(signal)
    if s < 1e-8:
        return np.zeros(n)
    return (signal - np.mean(signal)) / s


def detrend_linear(signal):
    n = len(signal)
    if n < 3:
        return signal
    x = np.arange(n)
    coeffs = np.polyfit(x, signal, 1)
    return signal - np.polyval(coeffs, x)


def bandpass_fft(signal, fs, low_hz, high_hz):
    n = len(signal)
    if n < 4:
        return signal
    spec = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    spec[~mask] = 0
    return np.fft.irfft(spec, n=n)


def welch_psd(signal, fs, seg_sec, overlap):
    n = len(signal)
    if n < 16:
        return None
    nperseg = max(32, min(round(seg_sec * fs), n))
    if n < nperseg:
        return None
    step = max(1, round(nperseg * (1 - overlap)))
    win = 0.5 * (1 - np.cos(2 * np.pi * np.arange(nperseg) / (nperseg - 1)))
    win_pow = np.sum(win ** 2) + 1e-12
    freqs = np.fft.rfftfreq(nperseg, d=1.0 / fs)
    acc = np.zeros(len(freqs))
    count = 0
    for start in range(0, n - nperseg + 1, step):
        seg = signal[start:start + nperseg].copy()
        seg -= np.mean(seg)
        seg *= win
        spec = np.fft.rfft(seg)
        acc += (np.abs(spec) ** 2) / win_pow
        count += 1
    if count == 0:
        return None
    return freqs, acc / count


def estimate_rate(welch_result, band_hz):
    if welch_result is None:
        return None
    freqs, psd = welch_result
    low, high = band_hz
    mask = (freqs >= low) & (freqs <= high)
    if not np.any(mask):
        return None
    band_psd = psd[mask]
    band_freqs = freqs[mask]
    peak_idx = np.argmax(band_psd)
    peak_hz = band_freqs[peak_idx]
    peak_pow = band_psd[peak_idx]
    total_pow = np.sum(band_psd)
    noise_pow = (total_pow - peak_pow) / max(len(band_psd) - 1, 1)
    snr_db = 10 * np.log10((peak_pow + 1e-12) / (noise_pow + 1e-12))
    return {"bpm": peak_hz * 60, "snr_db": snr_db, "peak_hz": peak_hz}


# ============== Tracker ==============
class RPPGTracker:
    def __init__(self):
        self.r_buf = []
        self.g_buf = []
        self.b_buf = []
        self.ts_buf = []  # ms
        self.smooth_bpm = None
        self.last_heart = None
        self.smooth_rpm = None
        self.last_resp = None
        self.last_compute_ms = 0
        self.compute_interval_ms = 1000
        self._last_signal = None  # último sinal CHROM filtrado (pra extração de IBI)
        self._last_signal_fs = None

    def feed_frame(self, rgb_frame, face_landmarks, timestamp_ms):
        roi = extract_combined_roi(rgb_frame, face_landmarks)
        if roi is None:
            return
        self.r_buf.append(roi[0])
        self.g_buf.append(roi[1])
        self.b_buf.append(roi[2])
        self.ts_buf.append(timestamp_ms)
        # Trim
        while (len(self.ts_buf) > 1 and
               (self.ts_buf[-1] - self.ts_buf[0]) > MAX_RPPG_BUFFER_SEC * 1000):
            self.r_buf.pop(0); self.g_buf.pop(0)
            self.b_buf.pop(0); self.ts_buf.pop(0)

    def maybe_recompute(self, timestamp_ms):
        """Recompute BPM/RPM se passou o intervalo. Retorna True se recomputou."""
        if timestamp_ms - self.last_compute_ms <= self.compute_interval_ms:
            return False
        self.last_compute_ms = timestamp_ms

        hr_signal_fs = self._compute_heart()
        if hr_signal_fs is not None:
            self._last_signal, self._last_signal_fs = hr_signal_fs

        self._compute_resp()
        return True

    def _window_buffers(self, window_sec, min_window_sec):
        if len(self.ts_buf) < 2:
            return None
        dur_sec = (self.ts_buf[-1] - self.ts_buf[0]) / 1000.0
        if dur_sec < min_window_sec:
            return None
        cutoff_ms = self.ts_buf[-1] - window_sec * 1000
        start_idx = 0
        while start_idx < len(self.ts_buf) - 1 and self.ts_buf[start_idx] < cutoff_ms:
            start_idx += 1
        r = np.array(self.r_buf[start_idx:], dtype=np.float64)
        g = np.array(self.g_buf[start_idx:], dtype=np.float64)
        b = np.array(self.b_buf[start_idx:], dtype=np.float64)
        ts = self.ts_buf[start_idx:]
        if len(ts) < 2:
            return None
        diffs = np.diff(ts) / 1000.0
        avg_dt = np.mean(diffs)
        if avg_dt <= 0:
            return None
        fs = 1.0 / avg_dt
        if fs < 1:
            return None
        return r, g, b, fs, ts

    def _compute_heart(self):
        win = self._window_buffers(HEART_WINDOW_SEC, MIN_HEART_WINDOW_SEC)
        if win is None:
            return None
        r, g, b, fs, _ts = win
        sig = chrom_signal(r, g, b, fs)
        sig = detrend_linear(sig)
        sig = bandpass_fft(sig, fs, HEART_BAND_HZ[0], HEART_BAND_HZ[1])
        s = np.std(sig)
        if s > 1e-8:
            sig = (sig - np.mean(sig)) / s
        psd_result = welch_psd(sig, fs, WELCH_SEG_SEC_HEART, WELCH_OVERLAP_HEART)
        result = estimate_rate(psd_result, HEART_BAND_HZ)
        if result is not None:
            self.last_heart = result
            if result["snr_db"] >= SNR_MIN_DB:
                if self.smooth_bpm is None:
                    self.smooth_bpm = result["bpm"]
                else:
                    self.smooth_bpm = (
                        EMA_ALPHA_HEART * result["bpm"]
                        + (1 - EMA_ALPHA_HEART) * self.smooth_bpm
                    )
        return sig, fs

    def _compute_resp(self):
        win = self._window_buffers(RESP_WINDOW_SEC, MIN_RESP_WINDOW_SEC)
        if win is None:
            return
        _r, g, _b, fs, _ts = win
        sig = g.copy()
        s = np.std(sig)
        if s < 1e-8:
            return
        sig = (sig - np.mean(sig)) / s
        sig = detrend_linear(sig)
        sig = bandpass_fft(sig, fs, RESP_BAND_HZ[0], RESP_BAND_HZ[1])
        s = np.std(sig)
        if s > 1e-8:
            sig = (sig - np.mean(sig)) / s
        psd_result = welch_psd(sig, fs, WELCH_SEG_SEC_RESP, WELCH_OVERLAP_RESP)
        result = estimate_rate(psd_result, RESP_BAND_HZ)
        if result is not None:
            self.last_resp = result
            if result["snr_db"] >= -5.0:
                rpm = result["bpm"]
                if self.smooth_rpm is None:
                    self.smooth_rpm = rpm
                else:
                    self.smooth_rpm = (
                        EMA_ALPHA_RESP * rpm
                        + (1 - EMA_ALPHA_RESP) * self.smooth_rpm
                    )

    def get_ibi_buffer(self, window_sec=IBI_WINDOW_SEC):
        """
        Detecta picos no sinal CHROM filtrado mais recente e retorna
        array de IBI em ms. Retorna None se não houver sinal disponível
        ou número de picos insuficiente.

        STATUS: experimental — precisão sub-frame não implementada (TODO:
        parabolic interpolation Gasior 2004). Uso atual só para protótipo
        de HRV até validação contra ground truth.
        """
        # Recomputar com janela específica de IBI (60s) — pode diferir da janela cardíaca (12s)
        win = self._window_buffers(window_sec, min_window_sec=20.0)
        if win is None:
            return None
        r, g, b, fs, ts = win
        sig = chrom_signal(r, g, b, fs)
        sig = detrend_linear(sig)
        sig = bandpass_fft(sig, fs, HEART_BAND_HZ[0], HEART_BAND_HZ[1])
        s = np.std(sig)
        if s < 1e-8:
            return None
        sig = (sig - np.mean(sig)) / s

        min_dist_samples = max(1, int(PEAK_MIN_DISTANCE_SEC * fs))
        peaks, _props = find_peaks(
            sig,
            distance=min_dist_samples,
            prominence=PEAK_PROMINENCE_REL,
        )
        if len(peaks) < 3:
            return None
        peak_times_ms = np.array(ts)[peaks]
        ibi_ms = np.diff(peak_times_ms.astype(np.float64))
        return ibi_ms

    def buffer_seconds(self):
        if len(self.ts_buf) < 2:
            return 0.0
        return (self.ts_buf[-1] - self.ts_buf[0]) / 1000.0

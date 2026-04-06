import cv2 as cv
import mediapipe as mp
import numpy as np
import time
import csv
import os
from datetime import datetime
from collections import deque

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles


# EAR (Eye Aspect Ratio) — 6 pontos por olho
RIGHT_EYE_EAR = [33, 159, 158, 133, 153, 145]
LEFT_EYE_EAR = [362, 386, 385, 263, 380, 374]

# Calibração — duas fases
CAL_OPEN_SEC = 5.0
CAL_CLOSED_SEC = 5.0
CAL_CLOSED_SKIP = 1.0
CAL_CLOSED_MEASURE = 2.0

# Monitoramento
MOVING_AVG_LEN = 7

# Piscadas
BLINK_MIN_FRAMES = 1
BLINK_MAX_FRAMES = 8
BLINK_INTERVAL_WINDOW = 5

# Face perdida
NO_FACE_RESET_FRAMES = 30

# Fadiga — PERCLOS
FATIGUE_WINDOW_SEC = 180.0
PERCLOS_80_RATIO = 0.80
FATIGUE_ALERT_PERCLOS = 15.0

# rPPG — CHROM (de Haan & Jeanne 2013)
FOREHEAD_IDX = [54, 10, 67, 103, 109, 338, 297, 332, 284]
LEFT_CHEEK_IDX = [117, 118, 50, 205, 187, 147, 213, 192]
RIGHT_CHEEK_IDX = [346, 347, 280, 425, 411, 376, 433, 416]

HEART_WINDOW_SEC = 12.0
MIN_HEART_WINDOW_SEC = 6.0
HEART_BAND_HZ = (0.8, 3.2)
WELCH_SEG_SEC_HEART = 5.0
WELCH_OVERLAP_HEART = 0.5
ALPHA_WINDOW_SEC = 1.6
MAX_RPPG_BUFFER_SEC = 35.0
EMA_ALPHA_HEART = 0.15
SNR_MIN_DB = 2.0

RESP_WINDOW_SEC = 30.0
MIN_RESP_WINDOW_SEC = 12.0
RESP_BAND_HZ = (0.1, 0.5)
WELCH_SEG_SEC_RESP = 20.0
WELCH_OVERLAP_RESP = 0.75
EMA_ALPHA_RESP = 0.10


# =====================================================
# Funções auxiliares
# =====================================================
def get_landmark_px(face_landmarks, idx, w, h):
    lm = face_landmarks[idx]
    return np.array([lm.x * w, lm.y * h])


def eye_aspect_ratio(face_landmarks, eye_indices, w, h):
    p1 = get_landmark_px(face_landmarks, eye_indices[0], w, h)
    p2 = get_landmark_px(face_landmarks, eye_indices[1], w, h)
    p3 = get_landmark_px(face_landmarks, eye_indices[2], w, h)
    p4 = get_landmark_px(face_landmarks, eye_indices[3], w, h)
    p5 = get_landmark_px(face_landmarks, eye_indices[4], w, h)
    p6 = get_landmark_px(face_landmarks, eye_indices[5], w, h)

    vertical_1 = np.linalg.norm(p2 - p6)
    vertical_2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)

    if horizontal < 1e-6:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def moving_average(buf):
    if not buf:
        return 0.0
    return sum(buf) / len(buf)


# =====================================================
# rPPG — CHROM + DSP
# =====================================================
def landmark_points(face_landmarks, idx_list, w, h):
    pts = []
    for idx in idx_list:
        lm = face_landmarks[idx]
        x = int(np.clip(lm.x * w, 0, w - 1))
        y = int(np.clip(lm.y * h, 0, h - 1))
        pts.append([x, y])
    return np.array(pts, dtype=np.int32)


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


def chrom_signal(r, g, b, fs):
    n = len(r)
    alpha_win = max(4, round(ALPHA_WINDOW_SEC * fs))
    signal = np.zeros(n)

    for start in range(0, n, alpha_win):
        end = min(start + alpha_win, n)
        length = end - start
        if length < 2:
            break

        mu_r = np.mean(r[start:end])
        mu_g = np.mean(g[start:end])
        mu_b = np.mean(b[start:end])
        mu_r = max(mu_r, 1e-6)
        mu_g = max(mu_g, 1e-6)
        mu_b = max(mu_b, 1e-6)

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


def estimate_heart_rate(welch_result, band_hz):
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


def compute_heart_rate_from_buffers(r_buf, g_buf, b_buf, ts_buf):
    if len(ts_buf) < 2:
        return None
    dur_sec = (ts_buf[-1] - ts_buf[0]) / 1000.0
    if dur_sec < MIN_HEART_WINDOW_SEC:
        return None

    # Extract window
    cutoff_ms = ts_buf[-1] - HEART_WINDOW_SEC * 1000
    start_idx = 0
    while start_idx < len(ts_buf) - 1 and ts_buf[start_idx] < cutoff_ms:
        start_idx += 1

    r = np.array(r_buf[start_idx:], dtype=np.float64)
    g = np.array(g_buf[start_idx:], dtype=np.float64)
    b = np.array(b_buf[start_idx:], dtype=np.float64)
    ts = ts_buf[start_idx:]
    if len(ts) < 2:
        return None

    diffs = np.diff(ts) / 1000.0
    avg_dt = np.mean(diffs)
    fs = 1.0 / avg_dt if avg_dt > 0 else None
    if fs is None or fs < 1:
        return None

    sig = chrom_signal(r, g, b, fs)
    sig = detrend_linear(sig)
    sig = bandpass_fft(sig, fs, HEART_BAND_HZ[0], HEART_BAND_HZ[1])
    s = np.std(sig)
    if s > 1e-8:
        sig = (sig - np.mean(sig)) / s

    psd_result = welch_psd(sig, fs, WELCH_SEG_SEC_HEART, WELCH_OVERLAP_HEART)
    return estimate_heart_rate(psd_result, HEART_BAND_HZ)


def compute_resp_rate_from_buffers(r_buf, g_buf, b_buf, ts_buf):
    if len(ts_buf) < 2:
        return None
    dur_sec = (ts_buf[-1] - ts_buf[0]) / 1000.0
    if dur_sec < MIN_RESP_WINDOW_SEC:
        return None

    cutoff_ms = ts_buf[-1] - RESP_WINDOW_SEC * 1000
    start_idx = 0
    while start_idx < len(ts_buf) - 1 and ts_buf[start_idx] < cutoff_ms:
        start_idx += 1

    r = np.array(r_buf[start_idx:], dtype=np.float64)
    g = np.array(g_buf[start_idx:], dtype=np.float64)
    b = np.array(b_buf[start_idx:], dtype=np.float64)
    ts = ts_buf[start_idx:]
    if len(ts) < 2:
        return None

    diffs = np.diff(ts) / 1000.0
    avg_dt = np.mean(diffs)
    fs = 1.0 / avg_dt if avg_dt > 0 else None
    if fs is None or fs < 1:
        return None

    # Respiração é mais robusta no canal verde bruto
    sig = g.copy()
    s = np.std(sig)
    if s < 1e-8:
        return None
    sig = (sig - np.mean(sig)) / s
    sig = detrend_linear(sig)
    sig = bandpass_fft(sig, fs, RESP_BAND_HZ[0], RESP_BAND_HZ[1])
    s = np.std(sig)
    if s > 1e-8:
        sig = (sig - np.mean(sig)) / s

    psd_result = welch_psd(sig, fs, WELCH_SEG_SEC_RESP, WELCH_OVERLAP_RESP)
    return estimate_heart_rate(psd_result, RESP_BAND_HZ)


def draw_landmarks_on_image(rgb_image, detection_result):
    annotated_image = np.copy(rgb_image)
    if not detection_result.face_landmarks:
        return annotated_image

    for face_landmarks in detection_result.face_landmarks:
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style()
        )
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style()
        )
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style()
        )
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style()
        )

    return annotated_image


# =====================================================
# Inicialização
# =====================================================
base_options = python.BaseOptions(model_asset_path="face_landmarker.task")
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False
)
detector = vision.FaceLandmarker.create_from_options(options)

cap = cv.VideoCapture(0)
timestamp_ms = 0
frame_duration_ms = 33

# Calibração (feita uma única vez)
cal_phase = "open"  # "open" → "closed" → "done"
cal_start_ms = None
cal_open_samples = []
cal_closed_samples_ts = []
# Dados do rosto calibrado (imutáveis após calibração)
calibrated = False
ear_open = None
ear_closed = None
perclos_threshold = None

# Estado de monitoramento (reseta ao perder/trocar rosto)
mon = {
    "ear_buf_r": deque(maxlen=MOVING_AVG_LEN),
    "ear_buf_l": deque(maxlen=MOVING_AVG_LEN),
    "blink_closed_frames": 0,
    "in_blink": False,
    "blink_last_time": None,
    "blink_intervals": deque(maxlen=BLINK_INTERVAL_WINDOW),
    "fatigue_buf": deque(),
}

no_face_frames = 0

# rPPG — buffers de cor
rppg_r_buf = []
rppg_g_buf = []
rppg_b_buf = []
rppg_ts_buf = []
rppg_smooth_bpm = None
rppg_last_heart = None
rppg_smooth_rpm = None
rppg_last_resp = None
rppg_last_compute_ms = 0
RPPG_COMPUTE_INTERVAL_MS = 1000

# Fatigue log — 1 sample per second
fatigue_log = []
log_last_time = 0.0
log_start_datetime = None

# =====================================================
# Loop principal
# =====================================================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect_for_video(mp_image, timestamp_ms)

    annotated_frame = draw_landmarks_on_image(rgb_frame, detection_result)
    h, w = rgb_frame.shape[:2]
    now = time.monotonic()

    if not detection_result.face_landmarks:
        no_face_frames += 1

        cv.putText(annotated_frame, "Sem face detectada", (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    else:
        no_face_frames = 0
        lm = detection_result.face_landmarks[0]

        ear_right = eye_aspect_ratio(lm, RIGHT_EYE_EAR, w, h)
        ear_left = eye_aspect_ratio(lm, LEFT_EYE_EAR, w, h)
        ear_avg = (ear_right + ear_left) / 2.0

        # Desenha os 6 pontos de cada olho
        for idx in RIGHT_EYE_EAR:
            pt = get_landmark_px(lm, idx, w, h).astype(int)
            cv.circle(annotated_frame, tuple(pt), 2, (0, 255, 0), -1)
        for idx in LEFT_EYE_EAR:
            pt = get_landmark_px(lm, idx, w, h).astype(int)
            cv.circle(annotated_frame, tuple(pt), 2, (0, 0, 255), -1)

        # --- rPPG: coleta RGB das ROIs a cada frame ---
        roi_rgb = extract_combined_roi(rgb_frame, lm)
        if roi_rgb is not None:
            rppg_r_buf.append(roi_rgb[0])
            rppg_g_buf.append(roi_rgb[1])
            rppg_b_buf.append(roi_rgb[2])
            rppg_ts_buf.append(timestamp_ms)
            # Trim ao tamanho máximo do buffer
            while (len(rppg_ts_buf) > 1 and
                   (rppg_ts_buf[-1] - rppg_ts_buf[0]) > MAX_RPPG_BUFFER_SEC * 1000):
                rppg_r_buf.pop(0)
                rppg_g_buf.pop(0)
                rppg_b_buf.pop(0)
                rppg_ts_buf.pop(0)

        # --- rPPG: calcula BPM periodicamente ---
        if timestamp_ms - rppg_last_compute_ms > RPPG_COMPUTE_INTERVAL_MS:
            rppg_last_compute_ms = timestamp_ms
            hr = compute_heart_rate_from_buffers(
                rppg_r_buf, rppg_g_buf, rppg_b_buf, rppg_ts_buf
            )
            if hr is not None:
                rppg_last_heart = hr
                if hr["snr_db"] >= SNR_MIN_DB:
                    if rppg_smooth_bpm is None:
                        rppg_smooth_bpm = hr["bpm"]
                    else:
                        rppg_smooth_bpm = (EMA_ALPHA_HEART * hr["bpm"] +
                                           (1 - EMA_ALPHA_HEART) * rppg_smooth_bpm)

            resp = compute_resp_rate_from_buffers(
                rppg_r_buf, rppg_g_buf, rppg_b_buf, rppg_ts_buf
            )
            if resp is not None:
                rppg_last_resp = resp
                if resp["snr_db"] >= -5.0:  # resp SNR tende a ser mais baixo que heart
                    rpm = resp["bpm"]  # estimate_heart_rate retorna rate=peakHz*60
                    if rppg_smooth_rpm is None:
                        rppg_smooth_rpm = rpm
                    else:
                        rppg_smooth_rpm = (EMA_ALPHA_RESP * rpm +
                                           (1 - EMA_ALPHA_RESP) * rppg_smooth_rpm)

        # =============================================
        # Calibração (apenas 1x no início do programa)
        # =============================================
        if not calibrated:
            if cal_start_ms is None:
                cal_start_ms = timestamp_ms

            cal_elapsed = (timestamp_ms - cal_start_ms) / 1000.0
            bar_w = 300
            bar_x = w // 2 - bar_w // 2

            # --- Fase 1: Olhos abertos (5s) ---
            if cal_phase == "open":
                cal_open_samples.append(ear_avg)

                remaining = max(0, CAL_OPEN_SEC - cal_elapsed)
                progress = min(1.0, cal_elapsed / CAL_OPEN_SEC)

                cv.rectangle(annotated_frame, (bar_x, 10), (bar_x + bar_w, 35), (80, 80, 80), -1)
                cv.rectangle(annotated_frame, (bar_x, 10), (bar_x + int(bar_w * progress), 35), (0, 255, 0), -1)
                cv.putText(annotated_frame, f"OLHOS ABERTOS - {remaining:.1f}s",
                           (bar_x - 20, 55), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv.putText(annotated_frame, "Olhe para a camera sem piscar",
                           (bar_x - 60, 80), cv.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                if cal_elapsed >= CAL_OPEN_SEC:
                    ear_open = float(np.median(cal_open_samples))
                    cal_phase = "closed"
                    cal_start_ms = timestamp_ms
                    print(f"Fase 1 concluida — EAR aberto: {ear_open:.3f}")

            # --- Fase 2: Olhos fechados (5s, mede 1s-3s) ---
            elif cal_phase == "closed":
                cal_elapsed = (timestamp_ms - cal_start_ms) / 1000.0
                cal_closed_samples_ts.append((timestamp_ms, ear_avg))

                remaining = max(0, CAL_CLOSED_SEC - cal_elapsed)
                progress = min(1.0, cal_elapsed / CAL_CLOSED_SEC)

                cv.rectangle(annotated_frame, (bar_x, 10), (bar_x + bar_w, 35), (80, 80, 80), -1)
                cv.rectangle(annotated_frame, (bar_x, 10), (bar_x + int(bar_w * progress), 35), (0, 0, 255), -1)
                cv.putText(annotated_frame, f"FECHE OS OLHOS - {remaining:.1f}s",
                           (bar_x - 20, 55), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv.putText(annotated_frame, "Mantenha os olhos fechados",
                           (bar_x - 40, 80), cv.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                if cal_elapsed >= CAL_CLOSED_SEC:
                    start_ms = cal_start_ms + int(CAL_CLOSED_SKIP * 1000)
                    end_ms = start_ms + int(CAL_CLOSED_MEASURE * 1000)
                    stable_samples = [ear for ts, ear in cal_closed_samples_ts
                                      if start_ms <= ts <= end_ms]
                    if stable_samples:
                        ear_closed = float(np.median(stable_samples))
                    else:
                        ear_closed = float(np.median([ear for _, ear in cal_closed_samples_ts]))

                    ear_range = ear_open - ear_closed
                    perclos_threshold = ear_closed + ear_range * (1.0 - PERCLOS_80_RATIO)

                    calibrated = True

                    print(f"Fase 2 concluida — EAR fechado: {ear_closed:.3f}")
                    print(f"PERCLOS P80 threshold: {perclos_threshold:.3f}")
                    print(f"Range: [{ear_closed:.3f} (fechado) ... {ear_open:.3f} (aberto)]")

        # =============================================
        # Monitoramento
        # =============================================
        else:
            mon["ear_buf_r"].append(ear_right)
            mon["ear_buf_l"].append(ear_left)

            smooth_r = moving_average(mon["ear_buf_r"])
            smooth_l = moving_average(mon["ear_buf_l"])

            ear_range = ear_open - ear_closed
            if ear_range > 1e-6:
                pct_r = max(0, min(100, (smooth_r - ear_closed) / ear_range * 100))
                pct_l = max(0, min(100, (smooth_l - ear_closed) / ear_range * 100))
            else:
                pct_r = pct_l = 100

            alert_r = pct_r < 80
            alert_l = pct_l < 80

            # --- Piscadas ---
            blink_th = ear_closed + ear_range * 0.45
            if ear_avg < blink_th:
                mon["blink_closed_frames"] += 1
            else:
                if mon["in_blink"]:
                    mon["in_blink"] = False
                if BLINK_MIN_FRAMES <= mon["blink_closed_frames"] <= BLINK_MAX_FRAMES:
                    if mon["blink_last_time"] is not None:
                        mon["blink_intervals"].append(now - mon["blink_last_time"])
                    mon["blink_last_time"] = now
                mon["blink_closed_frames"] = 0

            if mon["blink_closed_frames"] == BLINK_MIN_FRAMES:
                mon["in_blink"] = True

            if mon["blink_intervals"]:
                avg_interval = sum(mon["blink_intervals"]) / len(mon["blink_intervals"])
                blinks_per_min = 60.0 / avg_interval if avg_interval > 0 else 0
            else:
                blinks_per_min = 0

            # --- Fadiga (PERCLOS P80) ---
            eyes_closed_p80 = alert_r or alert_l
            mon["fatigue_buf"].append((now, eyes_closed_p80))

            cutoff = now - FATIGUE_WINDOW_SEC
            while mon["fatigue_buf"] and mon["fatigue_buf"][0][0] < cutoff:
                mon["fatigue_buf"].popleft()

            total = len(mon["fatigue_buf"])
            closed_count = sum(1 for _, c in mon["fatigue_buf"] if c)
            perclos = (closed_count / total * 100) if total > 0 else 0
            fatigue_alert = perclos > FATIGUE_ALERT_PERCLOS
            buf_duration = mon["fatigue_buf"][-1][0] - mon["fatigue_buf"][0][0] if total > 1 else 0

            # --- Fatigue log (1 sample/sec) ---
            if log_start_datetime is None:
                log_start_datetime = datetime.now()
            if now - log_last_time >= 1.0:
                log_last_time = now
                elapsed_s = (timestamp_ms - cal_start_ms) / 1000.0 if cal_start_ms else 0
                fatigue_log.append({
                    "timestamp_s": round(elapsed_s, 1),
                    "datetime": datetime.now().isoformat(timespec="seconds"),
                    "perclos_pct": round(perclos, 1),
                    "blinks_per_min": round(blinks_per_min, 1),
                    "ear_avg": round(ear_avg, 4),
                    "bpm": round(rppg_smooth_bpm, 1) if rppg_smooth_bpm is not None else "",
                    "rpm": round(rppg_smooth_rpm, 1) if rppg_smooth_rpm is not None else "",
                })

            # --- Console ---
            if fatigue_alert:
                print(f"[{timestamp_ms / 1000:.1f}s] FADIGA: PERCLOS={perclos:.1f}%")
            if alert_r or alert_l:
                sides = []
                if alert_r:
                    sides.append(f"Dir {pct_r:.0f}%")
                if alert_l:
                    sides.append(f"Esq {pct_l:.0f}%")
                print(f"[{timestamp_ms / 1000:.1f}s] ALERTA fechamento: {' | '.join(sides)}")

            # --- HUD ---
            color_r = (0, 0, 255) if alert_r else (0, 255, 0)
            color_l = (0, 0, 255) if alert_l else (0, 255, 0)

            cv.putText(annotated_frame, f"Dir: {pct_r:.0f}%", (10, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, color_r, 2)
            cv.putText(annotated_frame, f"Esq: {pct_l:.0f}%", (10, 60),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, color_l, 2)

            blink_color = (0, 200, 255)
            n_intervals = len(mon["blink_intervals"])
            cv.putText(annotated_frame, f"Piscadas: {blinks_per_min:.0f}/min (avg {n_intervals} intervalos)",
                       (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.6, blink_color, 2)

            fatigue_color = (0, 0, 255) if fatigue_alert else (0, 220, 120)
            fatigue_label = "FADIGA" if fatigue_alert else "OK"
            cv.putText(annotated_frame, f"PERCLOS: {perclos:.1f}% ({fatigue_label}) [{buf_duration:.0f}s]",
                       (10, 120), cv.FONT_HERSHEY_SIMPLEX, 0.6, fatigue_color, 2)

            # --- rPPG BPM + RPM ---
            rppg_buf_sec = ((rppg_ts_buf[-1] - rppg_ts_buf[0]) / 1000.0
                            if len(rppg_ts_buf) > 1 else 0)
            if rppg_smooth_bpm is not None and rppg_last_heart is not None:
                snr = rppg_last_heart["snr_db"]
                snr_color = (0, 255, 0) if snr > 3 else (0, 200, 255) if snr > 0 else (0, 0, 255)
                cv.putText(annotated_frame,
                           f"BPM: {rppg_smooth_bpm:.0f} (SNR: {snr:.1f} dB)",
                           (10, 150), cv.FONT_HERSHEY_SIMPLEX, 0.6, snr_color, 2)
            else:
                label = (f"rPPG: coletando... {rppg_buf_sec:.0f}s"
                         if rppg_buf_sec < MIN_HEART_WINDOW_SEC
                         else "rPPG: calculando...")
                cv.putText(annotated_frame, label,
                           (10, 150), cv.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2)

            if rppg_smooth_rpm is not None and rppg_last_resp is not None:
                snr_r = rppg_last_resp["snr_db"]
                snr_r_color = (0, 255, 0) if snr_r > 3 else (0, 200, 255) if snr_r > 0 else (0, 0, 255)
                cv.putText(annotated_frame,
                           f"RPM: {rppg_smooth_rpm:.0f} (SNR: {snr_r:.1f} dB)",
                           (10, 180), cv.FONT_HERSHEY_SIMPLEX, 0.6, snr_r_color, 2)
            else:
                if rppg_buf_sec < MIN_RESP_WINDOW_SEC:
                    resp_label = f"Resp: coletando... {rppg_buf_sec:.0f}/{MIN_RESP_WINDOW_SEC:.0f}s"
                else:
                    resp_label = "Resp: calculando..."
                cv.putText(annotated_frame, resp_label,
                           (10, 180), cv.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 2)

            cv.putText(annotated_frame,
                       f"EAR aberto: {ear_open:.3f} | fechado: {ear_closed:.3f} | P80: {perclos_threshold:.3f}",
                       (10, h - 15), cv.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

            if alert_r or alert_l or fatigue_alert:
                cv.rectangle(annotated_frame, (0, 0), (w, 5), (0, 0, 255), -1)

    cv.imshow("Eye Tracker", cv.cvtColor(annotated_frame, cv.COLOR_RGB2BGR))
    timestamp_ms += frame_duration_ms

    if cv.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv.destroyAllWindows()

# =====================================================
# Export fatigue log to CSV
# =====================================================
if fatigue_log:
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)

    ts_label = datetime.now().strftime("%Y-%m-%d_%Hh%M")
    csv_path = os.path.join(log_dir, f"fatigue_{ts_label}.csv")

    fields = ["timestamp_s", "datetime", "perclos_pct", "blinks_per_min", "ear_avg", "bpm", "rpm"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(fatigue_log)

    print(f"\nLog salvo: {csv_path} ({len(fatigue_log)} amostras)")

    # =====================================================
    # Generate fatigue report graph
    # =====================================================
    import matplotlib.pyplot as plt

    ts_min = [r["timestamp_s"] / 60.0 for r in fatigue_log]
    perclos_vals = [r["perclos_pct"] for r in fatigue_log]
    blink_vals = [r["blinks_per_min"] for r in fatigue_log]
    ear_vals = [r["ear_avg"] for r in fatigue_log]
    bpm_vals = [r["bpm"] if r["bpm"] != "" else None for r in fatigue_log]
    rpm_vals = [r["rpm"] if r["rpm"] != "" else None for r in fatigue_log]

    duration_min = ts_min[-1] if ts_min else 0
    start_str = log_start_datetime.strftime("%Y-%m-%d %H:%M") if log_start_datetime else "?"

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f"Relatório de Fadiga — {start_str} ({duration_min:.0f} min)", fontsize=14)

    # --- Subplot 1: PERCLOS ---
    ax1.plot(ts_min, perclos_vals, color="#d32f2f", linewidth=1.2, label="PERCLOS %")
    ax1.axhline(y=FATIGUE_ALERT_PERCLOS, color="#ff8a80", linestyle="--", linewidth=0.8, label=f"Alerta ({FATIGUE_ALERT_PERCLOS}%)")
    ax1.fill_between(ts_min, perclos_vals, alpha=0.15, color="#d32f2f")
    ax1.set_ylabel("PERCLOS (%)")
    ax1.set_ylim(0, max(max(perclos_vals) * 1.2, FATIGUE_ALERT_PERCLOS * 1.5))
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # --- Subplot 2: Blinks/min + EAR ---
    ax2.plot(ts_min, blink_vals, color="#1565c0", linewidth=1.2, label="Piscadas/min")
    ax2.set_ylabel("Piscadas/min", color="#1565c0")
    ax2.tick_params(axis="y", labelcolor="#1565c0")
    ax2.grid(True, alpha=0.3)

    ax2b = ax2.twinx()
    ax2b.plot(ts_min, ear_vals, color="#ff8f00", linewidth=1.0, alpha=0.7, label="EAR médio")
    ax2b.set_ylabel("EAR", color="#ff8f00")
    ax2b.tick_params(axis="y", labelcolor="#ff8f00")

    lines2 = ax2.get_legend_handles_labels()
    lines2b = ax2b.get_legend_handles_labels()
    ax2.legend(lines2[0] + lines2b[0], lines2[1] + lines2b[1], loc="upper right", fontsize=8)

    # --- Subplot 3: BPM + RPM ---
    bpm_ts = [t for t, v in zip(ts_min, bpm_vals) if v is not None]
    bpm_clean = [v for v in bpm_vals if v is not None]
    rpm_ts = [t for t, v in zip(ts_min, rpm_vals) if v is not None]
    rpm_clean = [v for v in rpm_vals if v is not None]

    if bpm_clean:
        ax3.plot(bpm_ts, bpm_clean, color="#2e7d32", linewidth=1.2, label="BPM")
    ax3.set_ylabel("BPM", color="#2e7d32")
    ax3.tick_params(axis="y", labelcolor="#2e7d32")
    ax3.set_xlabel("Tempo (min)")
    ax3.grid(True, alpha=0.3)

    ax3b = ax3.twinx()
    if rpm_clean:
        ax3b.plot(rpm_ts, rpm_clean, color="#7b1fa2", linewidth=1.0, alpha=0.7, label="RPM")
    ax3b.set_ylabel("RPM", color="#7b1fa2")
    ax3b.tick_params(axis="y", labelcolor="#7b1fa2")

    lines3 = ax3.get_legend_handles_labels()
    lines3b = ax3b.get_legend_handles_labels()
    ax3.legend(lines3[0] + lines3b[0], lines3[1] + lines3b[1], loc="upper right", fontsize=8)

    plt.tight_layout()

    png_path = csv_path.replace(".csv", ".png")
    fig.savefig(png_path, dpi=150)
    print(f"Gráfico salvo: {png_path}")

    plt.show()
else:
    print("\nNenhum dado de fadiga coletado — log não gerado.")

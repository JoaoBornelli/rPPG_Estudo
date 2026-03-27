from collections import deque

import cv2 as cv
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =========================
# Config
# =========================
MODEL_PATH = "face_landmarker.task"
SOURCE = 0  # 0 para webcam
HEART_WINDOW_SEC = 20.0
MIN_HEART_WINDOW_SEC = 10.0
RESP_WINDOW_SEC = 30.0
MIN_RESP_WINDOW_SEC = 12.0
HEART_BAND_HZ = (0.8, 3.2)  # 48-192 bpm
RESP_BAND_HZ = (0.1, 0.5)  # 6-30 rpm
SHOW_EVERY_N_FRAMES = 1
WELCH_SEG_SEC_HEART = 4.0
WELCH_SEG_SEC_RESP = 20.0
WELCH_OVERLAP_HEART = 0.5
WELCH_OVERLAP_RESP = 0.75
HR_MIN_SNR_DB = 1.5
RR_MIN_SNR_DB = -5.0
SMOOTHING_LEN = 5


FOREHEAD_IDX = [54, 10, 67, 103, 109, 338, 297, 332, 284]
LEFT_CHEEK_IDX = [117, 118, 50, 205, 187, 147, 213, 192]
RIGHT_CHEEK_IDX = [346, 347, 280, 425, 411, 376, 433, 416]


def create_face_detector(model_path: str) -> vision.FaceLandmarker:
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return vision.FaceLandmarker.create_from_options(options)


def normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    s = np.std(x)
    if s < 1e-8:
        return np.zeros_like(x)
    return (x - np.mean(x)) / s


def landmark_points(face_landmarks, idx_list: list[int], width: int, height: int) -> np.ndarray:
    pts = []
    for idx in idx_list:
        lm = face_landmarks[idx]
        x = int(np.clip(lm.x * width, 0, width - 1))
        y = int(np.clip(lm.y * height, 0, height - 1))
        pts.append([x, y])
    return np.array(pts, dtype=np.int32)


def build_roi_mask(frame_rgb: np.ndarray, face_landmarks) -> np.ndarray:
    h, w = frame_rgb.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    forehead = landmark_points(face_landmarks, FOREHEAD_IDX, w, h)
    c_x = np.mean(forehead[:, 0])
    c_y = np.mean(forehead[:, 1])
    forehead = np.stack(
        [c_x + (forehead[:, 0] - c_x) * 1.12, c_y + (forehead[:, 1] - c_y) * 1.18],
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


def signal_green(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    return normalize(g)


def signal_pos(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    rgb = np.vstack([r.astype(np.float64), g.astype(np.float64), b.astype(np.float64)])
    means = np.mean(rgb, axis=1, keepdims=True)
    means[means == 0.0] = 1.0
    c = (rgb / means) - 1.0
    x = c[1] - c[2]
    y = c[1] + c[2] - 2.0 * c[0]
    alpha = np.std(x) / (np.std(y) + 1e-8)
    return normalize(x + alpha * y)


def signal_chrom(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    r_n = normalize(r)
    g_n = normalize(g)
    b_n = normalize(b)
    x = 3.0 * r_n - 2.0 * g_n
    y = 1.5 * r_n + g_n - 1.5 * b_n
    alpha = np.std(x) / (np.std(y) + 1e-8)
    return normalize(x - alpha * y)


def sampling_rate_from_timestamps(timestamps_ms: np.ndarray) -> float | None:
    if len(timestamps_ms) < 2:
        return None
    dt = np.diff(timestamps_ms.astype(np.float64)) / 1000.0
    dt = dt[dt > 0]
    if dt.size == 0:
        return None
    return float(1.0 / np.mean(dt))


def detrend_linear(signal: np.ndarray) -> np.ndarray:
    n = len(signal)
    if n < 3:
        return signal
    x = np.arange(n, dtype=np.float64)
    p = np.polyfit(x, signal.astype(np.float64), deg=1)
    trend = p[0] * x + p[1]
    return signal - trend


def bandpass_fft(signal: np.ndarray, fs: float, low_hz: float, high_hz: float) -> np.ndarray:
    if len(signal) < 4:
        return signal
    spec = np.fft.rfft(signal.astype(np.float64))
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / fs)
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    spec[~mask] = 0.0
    return np.fft.irfft(spec, n=len(signal))


def welch_psd(signal: np.ndarray, fs: float, seg_sec: float, overlap: float) -> tuple[np.ndarray | None, np.ndarray | None]:
    n = len(signal)
    if n < 16:
        return None, None

    nperseg = int(round(seg_sec * fs))
    nperseg = max(32, min(nperseg, n))
    step = int(round(nperseg * (1.0 - overlap)))
    step = max(1, step)
    if n < nperseg:
        return None, None

    window = np.hanning(nperseg)
    win_pow = np.sum(window**2) + 1e-12
    acc = None
    count = 0
    for start in range(0, n - nperseg + 1, step):
        seg = signal[start:start + nperseg]
        seg = seg - np.mean(seg)
        p = (np.abs(np.fft.rfft(seg * window)) ** 2) / win_pow
        if acc is None:
            acc = p
        else:
            acc += p
        count += 1

    if acc is None or count == 0:
        return None, None
    psd = acc / float(count)
    freqs = np.fft.rfftfreq(nperseg, d=1.0 / fs)
    return freqs, psd


def compute_psd_standard(
    signal: np.ndarray,
    timestamps_ms: np.ndarray,
    band_hz: tuple[float, float],
    welch_seg_sec: float,
    welch_overlap: float,
) -> tuple[float | None, np.ndarray | None, np.ndarray | None]:
    if len(signal) < 16 or len(timestamps_ms) < 2:
        return None, None, None

    fs = sampling_rate_from_timestamps(timestamps_ms)
    if fs is None:
        return None, None, None

    sig = normalize(signal.astype(np.float64))
    sig = detrend_linear(sig)
    sig = bandpass_fft(sig, fs, band_hz[0], band_hz[1])
    sig = normalize(sig)

    freqs, spec = welch_psd(sig, fs, welch_seg_sec, welch_overlap)
    return float(fs), freqs, spec


def estimate_rate(freqs: np.ndarray | None, spec: np.ndarray | None, band_hz: tuple[float, float]) -> tuple[float | None, float | None, float | None]:
    if freqs is None or spec is None:
        return None, None, None
    mask = (freqs >= band_hz[0]) & (freqs <= band_hz[1])
    if not np.any(mask):
        return None, None, None
    s = spec[mask]
    f = freqs[mask]
    i = int(np.argmax(s))
    peak_hz = float(f[i])
    rate = peak_hz * 60.0
    peak_power = float(s[i])
    noise_power = float((np.sum(s) - peak_power) / max(len(s) - 1, 1))
    snr_db = 10.0 * np.log10((peak_power + 1e-12) / (noise_power + 1e-12))
    return float(rate), float(snr_db), peak_hz


def stabilize_rate(
    new_rate: float | None,
    new_snr: float | None,
    prev_rate: float | None,
    hist: deque[float],
    min_snr_db: float,
) -> float | None:
    if new_rate is None or new_snr is None:
        return prev_rate
    if prev_rate is None:
        hist.append(float(new_rate))
        while len(hist) > SMOOTHING_LEN:
            hist.popleft()
        return float(np.median(np.array(hist, dtype=np.float64)))
    if new_snr < min_snr_db:
        return prev_rate

    hist.append(float(new_rate))
    while len(hist) > SMOOTHING_LEN:
        hist.popleft()
    return float(np.median(np.array(hist, dtype=np.float64)))


def extract_window(series: np.ndarray, ts: np.ndarray, window_sec: float) -> tuple[np.ndarray, np.ndarray]:
    if len(ts) == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.int64)
    t_min = ts[-1] - int(window_sec * 1000.0)
    m = ts >= t_min
    return series[m], ts[m]


def build_fft_canvas(
    method_name: str,
    heart_freqs: np.ndarray | None,
    heart_spec: np.ndarray | None,
    resp_freqs: np.ndarray | None,
    resp_spec: np.ndarray | None,
    heart_peak_hz: float | None,
    heart_bpm: float | None,
    heart_snr: float | None,
    resp_peak_hz: float | None,
    resp_rpm: float | None,
    resp_snr: float | None,
    width: int = 1200,
    height: int = 300,
) -> np.ndarray:
    c = np.full((height, width, 3), 16, dtype=np.uint8)
    cv.rectangle(c, (0, 0), (width - 1, height - 1), (80, 80, 80), 1)
    cv.putText(c, f"{method_name} Welch PSD", (14, 28), cv.FONT_HERSHEY_SIMPLEX, 0.8, (240, 240, 240), 2)

    if (heart_freqs is None or heart_spec is None or len(heart_freqs) < 2) and (
        resp_freqs is None or resp_spec is None or len(resp_freqs) < 2
    ):
        cv.putText(c, "Dados insuficientes", (14, 64), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 160, 255), 2)
        return c

    left, right = 20, width - 20
    top, bottom = 44, height - 40
    max_hz = max(HEART_BAND_HZ[1] * 1.2, 4.5)

    # Grade de referencia linear (0..1 normalizado)
    for p in [0.2, 0.4, 0.6, 0.8]:
        y_tick = int(bottom - p * (bottom - top))
        cv.line(c, (left, y_tick), (right, y_tick), (38, 38, 38), 1)
        cv.putText(c, f"{p:.1f}", (right - 44, max(top + 12, y_tick - 4)), cv.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1)

    rx0 = int(left + (RESP_BAND_HZ[0] / max_hz) * (right - left))
    rx1 = int(left + (RESP_BAND_HZ[1] / max_hz) * (right - left))
    hx0 = int(left + (HEART_BAND_HZ[0] / max_hz) * (right - left))
    hx1 = int(left + (HEART_BAND_HZ[1] / max_hz) * (right - left))
    cv.rectangle(c, (rx0, top), (rx1, bottom), (45, 35, 25), -1)
    cv.rectangle(c, (hx0, top), (hx1, bottom), (35, 55, 35), -1)

    if heart_freqs is not None and heart_spec is not None and len(heart_freqs) >= 2:
        keep_h = (heart_freqs >= 0.0) & (heart_freqs <= max_hz)
        f_h = heart_freqs[keep_h]
        s_h = heart_spec[keep_h]
        if len(f_h) >= 2:
            sh = s_h - np.min(s_h)
            den_h = np.max(sh) - np.min(sh)
            shn = np.zeros_like(sh) if den_h < 1e-12 else sh / den_h
            xh = (left + (f_h / max_hz) * (right - left)).astype(np.int32)
            yh = (bottom - shn * (bottom - top)).astype(np.int32)
            pts_h = np.stack([xh, yh], axis=1).reshape(-1, 1, 2)
            cv.polylines(c, [pts_h], False, (80, 220, 240), 2)

    if resp_freqs is not None and resp_spec is not None and len(resp_freqs) >= 2:
        keep_r = (resp_freqs >= 0.0) & (resp_freqs <= max_hz)
        f_r = resp_freqs[keep_r]
        s_r = resp_spec[keep_r]
        if len(f_r) >= 2:
            sr = s_r - np.min(s_r)
            den_r = np.max(sr) - np.min(sr)
            srn = np.zeros_like(sr) if den_r < 1e-12 else sr / den_r
            xr = (left + (f_r / max_hz) * (right - left)).astype(np.int32)
            yr = (bottom - srn * (bottom - top)).astype(np.int32)
            pts_r = np.stack([xr, yr], axis=1).reshape(-1, 1, 2)
            cv.polylines(c, [pts_r], False, (0, 180, 255), 1)

    if heart_peak_hz is not None:
        px = int(left + (heart_peak_hz / max_hz) * (right - left))
        cv.line(c, (px, top), (px, bottom), (0, 230, 90), 2)
        if heart_freqs is not None and heart_spec is not None and len(heart_freqs) >= 2:
            kh = int(np.argmin(np.abs(heart_freqs - heart_peak_hz)))
            heart_peak_pow = float(heart_spec[kh])
            cv.putText(c, f"Peak HR power: {heart_peak_pow:.3g}", (14, 50), cv.FONT_HERSHEY_SIMPLEX, 0.52, (130, 255, 130), 1)
    if resp_peak_hz is not None:
        px = int(left + (resp_peak_hz / max_hz) * (right - left))
        cv.line(c, (px, top), (px, bottom), (0, 180, 255), 2)

    heart_txt = f"HR: {heart_bpm:.1f} bpm | SNR {heart_snr:.1f} dB" if heart_bpm is not None and heart_snr is not None else "HR: --"
    resp_txt = f"RR: {resp_rpm:.1f} rpm | SNR {resp_snr:.1f} dB" if resp_rpm is not None and resp_snr is not None else "RR: --"
    cv.putText(c, heart_txt, (14, height - 14), cv.FONT_HERSHEY_SIMPLEX, 0.58, (130, 255, 130), 2)
    cv.putText(c, resp_txt, (420, height - 14), cv.FONT_HERSHEY_SIMPLEX, 0.58, (130, 210, 255), 2)
    cv.putText(c, "Escala PSD: potencia linear (normalizada)", (760, height - 14), cv.FONT_HERSHEY_SIMPLEX, 0.52, (210, 210, 210), 1)
    return c


def method_signals(name: str, r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    if name == "GREEN":
        return signal_green(r, g, b)
    if name == "POS":
        return signal_pos(r, g, b)
    return signal_chrom(r, g, b)


def main() -> None:
    cap = cv.VideoCapture(SOURCE)
    if not cap.isOpened():
        raise RuntimeError(f"Nao foi possivel abrir a fonte: {SOURCE}")

    fps = float(cap.get(cv.CAP_PROP_FPS))
    nominal_fps = fps if np.isfinite(fps) and 1.0 <= fps <= 240.0 else 30.0
    step_ms = max(1, int(round(1000.0 / nominal_fps)))
    print(f"[VideoSource] FPS detectado: {nominal_fps:.2f}")

    detector = create_face_detector(MODEL_PATH)
    frame_idx = -1
    t_ms = 0
    last_pos_ms = -1

    r_buf: deque[float] = deque()
    g_buf: deque[float] = deque()
    b_buf: deque[float] = deque()
    ts_buf: deque[int] = deque()

    methods = ["GREEN", "POS", "CHROM"]
    state = {
        m: {
            "hr_prev": None,
            "rr_prev": None,
            "hr_hist": deque(),
            "rr_hist": deque(),
        }
        for m in methods
    }
    last_canvas = np.zeros((920, 1200, 3), dtype=np.uint8)
    last_video_display = None
    no_face_count = 0
    current_bpm = None

    while cap.isOpened():
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_idx += 1
        frame_rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)

        pos_ms = int(cap.get(cv.CAP_PROP_POS_MSEC))
        if pos_ms > last_pos_ms and pos_ms > t_ms:
            t_ms = pos_ms
            last_pos_ms = pos_ms
        else:
            t_ms += step_ms

        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        res = detector.detect_for_video(mp_img, t_ms)
        video_display = frame_bgr.copy()

        if res.face_landmarks:
            no_face_count = 0
            lm = res.face_landmarks[0]
            h, w = frame_rgb.shape[:2]
            mask = build_roi_mask(frame_rgb, lm)
            px = frame_rgb[mask == 255]
            if px.size > 0:
                r_buf.append(float(np.mean(px[:, 0])))
                g_buf.append(float(np.mean(px[:, 1])))
                b_buf.append(float(np.mean(px[:, 2])))
                ts_buf.append(t_ms)

            # Draw ROI landmarks on video
            forehead = landmark_points(lm, FOREHEAD_IDX, w, h)
            left_cheek = landmark_points(lm, LEFT_CHEEK_IDX, w, h)
            right_cheek = landmark_points(lm, RIGHT_CHEEK_IDX, w, h)

            c_x = np.mean(forehead[:, 0])
            c_y = np.mean(forehead[:, 1])
            forehead_scaled = np.stack(
                [c_x + (forehead[:, 0] - c_x) * 1.12, c_y + (forehead[:, 1] - c_y) * 1.18],
                axis=1,
            )
            forehead_scaled[:, 0] = np.clip(forehead_scaled[:, 0], 0, w - 1)
            forehead_scaled[:, 1] = np.clip(forehead_scaled[:, 1], 0, h - 1)
            forehead_scaled = forehead_scaled.astype(np.int32)

            cv.polylines(video_display, [cv.convexHull(forehead_scaled)], True, (0, 255, 0), 2)
            cv.polylines(video_display, [cv.convexHull(left_cheek)], True, (255, 0, 0), 2)
            cv.polylines(video_display, [cv.convexHull(right_cheek)], True, (0, 0, 255), 2)

            # Draw individual landmark points
            for pt in np.vstack([forehead_scaled, left_cheek, right_cheek]):
                cv.circle(video_display, tuple(pt), 2, (255, 255, 0), -1)
        else:
            no_face_count += 1
            if no_face_count > int(nominal_fps):
                r_buf.clear()
                g_buf.clear()
                b_buf.clear()
                ts_buf.clear()

        max_window = max(HEART_WINDOW_SEC, RESP_WINDOW_SEC)
        while len(ts_buf) > 1 and (ts_buf[-1] - ts_buf[0]) > int(max_window * 1000):
            r_buf.popleft()
            g_buf.popleft()
            b_buf.popleft()
            ts_buf.popleft()

        if frame_idx % SHOW_EVERY_N_FRAMES == 0:
            plots = []
            if len(ts_buf) > 1:
                r_all = np.array(r_buf, dtype=np.float64)
                g_all = np.array(g_buf, dtype=np.float64)
                b_all = np.array(b_buf, dtype=np.float64)
                ts_all = np.array(ts_buf, dtype=np.int64)

                for name in methods:
                    # heart window
                    r_h, ts_h = extract_window(r_all, ts_all, HEART_WINDOW_SEC)
                    g_h, _ = extract_window(g_all, ts_all, HEART_WINDOW_SEC)
                    b_h, _ = extract_window(b_all, ts_all, HEART_WINDOW_SEC)
                    heart_bpm = heart_snr = heart_peak_hz = None
                    freqs_h = spec_h = None
                    if len(ts_h) > 1 and (ts_h[-1] - ts_h[0]) >= int(MIN_HEART_WINDOW_SEC * 1000):
                        sig_h = method_signals(name, r_h, g_h, b_h)
                        _, freqs_h, spec_h = compute_psd_standard(
                            sig_h,
                            ts_h,
                            HEART_BAND_HZ,
                            WELCH_SEG_SEC_HEART,
                            WELCH_OVERLAP_HEART,
                        )
                        heart_bpm, heart_snr, heart_peak_hz = estimate_rate(freqs_h, spec_h, HEART_BAND_HZ)
                        heart_bpm = stabilize_rate(
                            heart_bpm,
                            heart_snr,
                            state[name]["hr_prev"],
                            state[name]["hr_hist"],
                            HR_MIN_SNR_DB,
                        )
                        state[name]["hr_prev"] = heart_bpm

                        # Update current BPM from CHROM method
                        if name == "CHROM":
                            current_bpm = heart_bpm

                    # resp window
                    r_r, ts_r = extract_window(r_all, ts_all, RESP_WINDOW_SEC)
                    g_r, _ = extract_window(g_all, ts_all, RESP_WINDOW_SEC)
                    b_r, _ = extract_window(b_all, ts_all, RESP_WINDOW_SEC)
                    resp_rpm = resp_snr = resp_peak_hz = None
                    freqs_r = spec_r = None
                    if len(ts_r) > 1 and (ts_r[-1] - ts_r[0]) >= int(MIN_RESP_WINDOW_SEC * 1000):
                        # Respiracao e mais robusta no canal verde bruto.
                        sig_r = signal_green(r_r, g_r, b_r)
                        _, freqs_r, spec_r = compute_psd_standard(
                            sig_r,
                            ts_r,
                            RESP_BAND_HZ,
                            WELCH_SEG_SEC_RESP,
                            WELCH_OVERLAP_RESP,
                        )
                        resp_rpm, resp_snr, resp_peak_hz = estimate_rate(freqs_r, spec_r, RESP_BAND_HZ)
                        resp_rpm = stabilize_rate(
                            resp_rpm,
                            resp_snr,
                            state[name]["rr_prev"],
                            state[name]["rr_hist"],
                            RR_MIN_SNR_DB,
                        )
                        state[name]["rr_prev"] = resp_rpm

                    plots.append(
                        build_fft_canvas(
                            name,
                            freqs_h,
                            spec_h,
                            freqs_r,
                            spec_r,
                            heart_peak_hz,
                            heart_bpm,
                            heart_snr,
                            resp_peak_hz,
                            resp_rpm,
                            resp_snr,
                        )
                    )

            if len(plots) == 3:
                last_canvas = cv.vconcat(plots)
            else:
                last_canvas = np.full((920, 1200, 3), 16, dtype=np.uint8)
                cv.putText(last_canvas, "Aguardando dados suficientes...", (30, 80), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 180, 255), 2)

        # Add BPM text to video display
        if current_bpm is not None:
            bpm_text = f"BPM: {current_bpm:.1f}"
            cv.putText(video_display, bpm_text, (20, 60), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        else:
            cv.putText(video_display, "BPM: --", (20, 60), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 3)

        last_video_display = video_display

        cv.imshow("Webcam with Landmarks", last_video_display)
        cv.imshow("FFT Compare: GREEN vs POS vs CHROM", last_canvas)
        if cv.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()

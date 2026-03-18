from collections import deque

import cv2 as cv
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =========================
# Config de experimento
# =========================
MODEL_PATH = "face_landmarker.task"
SOURCE = 0  # 0 para webcam

# Metodos: GREEN | POS | CHROM
HEART_METHOD = "CHROM"
RESP_METHOD = "GREEN"

HEART_WINDOW_SEC = 12.0
MIN_HEART_WINDOW_SEC = 6.0
RESP_WINDOW_SEC = 30.0
MIN_RESP_WINDOW_SEC = 12.0

HEART_BAND_HZ = (0.8, 3.2)  # 48-192 bpm
RESP_BAND_HZ = (0.1, 0.5)  # 6-30 rpm

# Welch (padrao mais robusto para PSD)
WELCH_SEG_SEC_HEART = 5.0
WELCH_OVERLAP_HEART = 0.5
WELCH_SEG_SEC_RESP = 20.0
WELCH_OVERLAP_RESP = 0.75

TARGET_TILE_HEIGHT = 360
RENDER_EVERY_N_FRAMES = 1
SHOW_GRAPH = True
SHOW_LANDMARK_IDS = True
SHOW_ALL_LANDMARKS = True
SHOW_ALL_LANDMARK_IDS = False
SHOW_ALL_LANDMARK_IDS_EVERY = 20


# ROIs (testa + bochechas)
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
    # expansao leve da testa
    c_x = np.mean(forehead[:, 0])
    c_y = np.mean(forehead[:, 1])
    forehead = np.stack(
        [
            c_x + (forehead[:, 0] - c_x) * 1.12,
            c_y + (forehead[:, 1] - c_y) * 1.18,
        ],
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


def get_face_rect(face_landmarks, width: int, height: int) -> tuple[int, int, int, int]:
    xs = [lm.x for lm in face_landmarks]
    ys = [lm.y for lm in face_landmarks]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x = max(0, int(x_min * width))
    y = max(0, int(y_min * height))
    w = max(1, int((x_max - x_min) * width))
    h = max(1, int((y_max - y_min) * height))
    return x, y, w, h


def draw_roi_landmark_ids(frame_bgr: np.ndarray, face_landmarks) -> np.ndarray:
    out = frame_bgr.copy()
    h, w = out.shape[:2]

    if SHOW_ALL_LANDMARKS:
        for idx, lm in enumerate(face_landmarks):
            x = int(np.clip(lm.x * w, 0, w - 1))
            y = int(np.clip(lm.y * h, 0, h - 1))
            cv.circle(out, (x, y), 1, (120, 120, 120), -1)
            if SHOW_ALL_LANDMARK_IDS and idx % SHOW_ALL_LANDMARK_IDS_EVERY == 0:
                cv.putText(out, str(idx), (x + 2, y - 2), cv.FONT_HERSHEY_SIMPLEX, 0.24, (150, 150, 150), 1)

    def draw_idx(idx: int, color: tuple[int, int, int]) -> None:
        lm = face_landmarks[idx]
        x = int(np.clip(lm.x * w, 0, w - 1))
        y = int(np.clip(lm.y * h, 0, h - 1))
        cv.circle(out, (x, y), 3, color, -1)
        cv.putText(out, str(idx), (x + 4, y - 4), cv.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

    for i in FOREHEAD_IDX:
        draw_idx(i, (0, 255, 255))  # amarelo
    for i in LEFT_CHEEK_IDX:
        draw_idx(i, (0, 255, 0))  # verde
    for i in RIGHT_CHEEK_IDX:
        draw_idx(i, (255, 255, 0))  # ciano

    return out


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


def method_signal(name: str, r: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    if name.upper() == "GREEN":
        return signal_green(r, g, b)
    if name.upper() == "POS":
        return signal_pos(r, g, b)
    return signal_chrom(r, g, b)


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
    return signal - (p[0] * x + p[1])


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

    win = np.hanning(nperseg)
    win_pow = np.sum(win**2) + 1e-12
    acc = None
    count = 0
    for start in range(0, n - nperseg + 1, step):
        seg = signal[start:start + nperseg]
        seg = seg - np.mean(seg)
        p = (np.abs(np.fft.rfft(seg * win)) ** 2) / win_pow
        acc = p if acc is None else acc + p
        count += 1

    if acc is None or count == 0:
        return None, None
    return np.fft.rfftfreq(nperseg, d=1.0 / fs), acc / float(count)


def compute_psd_standard(
    signal: np.ndarray,
    timestamps_ms: np.ndarray,
    band_hz: tuple[float, float],
    welch_seg_sec: float,
    welch_overlap: float,
) -> tuple[float | None, np.ndarray | None, np.ndarray | None]:
    fs = sampling_rate_from_timestamps(timestamps_ms)
    if fs is None or len(signal) < 16:
        return None, None, None

    sig = normalize(signal.astype(np.float64))
    sig = detrend_linear(sig)
    sig = bandpass_fft(sig, fs, band_hz[0], band_hz[1])
    sig = normalize(sig)

    freqs, spec = welch_psd(sig, fs, welch_seg_sec, welch_overlap)
    return fs, freqs, spec


def estimate_rate(
    freqs: np.ndarray | None,
    spec: np.ndarray | None,
    band_hz: tuple[float, float],
) -> tuple[float | None, float | None, float | None]:
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


def label(frame: np.ndarray, text: str) -> np.ndarray:
    out = frame.copy()
    cv.rectangle(out, (8, 8), (420, 42), (0, 0, 0), -1)
    cv.putText(out, text, (16, 32), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return out


def apply_chrom_spatial(region_bgr: np.ndarray, roi_mask_local: np.ndarray, alpha: float) -> np.ndarray:
    """Aplica CHROM espacialmente e retorna heatmap BGR."""
    b = region_bgr[:, :, 0].astype(np.float32)
    g = region_bgr[:, :, 1].astype(np.float32)
    r = region_bgr[:, :, 2].astype(np.float32)
    x_ch = 3.0 * r - 2.0 * g
    y_ch = 1.5 * r + g - 1.5 * b
    chrom = x_ch - alpha * y_ch
    chrom[roi_mask_local == 0] = 0.0
    roi_vals = chrom[roi_mask_local == 255]
    if roi_vals.size > 0:
        lo, hi = float(np.percentile(roi_vals, 2)), float(np.percentile(roi_vals, 98))
        span = hi - lo if hi - lo > 1e-6 else 1.0
        chrom = np.clip((chrom - lo) / span, 0.0, 1.0)
    gray = (chrom * 255).astype(np.uint8)
    gray[roi_mask_local == 0] = 0
    return cv.applyColorMap(gray, cv.COLORMAP_INFERNO)


def build_roi_chrom_tile(frame_bgr: np.ndarray, face_landmarks, chrom_alpha: float = 1.0, target_h: int = 360) -> np.ndarray:
    h, w = frame_bgr.shape[:2]

    roi_defs = [
        ("Testa", FOREHEAD_IDX, (255, 255, 100)),
        ("B.Esq", LEFT_CHEEK_IDX, (100, 255, 100)),
        ("B.Dir", RIGHT_CHEEK_IDX, (255, 200, 100)),
    ]
    n_cols = len(roi_defs)
    col_w = target_h * 2 // 3
    tile_w = col_w * n_cols
    canvas = np.zeros((target_h, tile_w, 3), dtype=np.uint8)
    header_h = 48

    for col, (roi_name, idx_list, color) in enumerate(roi_defs):
        pts = landmark_points(face_landmarks, idx_list, w, h)
        bx, by, bw, bh = cv.boundingRect(pts)
        pad = 12
        x1, y1 = max(0, bx - pad), max(0, by - pad)
        x2, y2 = min(w, bx + bw + pad), min(h, by + bh + pad)

        region = frame_bgr[y1:y2, x1:x2].copy()
        local_mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
        shifted = pts - np.array([x1, y1])
        cv.fillConvexPoly(local_mask, cv.convexHull(shifted), 255)

        heatmap = apply_chrom_spatial(region, local_mask, chrom_alpha)

        rh, rw = heatmap.shape[:2]
        avail_h = target_h - header_h
        scale = min(col_w / max(rw, 1), avail_h / max(rh, 1))
        nw, nh = max(1, int(rw * scale)), max(1, int(rh * scale))
        resized = cv.resize(heatmap, (nw, nh), interpolation=cv.INTER_AREA)

        x_off = col * col_w + (col_w - nw) // 2
        y_off = header_h + (avail_h - nh) // 2
        canvas[y_off:y_off + nh, x_off:x_off + nw] = resized

        cv.putText(canvas, roi_name, (col * col_w + 6, header_h - 6), cv.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
        if col > 0:
            cv.line(canvas, (col * col_w, header_h), (col * col_w, target_h - 1), (60, 60, 60), 1)

    cv.rectangle(canvas, (0, 0), (tile_w - 1, target_h - 1), (80, 80, 80), 1)
    cv.rectangle(canvas, (0, 0), (tile_w - 1, header_h - 14), (0, 0, 0), -1)
    cv.putText(canvas, f"ROIs - CHROM (alpha={chrom_alpha:.2f})", (8, 22), cv.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    return canvas


def resize_keep_aspect(frame: np.ndarray, target_h: int) -> np.ndarray:
    h, w = frame.shape[:2]
    target_w = max(80, int(round(target_h * (w / float(max(h, 1))))))
    return cv.resize(frame, (target_w, target_h), interpolation=cv.INTER_AREA)


def build_plot(signal: np.ndarray | None, fs: float | None, bpm: float | None, width: int = 900, height: int = 260) -> np.ndarray:
    canvas = np.full((height, width, 3), 18, dtype=np.uint8)
    cv.rectangle(canvas, (0, 0), (width - 1, height - 1), (80, 80, 80), 1)
    cv.putText(canvas, "rPPG Signal Graph", (14, 24), cv.FONT_HERSHEY_SIMPLEX, 0.62, (220, 220, 220), 1)

    if signal is None or len(signal) < 2:
        cv.putText(canvas, "Sinal insuficiente", (14, 56), cv.FONT_HERSHEY_SIMPLEX, 0.62, (0, 160, 255), 2)
        return canvas

    sig = signal.astype(np.float64)
    sig = sig - np.mean(sig)
    peak = np.max(np.abs(sig))
    if peak < 1e-8:
        cv.putText(canvas, "Sinal quase constante", (14, 56), cv.FONT_HERSHEY_SIMPLEX, 0.62, (0, 160, 255), 2)
        return canvas

    sig = sig / peak
    left, right = 18, width - 18
    top, bottom = 36, height - 22
    y_mid = (top + bottom) // 2
    amp = max(10, (bottom - top) // 2 - 4)

    cv.line(canvas, (left, y_mid), (right, y_mid), (70, 70, 70), 1)
    x = np.linspace(left, right, num=len(sig)).astype(np.int32)
    y = (y_mid - sig * amp).astype(np.int32)
    points = np.stack([x, y], axis=1).reshape(-1, 1, 2)
    cv.polylines(canvas, [points], isClosed=False, color=(60, 220, 90), thickness=2)

    txt = f"N={len(sig)}"
    if fs is not None:
        txt += f" | fs={fs:.2f} Hz"
    if bpm is not None:
        txt += f" | BPM={bpm:.1f}"
    cv.putText(canvas, txt, (14, height - 8), cv.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1)
    return canvas


def build_fft_plot(
    heart_freqs: np.ndarray | None,
    heart_spec: np.ndarray | None,
    heart_peak_hz: float | None,
    resp_freqs: np.ndarray | None,
    resp_spec: np.ndarray | None,
    resp_peak_hz: float | None,
    width: int = 900,
    height: int = 300,
) -> np.ndarray:
    canvas = np.full((height, width, 3), 18, dtype=np.uint8)
    cv.rectangle(canvas, (0, 0), (width - 1, height - 1), (80, 80, 80), 1)
    cv.putText(canvas, "Welch PSD (heart + resp)", (14, 24), cv.FONT_HERSHEY_SIMPLEX, 0.62, (220, 220, 220), 1)

    left, right = 18, width - 18
    top, bottom = 36, height - 28
    max_hz = max(HEART_BAND_HZ[1] * 1.15, 4.5)

    rx0 = int(left + (RESP_BAND_HZ[0] / max_hz) * (right - left))
    rx1 = int(left + (RESP_BAND_HZ[1] / max_hz) * (right - left))
    hx0 = int(left + (HEART_BAND_HZ[0] / max_hz) * (right - left))
    hx1 = int(left + (HEART_BAND_HZ[1] / max_hz) * (right - left))
    cv.rectangle(canvas, (rx0, top), (rx1, bottom), (45, 35, 25), -1)
    cv.rectangle(canvas, (hx0, top), (hx1, bottom), (35, 55, 35), -1)

    def draw_curve(freqs: np.ndarray | None, spec: np.ndarray | None, color: tuple[int, int, int], thickness: int) -> None:
        if freqs is None or spec is None or len(freqs) < 2:
            return
        keep = (freqs >= 0.0) & (freqs <= max_hz)
        f = freqs[keep]
        s = spec[keep]
        if len(f) < 2:
            return
        s_shift = s - np.min(s)
        den = np.max(s_shift) - np.min(s_shift)
        s_norm = np.zeros_like(s_shift) if den < 1e-12 else s_shift / den
        x = (left + (f / max_hz) * (right - left)).astype(np.int32)
        y = (bottom - s_norm * (bottom - top)).astype(np.int32)
        pts = np.stack([x, y], axis=1).reshape(-1, 1, 2)
        cv.polylines(canvas, [pts], isClosed=False, color=color, thickness=thickness)

    draw_curve(heart_freqs, heart_spec, (80, 220, 240), 2)
    draw_curve(resp_freqs, resp_spec, (0, 180, 255), 1)

    if heart_peak_hz is not None:
        xh = int(left + (heart_peak_hz / max_hz) * (right - left))
        cv.line(canvas, (xh, top), (xh, bottom), (0, 230, 90), 2)
    if resp_peak_hz is not None:
        xr = int(left + (resp_peak_hz / max_hz) * (right - left))
        cv.line(canvas, (xr, top), (xr, bottom), (0, 180, 255), 2)

    cv.putText(canvas, f"Resp: {RESP_BAND_HZ[0]:.1f}-{RESP_BAND_HZ[1]:.1f} Hz", (width - 330, 24), cv.FONT_HERSHEY_SIMPLEX, 0.5, (180, 200, 255), 1)
    cv.putText(canvas, f"Heart: {HEART_BAND_HZ[0]:.1f}-{HEART_BAND_HZ[1]:.1f} Hz", (width - 330, 44), cv.FONT_HERSHEY_SIMPLEX, 0.5, (180, 220, 180), 1)
    return canvas


def main() -> None:
    cap = cv.VideoCapture(SOURCE)
    if not cap.isOpened():
        raise RuntimeError(f"Nao foi possivel abrir a fonte: {SOURCE}")

    reported_fps = float(cap.get(cv.CAP_PROP_FPS))
    nominal_fps = reported_fps if np.isfinite(reported_fps) and 1.0 <= reported_fps <= 240.0 else 30.0
    step_ms = max(1, int(round(1000.0 / nominal_fps)))
    print(f"[VideoSource] FPS detectado: {nominal_fps:.2f}")
    print(f"[Methods] HEART={HEART_METHOD} | RESP={RESP_METHOD}")

    detector = create_face_detector(MODEL_PATH)

    t_ms = 0
    last_pos_ms = -1
    r_buffer: deque[float] = deque()
    g_buffer: deque[float] = deque()
    b_buffer: deque[float] = deque()
    ts_buffer: deque[int] = deque()
    last_strip = None
    last_plot = None
    last_fft_plot = None
    frame_idx = -1

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

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = detector.detect_for_video(mp_image, t_ms)

        roi_mask = None
        face_box = None
        current_face_landmarks = None
        if result.face_landmarks:
            face_landmarks = result.face_landmarks[0]
            current_face_landmarks = face_landmarks
            roi_mask = build_roi_mask(frame_rgb, face_landmarks)
            face_box = get_face_rect(face_landmarks, frame_rgb.shape[1], frame_rgb.shape[0])
            roi_pixels = frame_rgb[roi_mask == 255]
            if roi_pixels.size > 0:
                r_buffer.append(float(np.mean(roi_pixels[:, 0])))
                g_buffer.append(float(np.mean(roi_pixels[:, 1])))
                b_buffer.append(float(np.mean(roi_pixels[:, 2])))
                ts_buffer.append(t_ms)

        max_window_sec = max(HEART_WINDOW_SEC, RESP_WINDOW_SEC)
        while len(ts_buffer) > 1 and (ts_buffer[-1] - ts_buffer[0]) > int(max_window_sec * 1000):
            r_buffer.popleft()
            g_buffer.popleft()
            b_buffer.popleft()
            ts_buffer.popleft()

        bpm = snr = resp_rpm = resp_snr = None
        fs = None
        sig_plot = None
        heart_freqs = heart_spec = None
        resp_freqs = resp_spec = None
        peak_hz = resp_hz = None
        chrom_alpha = 1.0

        if len(ts_buffer) > 1:
            ts_all = np.array(ts_buffer, dtype=np.int64)
            r_all = np.array(r_buffer, dtype=np.float64)
            g_all = np.array(g_buffer, dtype=np.float64)
            b_all = np.array(b_buffer, dtype=np.float64)

            heart_start = ts_all[-1] - int(HEART_WINDOW_SEC * 1000.0)
            heart_mask = ts_all >= heart_start
            ts_h = ts_all[heart_mask]
            r_h = r_all[heart_mask]
            g_h = g_all[heart_mask]
            b_h = b_all[heart_mask]

            if len(ts_h) > 1 and (ts_h[-1] - ts_h[0]) >= int(MIN_HEART_WINDOW_SEC * 1000):
                # computa alpha do CHROM temporal para usar na visualizacao espacial
                r_n, g_n, b_n = normalize(r_h), normalize(g_h), normalize(b_h)
                x_sig = 3.0 * r_n - 2.0 * g_n
                y_sig = 1.5 * r_n + g_n - 1.5 * b_n
                chrom_alpha = float(np.std(x_sig) / (np.std(y_sig) + 1e-8))

                sig_h = method_signal(HEART_METHOD, r_h, g_h, b_h)
                sig_plot = sig_h
                fs, heart_freqs, heart_spec = compute_psd_standard(
                    sig_h,
                    ts_h,
                    HEART_BAND_HZ,
                    WELCH_SEG_SEC_HEART,
                    WELCH_OVERLAP_HEART,
                )
                bpm, snr, peak_hz = estimate_rate(heart_freqs, heart_spec, HEART_BAND_HZ)

            resp_start = ts_all[-1] - int(RESP_WINDOW_SEC * 1000.0)
            resp_mask = ts_all >= resp_start
            ts_r = ts_all[resp_mask]
            r_r = r_all[resp_mask]
            g_r = g_all[resp_mask]
            b_r = b_all[resp_mask]

            if len(ts_r) > 1 and (ts_r[-1] - ts_r[0]) >= int(MIN_RESP_WINDOW_SEC * 1000):
                sig_r = method_signal(RESP_METHOD, r_r, g_r, b_r)
                _, resp_freqs, resp_spec = compute_psd_standard(
                    sig_r,
                    ts_r,
                    RESP_BAND_HZ,
                    WELCH_SEG_SEC_RESP,
                    WELCH_OVERLAP_RESP,
                )
                resp_rpm, resp_snr, resp_hz = estimate_rate(resp_freqs, resp_spec, RESP_BAND_HZ)

        if frame_idx % RENDER_EVERY_N_FRAMES == 0 or last_strip is None:
            stage_landmarks = frame_bgr.copy()
            if face_box is not None:
                x, y, w, h = face_box
                cv.rectangle(stage_landmarks, (x, y), (x + w, y + h), (60, 220, 60), 2)
            if current_face_landmarks is not None:
                stage_landmarks = draw_roi_landmark_ids(stage_landmarks, current_face_landmarks)
            stage_landmarks = label(stage_landmarks, "Facial Landmarks")

            if current_face_landmarks is not None:
                stage_cheeks = build_roi_chrom_tile(frame_bgr, current_face_landmarks, chrom_alpha, TARGET_TILE_HEIGHT)
            else:
                stage_cheeks = np.zeros((TARGET_TILE_HEIGHT, TARGET_TILE_HEIGHT, 3), dtype=np.uint8)
                stage_cheeks = label(stage_cheeks, "ROIs - CHROM")

            stage_final = frame_bgr.copy()
            cv.putText(stage_final, f"BPM: {bpm:.1f}" if bpm is not None else "BPM: --", (20, 80), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv.putText(stage_final, f"SNR: {snr:.1f} dB" if snr is not None else "SNR: --", (20, 115), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 220, 80), 2)
            cv.putText(stage_final, f"Resp: {resp_rpm:.1f} rpm" if resp_rpm is not None else "Resp: --", (20, 150), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 180, 255), 2)
            cv.putText(stage_final, f"Resp SNR: {resp_snr:.1f} dB" if resp_snr is not None else "Resp SNR: --", (20, 185), cv.FONT_HERSHEY_SIMPLEX, 0.72, (190, 220, 255), 2)
            stage_final = label(stage_final, f"Resultado Final ({HEART_METHOD})")

            last_strip = cv.hconcat(
                [
                    resize_keep_aspect(stage_landmarks, TARGET_TILE_HEIGHT),
                    stage_cheeks,
                    resize_keep_aspect(stage_final, TARGET_TILE_HEIGHT),
                ]
            )

            if SHOW_GRAPH:
                last_plot = build_plot(sig_plot, fs, bpm)
                last_fft_plot = build_fft_plot(
                    heart_freqs,
                    heart_spec,
                    peak_hz,
                    resp_freqs,
                    resp_spec,
                    resp_hz,
                )

        cv.imshow("rPPG Pipeline", last_strip)
        if SHOW_GRAPH and last_plot is not None:
            cv.imshow("rPPG Signal Graph", last_plot)
        if SHOW_GRAPH and last_fft_plot is not None:
            cv.imshow("rPPG Welch PSD", last_fft_plot)

        if cv.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()

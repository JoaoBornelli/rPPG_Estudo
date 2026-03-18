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
SOURCE = "media/WhatsApp Video 2026-03-17 at 20.58.53.mp4"  # 0 para webcam
WINDOW_SEC = 10.0
MIN_WINDOW_SEC = 6.0
RESP_WINDOW_SEC = 30.0
MIN_RESP_WINDOW_SEC = 12.0
BAND_HZ = (0.8, 3.2)  # 48-192 BPM
RESP_BAND_HZ = (0.1, 0.5)  # 6-30 rpm
TARGET_TILE_HEIGHT = 400
RENDER_EVERY_N_FRAMES = 1
SHOW_GRAPH = True


# ROIs simples (testa + bochechas) com índices da malha facial
FOREHEAD_IDX = [10, 67, 103, 109, 338, 297, 332, 284]
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
    # Expande levemente a testa para aumentar a area de amostragem rPPG.
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


def normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    std = np.std(x)
    if std < 1e-8:
        return np.zeros_like(x)
    return (x - np.mean(x)) / std


def pos_from_rgb(r_series: np.ndarray, g_series: np.ndarray, b_series: np.ndarray) -> np.ndarray:
    r = r_series.astype(np.float64)
    g = g_series.astype(np.float64)
    b = b_series.astype(np.float64)

    rgb = np.vstack([r, g, b])
    means = np.mean(rgb, axis=1, keepdims=True)
    means[means == 0.0] = 1.0
    c = (rgb / means) - 1.0

    x = c[1] - c[2]
    y = c[1] + c[2] - 2.0 * c[0]
    alpha = np.std(x) / (np.std(y) + 1e-8)
    pos = x + alpha * y
    return normalize(pos)


def compute_fft(signal: np.ndarray, timestamps_ms: np.ndarray) -> tuple[np.ndarray | None, float | None, np.ndarray | None, np.ndarray | None]:
    if len(signal) < 16 or len(timestamps_ms) < 2:
        return None, None, None, None

    dt = np.diff(timestamps_ms.astype(np.float64)) / 1000.0
    dt = dt[dt > 0]
    if dt.size == 0:
        return None, None, None, None
    fs = 1.0 / np.mean(dt)

    sig = signal.astype(np.float64)
    # Remove deriva muito lenta (iluminacao/exposicao), sem suprimir batimentos baixos como ~50 bpm.
    hp_cut_hz = 0.2
    spec_hp = np.fft.rfft(sig)
    freqs_hp = np.fft.rfftfreq(len(sig), d=1.0 / fs)
    spec_hp[freqs_hp < hp_cut_hz] = 0.0
    sig = np.fft.irfft(spec_hp, n=len(sig))
    sig = normalize(sig)
    n = len(sig)
    window = np.hanning(n)
    spec = np.abs(np.fft.rfft(sig * window)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    return sig, float(fs), freqs, spec


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

    b_spec = spec[mask]
    b_freqs = freqs[mask]
    idx = int(np.argmax(b_spec))
    peak_hz = float(b_freqs[idx])
    rate = peak_hz * 60.0
    peak_power = float(b_spec[idx])
    noise_power = float((np.sum(b_spec) - peak_power) / max(len(b_spec) - 1, 1))
    snr_db = 10.0 * np.log10((peak_power + 1e-12) / (noise_power + 1e-12))
    return float(rate), float(snr_db), peak_hz


def label(frame: np.ndarray, text: str) -> np.ndarray:
    out = frame.copy()
    cv.rectangle(out, (8, 8), (350, 42), (0, 0, 0), -1)
    cv.putText(out, text, (16, 32), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return out


def resize_keep_aspect(frame: np.ndarray, target_h: int) -> np.ndarray:
    h, w = frame.shape[:2]
    target_w = max(80, int(round(target_h * (w / float(max(h, 1))))))
    resized = cv.resize(frame, (target_w, target_h), interpolation=cv.INTER_AREA)
    return resized


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
    freqs: np.ndarray | None,
    spec: np.ndarray | None,
    peak_hz: float | None,
    resp_hz: float | None,
    width: int = 900,
    height: int = 300,
) -> np.ndarray:
    canvas = np.full((height, width, 3), 18, dtype=np.uint8)
    cv.rectangle(canvas, (0, 0), (width - 1, height - 1), (80, 80, 80), 1)
    cv.putText(canvas, "FFT/PSD usada no BPM", (14, 24), cv.FONT_HERSHEY_SIMPLEX, 0.62, (220, 220, 220), 1)

    if freqs is None or spec is None or len(freqs) < 2:
        cv.putText(canvas, "FFT indisponivel", (14, 56), cv.FONT_HERSHEY_SIMPLEX, 0.62, (0, 160, 255), 2)
        return canvas

    left, right = 18, width - 18
    top, bottom = 36, height - 28

    max_hz = max(BAND_HZ[1] * 1.15, 4.5)
    valid = (freqs >= 0.0) & (freqs <= max_hz)
    f = freqs[valid]
    s = spec[valid]
    if len(f) < 2:
        cv.putText(canvas, "FFT insuficiente", (14, 56), cv.FONT_HERSHEY_SIMPLEX, 0.62, (0, 160, 255), 2)
        return canvas

    s = s.astype(np.float64)
    s = np.log10(s + 1e-12)
    s -= np.min(s)
    denom = np.max(s) - np.min(s)
    if denom < 1e-8:
        s_norm = np.zeros_like(s)
    else:
        s_norm = s / denom

    # banda respiratoria destacada
    resp_x0 = int(left + (RESP_BAND_HZ[0] / max_hz) * (right - left))
    resp_x1 = int(left + (RESP_BAND_HZ[1] / max_hz) * (right - left))
    cv.rectangle(canvas, (resp_x0, top), (resp_x1, bottom), (45, 35, 25), -1)

    # banda cardiaca destacada
    band_x0 = int(left + (BAND_HZ[0] / max_hz) * (right - left))
    band_x1 = int(left + (BAND_HZ[1] / max_hz) * (right - left))
    cv.rectangle(canvas, (band_x0, top), (band_x1, bottom), (35, 55, 35), -1)

    x = (left + (f / max_hz) * (right - left)).astype(np.int32)
    y = (bottom - s_norm * (bottom - top)).astype(np.int32)
    points = np.stack([x, y], axis=1).reshape(-1, 1, 2)
    cv.polylines(canvas, [points], isClosed=False, color=(80, 220, 240), thickness=2)

    if peak_hz is not None:
        px = int(left + (peak_hz / max_hz) * (right - left))
        cv.line(canvas, (px, top), (px, bottom), (0, 230, 90), 2)
        cv.putText(canvas, f"peak={peak_hz:.2f} Hz ({peak_hz * 60.0:.1f} BPM)", (14, height - 8), cv.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1)

    if resp_hz is not None:
        rx = int(left + (resp_hz / max_hz) * (right - left))
        cv.line(canvas, (rx, top), (rx, bottom), (0, 180, 255), 2)

    cv.putText(canvas, f"Resp: {RESP_BAND_HZ[0]:.1f}-{RESP_BAND_HZ[1]:.1f} Hz", (width - 330, 24), cv.FONT_HERSHEY_SIMPLEX, 0.5, (180, 200, 255), 1)
    cv.putText(canvas, f"Cardio: {BAND_HZ[0]:.1f}-{BAND_HZ[1]:.1f} Hz", (width - 330, 44), cv.FONT_HERSHEY_SIMPLEX, 0.5, (180, 220, 180), 1)
    return canvas


def main() -> None:
    cap = cv.VideoCapture(SOURCE)
    if not cap.isOpened():
        raise RuntimeError(f"Nao foi possivel abrir a fonte: {SOURCE}")

    reported_fps = float(cap.get(cv.CAP_PROP_FPS))
    nominal_fps = reported_fps if np.isfinite(reported_fps) and 1.0 <= reported_fps <= 240.0 else 30.0
    step_ms = max(1, int(round(1000.0 / nominal_fps)))
    print(f"[VideoSource] FPS detectado: {nominal_fps:.2f}")

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

        has_face = bool(result.face_landmarks)
        roi_mask = None
        face_box = None
        r_mean = None
        g_mean = None
        b_mean = None
        if has_face:
            face_landmarks = result.face_landmarks[0]
            roi_mask = build_roi_mask(frame_rgb, face_landmarks)
            face_box = get_face_rect(face_landmarks, frame_rgb.shape[1], frame_rgb.shape[0])
            roi_pixels = frame_rgb[roi_mask == 255]
            if roi_pixels.size > 0:
                r_mean = float(np.mean(roi_pixels[:, 0]))
                g_mean = float(np.mean(roi_pixels[:, 1]))
                b_mean = float(np.mean(roi_pixels[:, 2]))

        if r_mean is not None and g_mean is not None and b_mean is not None:
            r_buffer.append(r_mean)
            g_buffer.append(g_mean)
            b_buffer.append(b_mean)
            ts_buffer.append(t_ms)

        max_window_sec = max(WINDOW_SEC, RESP_WINDOW_SEC)
        while len(ts_buffer) > 1 and (ts_buffer[-1] - ts_buffer[0]) > int(max_window_sec * 1000):
            r_buffer.popleft()
            ts_buffer.popleft()
            g_buffer.popleft()
            b_buffer.popleft()

        bpm = None
        fs = None
        snr = None
        resp_rpm = None
        resp_snr = None
        sig_plot = None
        fft_freqs = None
        fft_spec = None
        peak_hz = None
        resp_hz = None
        if len(ts_buffer) > 1:
            ts_arr_all = np.array(ts_buffer, dtype=np.int64)
            r_arr_all = np.array(r_buffer, dtype=np.float64)
            g_arr_all = np.array(g_buffer, dtype=np.float64)
            b_arr_all = np.array(b_buffer, dtype=np.float64)

            cardio_start_ts = ts_arr_all[-1] - int(WINDOW_SEC * 1000.0)
            cardio_mask = ts_arr_all >= cardio_start_ts
            ts_arr = ts_arr_all[cardio_mask]
            r_arr = r_arr_all[cardio_mask]
            g_arr = g_arr_all[cardio_mask]
            b_arr = b_arr_all[cardio_mask]

            if len(ts_arr) > 1 and (ts_arr[-1] - ts_arr[0]) >= int(MIN_WINDOW_SEC * 1000):
                pos_sig = pos_from_rgb(r_arr, g_arr, b_arr)
                sig_plot = normalize(pos_sig)
                _, fs, fft_freqs, fft_spec = compute_fft(pos_sig, ts_arr)
                if fs is not None:
                    bpm, snr, peak_hz = estimate_rate(fft_freqs, fft_spec, BAND_HZ)

            resp_start_ts = ts_arr_all[-1] - int(RESP_WINDOW_SEC * 1000.0)
            resp_mask = ts_arr_all >= resp_start_ts
            ts_resp = ts_arr_all[resp_mask]
            g_resp = g_arr_all[resp_mask]
            if len(ts_resp) > 1 and (ts_resp[-1] - ts_resp[0]) >= int(MIN_RESP_WINDOW_SEC * 1000):
                _, fs_resp, freqs_resp, spec_resp = compute_fft(normalize(g_resp), ts_resp)
                if fs_resp is not None:
                    resp_rpm, resp_snr, resp_hz = estimate_rate(freqs_resp, spec_resp, RESP_BAND_HZ)

        if frame_idx % RENDER_EVERY_N_FRAMES == 0 or last_strip is None:
            stage1 = label(frame_bgr, "Etapa 1: Frame Original")

            stage2 = frame_bgr.copy()
            if face_box is not None:
                x, y, w, h = face_box
                cv.rectangle(stage2, (x, y), (x + w, y + h), (60, 220, 60), 2)
            stage2 = label(stage2, "Etapa 2: Deteccao de Face")

            if roi_mask is None:
                stage3 = np.zeros_like(frame_bgr)
            else:
                stage3 = cv.cvtColor(roi_mask, cv.COLOR_GRAY2BGR)
            stage3 = label(stage3, "Etapa 3: Mascara ROI")

            if roi_mask is None:
                stage4 = frame_bgr.copy()
            else:
                stage4 = cv.bitwise_and(frame_bgr, frame_bgr, mask=roi_mask)
            stage4 = label(stage4, "Etapa 4: ROI Aplicada")

            if roi_mask is None:
                stage5 = np.zeros_like(frame_bgr)
            else:
                green = frame_bgr[:, :, 1]
                green_bgr = cv.cvtColor(green, cv.COLOR_GRAY2BGR)
                stage5 = cv.bitwise_and(green_bgr, green_bgr, mask=roi_mask)
            stage5 = label(stage5, "Etapa 5: Canal Verde")

            stage6 = frame_bgr.copy()
            bpm_text = f"BPM: {bpm:.1f}" if bpm is not None else "BPM: --"
            snr_text = f"SNR: {snr:.1f} dB" if snr is not None else "SNR: --"
            resp_text = f"Resp: {resp_rpm:.1f} rpm" if resp_rpm is not None else "Resp: --"
            resp_snr_text = f"Resp SNR: {resp_snr:.1f} dB" if resp_snr is not None else "Resp SNR: --"
            cv.putText(stage6, bpm_text, (20, 80), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv.putText(stage6, snr_text, (20, 115), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 220, 80), 2)
            cv.putText(stage6, resp_text, (20, 150), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 180, 255), 2)
            cv.putText(stage6, resp_snr_text, (20, 185), cv.FONT_HERSHEY_SIMPLEX, 0.72, (190, 220, 255), 2)
            stage6 = label(stage6, "Etapa 6: Saida Final")

            last_strip = cv.hconcat(
                [
                    resize_keep_aspect(stage1, TARGET_TILE_HEIGHT),
                    resize_keep_aspect(stage2, TARGET_TILE_HEIGHT),
                    resize_keep_aspect(stage3, TARGET_TILE_HEIGHT),
                    resize_keep_aspect(stage4, TARGET_TILE_HEIGHT),
                    resize_keep_aspect(stage5, TARGET_TILE_HEIGHT),
                    resize_keep_aspect(stage6, TARGET_TILE_HEIGHT),
                ]
            )
            if SHOW_GRAPH:
                last_plot = build_plot(sig_plot, fs, bpm)
                last_fft_plot = build_fft_plot(fft_freqs, fft_spec, peak_hz, resp_hz)

        cv.imshow("Pipeline Stages (1..6)", last_strip)
        if SHOW_GRAPH and last_plot is not None:
            cv.imshow("rPPG Signal Graph", last_plot)
        if SHOW_GRAPH and last_fft_plot is not None:
            cv.imshow("rPPG FFT Spectrum", last_fft_plot)

        if cv.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()

"""
PERCLOS + EAR + blinks.

Fluxo:
  1. PerclosTracker(): cria com defaults da literatura.
  2. .feed_calibration_open(ear_avg) durante CAL_OPEN_SEC.
  3. .finish_calibration_open() — registra ear_open.
  4. .feed_calibration_closed(ear_avg, ts_ms) durante CAL_CLOSED_SEC.
  5. .finish_calibration_closed() — registra ear_closed e perclos_threshold.
  6. .update(ear_right, ear_left, now_monotonic) → dict com state corrente.
"""
from collections import deque
import numpy as np

from modules.capture import get_landmark_px

# Landmarks dos olhos (6 pontos cada)
RIGHT_EYE_EAR = [33, 159, 158, 133, 153, 145]
LEFT_EYE_EAR = [362, 386, 385, 263, 380, 374]

# Calibração
CAL_OPEN_SEC = 5.0
CAL_CLOSED_SEC = 5.0
CAL_CLOSED_SKIP = 1.0
CAL_CLOSED_MEASURE = 2.0

# Suavização
MOVING_AVG_LEN = 7

# Piscadas
BLINK_MIN_FRAMES = 1
BLINK_MAX_FRAMES = 8
BLINK_INTERVAL_WINDOW = 5

# Fadiga
FATIGUE_WINDOW_SEC = 180.0
PERCLOS_80_RATIO = 0.80
FATIGUE_ALERT_PERCLOS = 15.0


def eye_aspect_ratio(face_landmarks, eye_indices, w, h):
    p = [get_landmark_px(face_landmarks, eye_indices[i], w, h) for i in range(6)]
    v1 = np.linalg.norm(p[1] - p[5])
    v2 = np.linalg.norm(p[2] - p[4])
    horiz = np.linalg.norm(p[0] - p[3])
    if horiz < 1e-6:
        return 0.0
    return (v1 + v2) / (2.0 * horiz)


def _moving_average(buf):
    return sum(buf) / len(buf) if buf else 0.0


class PerclosTracker:
    def __init__(self):
        self.ear_open = None
        self.ear_closed = None
        self.perclos_threshold = None

        self._cal_open_samples = []
        self._cal_closed_samples_ts = []
        self._cal_start_ms = None

        self.ear_buf_r = deque(maxlen=MOVING_AVG_LEN)
        self.ear_buf_l = deque(maxlen=MOVING_AVG_LEN)
        self.blink_closed_frames = 0
        self.in_blink = False
        self.blink_last_time = None
        self.blink_intervals = deque(maxlen=BLINK_INTERVAL_WINDOW)
        self.fatigue_buf = deque()

    def calibrated(self):
        return self.ear_open is not None and self.ear_closed is not None

    # --- Calibração ---
    def feed_calibration_open(self, ear_avg):
        self._cal_open_samples.append(ear_avg)

    def finish_calibration_open(self):
        self.ear_open = float(np.median(self._cal_open_samples))

    def feed_calibration_closed(self, ear_avg, ts_ms):
        self._cal_closed_samples_ts.append((ts_ms, ear_avg))

    def finish_calibration_closed(self, cal_start_ms):
        start_ms = cal_start_ms + int(CAL_CLOSED_SKIP * 1000)
        end_ms = start_ms + int(CAL_CLOSED_MEASURE * 1000)
        stable = [ear for ts, ear in self._cal_closed_samples_ts if start_ms <= ts <= end_ms]
        if not stable:
            stable = [ear for _, ear in self._cal_closed_samples_ts]
        self.ear_closed = float(np.median(stable))
        ear_range = self.ear_open - self.ear_closed
        self.perclos_threshold = self.ear_closed + ear_range * (1.0 - PERCLOS_80_RATIO)

    # --- Monitoramento ---
    def update(self, ear_right, ear_left, now_monotonic):
        """
        Retorna dict com:
          ear_avg, pct_r, pct_l, alert_r, alert_l, blinks_per_min,
          perclos_pct, fatigue_alert, buf_duration
        """
        self.ear_buf_r.append(ear_right)
        self.ear_buf_l.append(ear_left)
        smooth_r = _moving_average(self.ear_buf_r)
        smooth_l = _moving_average(self.ear_buf_l)
        ear_avg = (ear_right + ear_left) / 2.0

        ear_range = self.ear_open - self.ear_closed
        if ear_range > 1e-6:
            pct_r = max(0.0, min(100.0, (smooth_r - self.ear_closed) / ear_range * 100.0))
            pct_l = max(0.0, min(100.0, (smooth_l - self.ear_closed) / ear_range * 100.0))
        else:
            pct_r = pct_l = 100.0

        alert_r = pct_r < 80.0
        alert_l = pct_l < 80.0

        # Piscadas
        blink_th = self.ear_closed + ear_range * 0.45
        if ear_avg < blink_th:
            self.blink_closed_frames += 1
        else:
            if self.in_blink:
                self.in_blink = False
            if BLINK_MIN_FRAMES <= self.blink_closed_frames <= BLINK_MAX_FRAMES:
                if self.blink_last_time is not None:
                    self.blink_intervals.append(now_monotonic - self.blink_last_time)
                self.blink_last_time = now_monotonic
            self.blink_closed_frames = 0
        if self.blink_closed_frames == BLINK_MIN_FRAMES:
            self.in_blink = True

        if self.blink_intervals:
            avg_interval = sum(self.blink_intervals) / len(self.blink_intervals)
            blinks_per_min = 60.0 / avg_interval if avg_interval > 0 else 0.0
        else:
            blinks_per_min = 0.0

        # PERCLOS (janela rolling 180s)
        eyes_closed_p80 = alert_r or alert_l
        self.fatigue_buf.append((now_monotonic, eyes_closed_p80))
        cutoff = now_monotonic - FATIGUE_WINDOW_SEC
        while self.fatigue_buf and self.fatigue_buf[0][0] < cutoff:
            self.fatigue_buf.popleft()
        total = len(self.fatigue_buf)
        closed_count = sum(1 for _, c in self.fatigue_buf if c)
        perclos_pct = (closed_count / total * 100.0) if total > 0 else 0.0
        fatigue_alert = perclos_pct > FATIGUE_ALERT_PERCLOS
        buf_duration = (
            self.fatigue_buf[-1][0] - self.fatigue_buf[0][0]
            if total > 1 else 0.0
        )

        return {
            "ear_avg": ear_avg,
            "pct_r": pct_r,
            "pct_l": pct_l,
            "alert_r": alert_r,
            "alert_l": alert_l,
            "blinks_per_min": blinks_per_min,
            "perclos_pct": perclos_pct,
            "fatigue_alert": fatigue_alert,
            "buf_duration": buf_duration,
        }

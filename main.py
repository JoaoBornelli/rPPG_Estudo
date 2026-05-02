"""
Check-in multi-métrica de aptidão para direção.

Fluxo:
  1. Seleção de perfil (UI OpenCV).
  2. Calibração PERCLOS (5s aberto + 5s fechado).
  3. Check-in 3 min:
       - Thread auxiliar: captura OpenCV + PERCLOS + rPPG (passivo).
       - Main thread: PVT-B em janela Pygame.
  4. Painel de resultados (4 tiles + semáforo) com opção de marcar
     "descansado" e salvar (CSV + atualiza baseline pessoal).
"""
import csv
import json
import os
import threading
from collections import deque
from datetime import datetime

import cv2 as cv
import numpy as np

from modules.capture import Capture
from modules.perclos import (
    PerclosTracker, RIGHT_EYE_EAR, LEFT_EYE_EAR, eye_aspect_ratio,
    CAL_OPEN_SEC, CAL_CLOSED_SEC,
)
from modules.rppg import RPPGTracker
from modules.hrv import compute_hrv
from modules.pvt import run as run_pvt, compute_pvt_metrics
from modules import subjects
from modules.thresholds import (
    evaluate_metric, overall_status,
    GREEN, YELLOW, RED, EXPERIMENTAL, INVALID,
)
from modules.ui_panel import select_profile, show_results


SESSIONS_DIR = os.path.join("data", "sessions")
CSV_FIELDS = [
    "timestamp", "subject",
    "perclos_pct", "blinks_per_min",
    "bpm", "rpm", "snr_db",
    "rmssd_ms", "pnn50_pct", "sdnn_ms",
    "pvt_n_trials", "pvt_mean_rt_ms", "pvt_mean_inv_rt",
    "pvt_lapses", "pvt_false_starts", "pvt_slowest_10pct_inv_rt",
    "pvt_trials_json",
    "marked_rested",
    "status_per_axis_json", "status_overall",
]


def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def _save_session_csv(subject_name, metrics, trials, evaluations, overall, marked_rested):
    _ensure_dir(SESSIONS_DIR)
    ts = datetime.now().strftime("%Y-%m-%d_%H%M")
    safe_name = subject_name.replace(" ", "_").replace("/", "_")
    path = os.path.join(SESSIONS_DIR, f"{ts}_{safe_name}.csv")
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "subject": subject_name,
        "perclos_pct": metrics.get("perclos_pct"),
        "blinks_per_min": metrics.get("blinks_per_min"),
        "bpm": metrics.get("bpm"),
        "rpm": metrics.get("rpm"),
        "snr_db": metrics.get("snr_db"),
        "rmssd_ms": metrics.get("rmssd_ms"),
        "pnn50_pct": metrics.get("pnn50_pct"),
        "sdnn_ms": metrics.get("sdnn_ms"),
        "pvt_n_trials": metrics.get("pvt_n_trials"),
        "pvt_mean_rt_ms": metrics.get("pvt_mean_rt_ms"),
        "pvt_mean_inv_rt": metrics.get("pvt_mean_inv_rt"),
        "pvt_lapses": metrics.get("pvt_lapses"),
        "pvt_false_starts": metrics.get("pvt_false_starts"),
        "pvt_slowest_10pct_inv_rt": metrics.get("pvt_slowest_10pct_inv_rt"),
        "pvt_trials_json": json.dumps(trials),
        "marked_rested": marked_rested,
        "status_per_axis_json": json.dumps(evaluations),
        "status_overall": overall,
    }
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader(); w.writerow(row)
    return path


def _calibrate_perclos(cap, perclos):
    """
    Conduz a calibração de 5s aberto + 5s fechado.
    Mostra preview com barra de progresso na janela "Check-in - Calibracao".
    Retorna True se calibrou, False se câmera/face falhou.
    """
    cal_phase = "open"
    cal_start_ms = None
    while not perclos.calibrated():
        out = cap.next_frame()
        if out is None:
            return False
        rgb, det, ts = out
        annotated = rgb.copy()
        h, w = rgb.shape[:2]

        if not det.face_landmarks:
            cv.putText(annotated, "Sem face — alinhe-se a camera", (10, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        else:
            lm = det.face_landmarks[0]
            ear_r = eye_aspect_ratio(lm, RIGHT_EYE_EAR, w, h)
            ear_l = eye_aspect_ratio(lm, LEFT_EYE_EAR, w, h)
            ear_avg = (ear_r + ear_l) / 2.0

            if cal_start_ms is None:
                cal_start_ms = ts
            elapsed = (ts - cal_start_ms) / 1000.0

            if cal_phase == "open":
                perclos.feed_calibration_open(ear_avg)
                progress = min(1.0, elapsed / CAL_OPEN_SEC)
                _draw_cal_bar(annotated, w, "OLHOS ABERTOS", progress, (0, 255, 0))
                if elapsed >= CAL_OPEN_SEC:
                    perclos.finish_calibration_open()
                    cal_phase = "closed"
                    cal_start_ms = ts
            elif cal_phase == "closed":
                perclos.feed_calibration_closed(ear_avg, ts)
                progress = min(1.0, elapsed / CAL_CLOSED_SEC)
                _draw_cal_bar(annotated, w, "FECHE OS OLHOS", progress, (0, 0, 255))
                if elapsed >= CAL_CLOSED_SEC:
                    perclos.finish_calibration_closed(cal_start_ms)

        cv.imshow("Check-in - Calibracao", cv.cvtColor(annotated, cv.COLOR_RGB2BGR))
        if cv.waitKey(1) & 0xFF == 27:
            return False

    cv.destroyWindow("Check-in - Calibracao")
    return True


def _draw_cal_bar(annotated, w, label, progress, color):
    bar_w = 300
    bar_x = w // 2 - bar_w // 2
    cv.rectangle(annotated, (bar_x, 10), (bar_x + bar_w, 35), (80, 80, 80), -1)
    cv.rectangle(annotated, (bar_x, 10), (bar_x + int(bar_w * progress), 35), color, -1)
    cv.putText(annotated, label, (bar_x - 20, 55),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def _passive_loop(stop_event, cap, perclos, rppg, snapshot):
    """
    Roda na thread auxiliar durante o PVT. Atualiza snapshot dict in-place.
    """
    while not stop_event.is_set():
        out = cap.next_frame()
        if out is None:
            break
        rgb, det, ts = out
        if not det.face_landmarks:
            cv.putText(rgb, "Sem face", (10, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        else:
            lm = det.face_landmarks[0]
            h, w = rgb.shape[:2]
            ear_r = eye_aspect_ratio(lm, RIGHT_EYE_EAR, w, h)
            ear_l = eye_aspect_ratio(lm, LEFT_EYE_EAR, w, h)
            state = perclos.update(ear_r, ear_l, ts / 1000.0)
            snapshot["perclos_pct"] = state["perclos_pct"]
            snapshot["blinks_per_min"] = state["blinks_per_min"]

            rppg.feed_frame(rgb, lm, ts)
            rppg.maybe_recompute(ts)
            if rppg.smooth_bpm is not None:
                snapshot["bpm"] = rppg.smooth_bpm
            if rppg.smooth_rpm is not None:
                snapshot["rpm"] = rppg.smooth_rpm
            if rppg.last_heart is not None:
                snapshot["snr_db"] = rppg.last_heart["snr_db"]

            # HRV gated on SNR ≥ 2dB (spec §6.5 caveat 2 — low SNR pollutes IBI)
            ibi = rppg.get_ibi_buffer(window_sec=60.0)
            snr_ok = (rppg.last_heart is not None
                      and rppg.last_heart.get("snr_db", -99.0) >= 2.0)
            if ibi is not None and snr_ok:
                hrv = compute_hrv(ibi)
                if hrv is not None:
                    snapshot.update(hrv)


def main():
    # 1. Seleção de perfil
    try:
        profiles = subjects.list_profiles()
    except (RuntimeError, json.JSONDecodeError) as e:
        print(f"Erro lendo data/profiles.json: {e}")
        print("Renomeie/remova o arquivo e tente novamente.")
        return
    action, name = select_profile(profiles)
    if action == "quit":
        return
    if action == "new":
        try:
            subjects.create_profile(name)
        except ValueError as e:
            print(f"Erro: {e}")
            return
    profile = subjects.load_profile(name)

    # 2. Captura + calibração PERCLOS
    cap = Capture()
    perclos = PerclosTracker()
    if not _calibrate_perclos(cap, perclos):
        print("Calibração abortada.")
        cap.release()
        cv.destroyAllWindows()
        return

    # 3. Check-in 3 min: passivo em thread, PVT no main thread
    rppg = RPPGTracker()
    snapshot = {}
    stop_event = threading.Event()
    passive_thread = threading.Thread(
        target=_passive_loop,
        args=(stop_event, cap, perclos, rppg, snapshot),
        daemon=True,
    )
    passive_thread.start()

    # PVT-B (3 min) — bloqueante
    trials = run_pvt(duration_sec=180.0)
    pvt_metrics = compute_pvt_metrics(trials)
    # Spec §8.4: descartar PVT se < 10 trials válidos
    if pvt_metrics["n_trials"] < 10:
        pvt_metrics = {
            "n_trials": pvt_metrics["n_trials"],
            "mean_rt_ms": None, "mean_inv_rt": None,
            "lapses": None, "slowest_10pct_inv_rt": None,
            "false_starts": pvt_metrics["false_starts"],
        }

    stop_event.set()
    passive_thread.join()  # no timeout — thread checks stop_event every iteration
    snapshot = dict(snapshot)  # atomic snapshot copy after writer thread terminated
    cap.release()

    # 4. Agrega métricas e avalia
    metrics = {
        "perclos_pct": snapshot.get("perclos_pct"),
        "blinks_per_min": snapshot.get("blinks_per_min"),
        "bpm": snapshot.get("bpm"),
        "rpm": snapshot.get("rpm"),
        "snr_db": snapshot.get("snr_db"),
        "rmssd_ms": snapshot.get("rmssd_ms"),
        "pnn50_pct": snapshot.get("pnn50_pct"),
        "sdnn_ms": snapshot.get("sdnn_ms"),
        "pvt_n_trials": pvt_metrics["n_trials"],
        "pvt_mean_rt_ms": pvt_metrics["mean_rt_ms"],
        "pvt_mean_inv_rt": pvt_metrics["mean_inv_rt"],
        "pvt_lapses": pvt_metrics["lapses"],
        "pvt_false_starts": pvt_metrics["false_starts"],
        "pvt_slowest_10pct_inv_rt": pvt_metrics["slowest_10pct_inv_rt"],
    }

    evaluations = {
        "perclos": evaluate_metric("perclos_pct", metrics["perclos_pct"], profile),
        "rppg": evaluate_metric("bpm", metrics["bpm"], profile) if metrics["bpm"] else INVALID,
        "hrv": evaluate_metric("rmssd_ms", metrics["rmssd_ms"], profile),
        "pvt": _combine_pvt_evals(metrics, profile),
    }
    overall = overall_status(evaluations)

    # 5. Painel + save
    action, marked_rested = show_results(metrics, evaluations, overall, name)
    if action == "save":
        path = _save_session_csv(
            name, metrics, trials, evaluations, overall, marked_rested
        )
        # Welford (apenas se descansado)
        baseline_metrics = {
            "perclos_pct": metrics["perclos_pct"],
            "bpm": metrics["bpm"],
            "rmssd_ms": metrics["rmssd_ms"],
            "pvt_mean_inv_rt": metrics["pvt_mean_inv_rt"],
            "pvt_lapses": metrics["pvt_lapses"],
        }
        subjects.save_session(name, baseline_metrics, marked_rested)
        print(f"Sessão salva: {path}")
    else:
        print("Sessão descartada.")

    cv.destroyAllWindows()


def _combine_pvt_evals(metrics, profile):
    """Combina mean_inv_rt e lapses no pior dos dois."""
    e1 = evaluate_metric("pvt_mean_inv_rt", metrics["pvt_mean_inv_rt"], profile)
    e2 = evaluate_metric("pvt_lapses", metrics["pvt_lapses"], profile)
    rank = {GREEN: 0, YELLOW: 1, RED: 2, INVALID: -1}
    valid = [s for s in (e1, e2) if s in rank and rank[s] >= 0]
    if not valid:
        return INVALID
    return max(valid, key=lambda s: rank[s])


if __name__ == "__main__":
    main()

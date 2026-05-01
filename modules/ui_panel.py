"""
UI: tela de seleção de perfil + painel de resultados final.

Tudo desenhado em janelas OpenCV, interação via teclado.
"""
import cv2 as cv
import numpy as np

from modules.thresholds import GREEN, YELLOW, RED, EXPERIMENTAL, INVALID

PANEL_W = 900
PANEL_H = 600

COLOR_BG = (30, 30, 30)
COLOR_TEXT = (235, 235, 235)
COLOR_DIM = (160, 160, 160)
COLOR_GREEN = (60, 200, 90)
COLOR_YELLOW = (40, 180, 220)
COLOR_RED = (60, 60, 230)
COLOR_GRAY = (130, 130, 130)


def _color_for(status):
    if status == GREEN: return COLOR_GREEN
    if status == YELLOW: return COLOR_YELLOW
    if status == RED: return COLOR_RED
    if isinstance(status, str) and status.startswith(EXPERIMENTAL):
        # experimental:green/yellow/red — usa tom mais apagado da cor base
        sub = status.split(":", 1)[1] if ":" in status else GREEN
        base = _color_for(sub)
        return tuple(int(c * 0.6) for c in base)
    return COLOR_GRAY


def select_profile(profile_names):
    """
    Mostra lista de perfis + opção de criar novo. Bloqueante.
    Retorna ("existing", name) ou ("new", typed_name) ou ("quit", None).
    Teclas: setas para navegar, Enter pra confirmar, 'n' pra novo perfil, ESC sai.
    """
    options = profile_names + ["+ Novo perfil"]
    cursor = 0
    typing = False
    typed = ""

    while True:
        canvas = np.full((PANEL_H, PANEL_W, 3), COLOR_BG[::-1], dtype=np.uint8)  # BGR
        cv.putText(canvas, "Selecione o perfil:", (40, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_TEXT[::-1], 2)
        if typing:
            cv.putText(canvas, f"Nome: {typed}_", (60, 140),
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_GREEN[::-1], 2)
            cv.putText(canvas, "Enter pra criar  |  ESC pra cancelar", (60, 180),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_DIM[::-1], 1)
        else:
            for i, opt in enumerate(options):
                color = COLOR_GREEN[::-1] if i == cursor else COLOR_TEXT[::-1]
                prefix = "> " if i == cursor else "  "
                cv.putText(canvas, f"{prefix}{opt}", (60, 130 + i * 36),
                           cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv.putText(canvas, "Setas: navegar  |  Enter: confirmar  |  ESC: sair",
                       (40, PANEL_H - 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_DIM[::-1], 1)

        cv.imshow("Check-in - Selecao de perfil", canvas)
        key = cv.waitKey(50) & 0xFF

        if typing:
            if key == 27:  # ESC
                typing = False; typed = ""
            elif key == 13:  # Enter
                if typed.strip():
                    cv.destroyWindow("Check-in - Selecao de perfil")
                    return ("new", typed.strip())
            elif key == 8:  # backspace
                typed = typed[:-1]
            elif 32 <= key <= 126:
                typed += chr(key)
        else:
            if key == 27:  # ESC
                cv.destroyWindow("Check-in - Selecao de perfil")
                return ("quit", None)
            elif key == 82 or key == ord('w'):  # up
                cursor = (cursor - 1) % len(options)
            elif key == 84 or key == ord('s'):  # down
                cursor = (cursor + 1) % len(options)
            elif key == 13:  # Enter
                if cursor == len(options) - 1:
                    typing = True
                else:
                    cv.destroyWindow("Check-in - Selecao de perfil")
                    return ("existing", options[cursor])
            elif key == ord('n'):
                typing = True


def draw_results_panel(metrics, evaluations, overall, subject_name, marked_rested):
    """
    Retorna BGR canvas com 4 tiles + status geral.

    metrics: dict com chaves perclos_pct, blinks_per_min, bpm, rpm, snr_db,
             rmssd_ms, pnn50_pct, pvt_n_trials, pvt_mean_inv_rt, pvt_lapses,
             pvt_false_starts.
    evaluations: dict {axis_name: status_string}.
    overall: status string ("green"/"yellow"/"red").
    """
    canvas = np.full((PANEL_H, PANEL_W, 3), COLOR_BG[::-1], dtype=np.uint8)

    # Header
    cv.putText(canvas, f"CHECK-IN  -  {subject_name}", (30, 40),
               cv.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_TEXT[::-1], 2)

    # Tiles 2x2
    tiles = [
        ("PERCLOS", "perclos", [
            ("PERCLOS", f"{metrics.get('perclos_pct', 0):.1f}%"),
            ("Piscadas", f"{metrics.get('blinks_per_min', 0):.0f}/min"),
        ]),
        ("rPPG", "rppg", [
            ("BPM", f"{metrics.get('bpm', 0):.0f}" if metrics.get('bpm') else "-"),
            ("RPM", f"{metrics.get('rpm', 0):.0f}" if metrics.get('rpm') else "-"),
            ("SNR", f"{metrics.get('snr_db', 0):.1f} dB" if metrics.get('snr_db') is not None else "-"),
        ]),
        ("HRV (experimental)", "hrv", [
            ("RMSSD", f"{metrics.get('rmssd_ms', 0):.0f} ms" if metrics.get('rmssd_ms') else "-"),
            ("pNN50", f"{metrics.get('pnn50_pct', 0):.1f}%" if metrics.get('pnn50_pct') is not None else "-"),
        ]),
        ("PVT-B", "pvt", [
            ("Mean 1/RT", f"{metrics.get('pvt_mean_inv_rt', 0):.2f} resp/s" if metrics.get('pvt_mean_inv_rt') else "-"),
            ("Lapses", f"{metrics.get('pvt_lapses', 0)}"),
            ("False starts", f"{metrics.get('pvt_false_starts', 0)}"),
            ("Trials", f"{metrics.get('pvt_n_trials', 0)}"),
        ]),
    ]

    tile_w, tile_h = 410, 200
    margin_x, margin_y = 30, 70
    gap = 20
    positions = [(margin_x, margin_y),
                 (margin_x + tile_w + gap, margin_y),
                 (margin_x, margin_y + tile_h + gap),
                 (margin_x + tile_w + gap, margin_y + tile_h + gap)]

    for (title, axis_key, lines), (x, y) in zip(tiles, positions):
        status = evaluations.get(axis_key, INVALID)
        color = _color_for(status)
        # tile bg
        cv.rectangle(canvas, (x, y), (x + tile_w, y + tile_h), (50, 50, 50), -1)
        cv.rectangle(canvas, (x, y), (x + tile_w, y + tile_h), color, 3)
        # title
        cv.putText(canvas, title, (x + 15, y + 30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT[::-1], 2)
        # status badge
        cv.circle(canvas, (x + tile_w - 30, y + 25), 12, color, -1)
        # lines
        for i, (k, v) in enumerate(lines):
            cv.putText(canvas, f"{k}: {v}", (x + 15, y + 75 + i * 32),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT[::-1], 1)

    # Overall
    overall_color = _color_for(overall)
    cv.rectangle(canvas, (30, PANEL_H - 90), (PANEL_W - 30, PANEL_H - 30),
                 overall_color, -1)
    cv.putText(canvas, f"STATUS GERAL: {overall.upper()}",
               (50, PANEL_H - 50), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

    rested_label = "[X] descansado" if marked_rested else "[ ] descansado"
    cv.putText(canvas, f"R: alternar {rested_label}    S: salvar e sair    ESC: sair sem salvar",
               (30, PANEL_H - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_DIM[::-1], 1)

    return canvas


def show_results(metrics, evaluations, overall, subject_name):
    """
    Loop bloqueante mostrando o painel. Retorna (action, marked_rested):
      action = 'save' | 'discard'
    """
    marked_rested = False
    while True:
        canvas = draw_results_panel(metrics, evaluations, overall, subject_name, marked_rested)
        cv.imshow("Check-in - Resultado", canvas)
        key = cv.waitKey(50) & 0xFF
        if key == ord('r'):
            marked_rested = not marked_rested
        elif key == ord('s'):
            cv.destroyWindow("Check-in - Resultado")
            return ("save", marked_rested)
        elif key == 27:
            cv.destroyWindow("Check-in - Resultado")
            return ("discard", marked_rested)

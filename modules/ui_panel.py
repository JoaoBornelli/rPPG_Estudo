"""
UI: tela de seleção de perfil + painel de resultados final.

Tudo desenhado em janelas OpenCV, interação via teclado.

Visual: "operations console" aesthetic — fundo near-black warm, tipografia
antialiased, cards com left-accent strip, header/footer consistentes com
guide messages explícitas em cada tela.
"""
import cv2 as cv
import numpy as np

from modules.thresholds import GREEN, YELLOW, RED, EXPERIMENTAL, INVALID

PANEL_W = 1024
PANEL_H = 760

WINDOW_NAME = "Check-in"

# ---------- Palette (BGR for OpenCV) ----------
BG_MAIN        = (21, 17, 15)        # near-black warm
BG_CARD        = (32, 28, 25)
BG_HOVER       = (45, 40, 36)
DIVIDER        = (60, 56, 52)
BORDER         = (75, 70, 65)
TEXT_PRIMARY   = (224, 230, 232)     # warm off-white
TEXT_SECONDARY = (165, 170, 173)
TEXT_DIM       = (108, 110, 112)
ACCENT_MINT    = (118, 192, 121)     # green/positive
ACCENT_AMBER   = (61, 163, 232)      # yellow/caution
ACCENT_CORAL   = (76, 90, 229)       # red/alert
ACCENT_NEUTRAL = (180, 140, 90)      # blue-grey, neutral info / hint accent


# ---------- Helpers ----------

def _bg_canvas(h=PANEL_H, w=PANEL_W):
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:] = BG_MAIN
    return canvas


def _put(canvas, text, pos, font, scale, color, thickness=1):
    cv.putText(canvas, text, pos, font, scale, color, thickness, cv.LINE_AA)


def _text_size(text, font, scale, thickness=1):
    (tw, th), _ = cv.getTextSize(text, font, scale, thickness)
    return tw, th


def _draw_header(canvas, title, subtitle=None, right_text=None):
    w = canvas.shape[1]
    cv.line(canvas, (32, 80), (w - 32, 80), DIVIDER, 1, cv.LINE_AA)
    _put(canvas, title, (32, 44), cv.FONT_HERSHEY_DUPLEX, 0.95, TEXT_PRIMARY, 1)
    if subtitle:
        _put(canvas, subtitle, (32, 68), cv.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_SECONDARY, 1)
    if right_text:
        tw, _ = _text_size(right_text, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        _put(canvas, right_text, (w - 32 - tw, 44),
             cv.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_DIM, 1)


def _draw_footer(canvas, hints):
    """hints: list of (key, label) tuples — e.g. [('Enter', 'confirmar'), ('ESC', 'sair')]"""
    h = canvas.shape[0]
    w = canvas.shape[1]
    cv.line(canvas, (32, h - 36), (w - 32, h - 36), DIVIDER, 1, cv.LINE_AA)
    x = 32
    y = h - 14
    for key, label in hints:
        kw, _ = _text_size(key, cv.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        _put(canvas, key, (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.42, ACCENT_NEUTRAL, 1)
        _put(canvas, label, (x + kw + 6, y), cv.FONT_HERSHEY_SIMPLEX, 0.42, TEXT_DIM, 1)
        lw, _ = _text_size(label, cv.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        x += kw + lw + 28


def _draw_card(canvas, x, y, w, h, accent_color=None):
    cv.rectangle(canvas, (x, y), (x + w, y + h), BG_CARD, -1)
    if accent_color:
        cv.rectangle(canvas, (x, y), (x + 4, y + h), accent_color, -1)
    cv.rectangle(canvas, (x, y), (x + w, y + h), BORDER, 1, cv.LINE_AA)


def _draw_status_badge(canvas, cx, cy, color):
    cv.circle(canvas, (cx, cy), 12, color, 1, cv.LINE_AA)
    cv.circle(canvas, (cx, cy), 6, color, -1, cv.LINE_AA)


def _color_for(status):
    if status == GREEN: return ACCENT_MINT
    if status == YELLOW: return ACCENT_AMBER
    if status == RED: return ACCENT_CORAL
    if isinstance(status, str) and status.startswith(EXPERIMENTAL):
        sub = status.split(":", 1)[1] if ":" in status else GREEN
        base = _color_for(sub)
        return tuple(int(c * 0.6) for c in base)
    return TEXT_DIM


def _overall_label(overall):
    if overall == GREEN:   return "APTO"
    if overall == YELLOW:  return "ATENCAO"
    if overall == RED:     return "NAO RECOMENDADO"
    return "DADOS INSUFICIENTES"


# ---------- select_profile ----------

def select_profile(profile_names):
    """
    Mostra lista de perfis + opção de criar novo. Bloqueante.
    Retorna ("existing", name) ou ("new", typed_name) ou ("quit", None).
    """
    options = profile_names + ["+ Novo perfil"]
    cursor = 0
    typing = False
    typed = ""

    while True:
        canvas = _bg_canvas()
        _draw_header(
            canvas,
            "CHECK-IN",
            "Selecione um perfil para continuar — primeira vez? Pressione 'n' para criar.",
        )

        if typing:
            # Input box centered
            box_x, box_y = 32, 140
            box_w, box_h = PANEL_W - 64, 64
            cv.rectangle(canvas, (box_x, box_y), (box_x + box_w, box_y + box_h),
                         BG_CARD, -1)
            cv.rectangle(canvas, (box_x, box_y), (box_x + box_w, box_y + box_h),
                         ACCENT_NEUTRAL, 1, cv.LINE_AA)
            _put(canvas, "NOVO PERFIL", (box_x + 16, box_y - 10),
                 cv.FONT_HERSHEY_SIMPLEX, 0.42, TEXT_DIM, 1)
            _put(canvas, f"{typed}_", (box_x + 16, box_y + 42),
                 cv.FONT_HERSHEY_DUPLEX, 0.85, TEXT_PRIMARY, 1)
            _put(canvas,
                 "Digite um nome e pressione Enter. ESC cancela.",
                 (32, box_y + box_h + 36),
                 cv.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_SECONDARY, 1)

            _draw_footer(canvas, [("Enter", "criar"), ("ESC", "cancelar")])
        else:
            # List of options
            list_x = 32
            list_y = 120
            row_h = 44
            list_w = PANEL_W - 64
            for i, opt in enumerate(options):
                y = list_y + i * row_h
                if i == cursor:
                    cv.rectangle(canvas, (list_x, y), (list_x + list_w, y + row_h - 4),
                                 BG_HOVER, -1)
                    cv.rectangle(canvas, (list_x, y), (list_x + 4, y + row_h - 4),
                                 ACCENT_NEUTRAL, -1)
                    text_color = TEXT_PRIMARY
                else:
                    text_color = TEXT_SECONDARY
                _put(canvas, opt, (list_x + 20, y + 28),
                     cv.FONT_HERSHEY_DUPLEX, 0.65, text_color, 1)

            _draw_footer(canvas, [
                ("UP/DN", "navegar"),
                ("Enter", "selecionar"),
                ("n", "novo perfil"),
                ("ESC", "sair"),
            ])

        cv.imshow(WINDOW_NAME, canvas)
        key = cv.waitKey(50) & 0xFF

        if typing:
            if key == 27:  # ESC
                typing = False; typed = ""
            elif key == 13:  # Enter
                if typed.strip():
                    return ("new", typed.strip())
            elif key == 8:  # backspace
                typed = typed[:-1]
            elif 32 <= key <= 126:
                typed += chr(key)
        else:
            if key == 27:  # ESC
                return ("quit", None)
            elif key == 82 or key == ord('w'):  # up
                cursor = (cursor - 1) % len(options)
            elif key == 84 or key == ord('s'):  # down
                cursor = (cursor + 1) % len(options)
            elif key == 13:  # Enter
                if cursor == len(options) - 1:
                    typing = True
                else:
                    return ("existing", options[cursor])
            elif key == ord('n'):
                typing = True


# ---------- draw_results_panel ----------

def _draw_tile(canvas, x, y, w, h, title, subtitle, lines, criteria_hint,
               status, experimental_warn=False):
    color = _color_for(status)
    _draw_card(canvas, x, y, w, h, accent_color=color)

    # Title
    title_x = x + 16
    if experimental_warn:
        _put(canvas, title, (title_x, y + 28),
             cv.FONT_HERSHEY_DUPLEX, 0.7, TEXT_PRIMARY, 1)
        # warning tag right after title
        tw, _ = _text_size(title, cv.FONT_HERSHEY_DUPLEX, 0.7, 1)
        _put(canvas, "EXPERIMENTAL", (title_x + tw + 12, y + 28),
             cv.FONT_HERSHEY_SIMPLEX, 0.42, ACCENT_AMBER, 1)
    else:
        _put(canvas, title, (title_x, y + 28),
             cv.FONT_HERSHEY_DUPLEX, 0.7, TEXT_PRIMARY, 1)

    # Subtitle
    _put(canvas, subtitle, (title_x, y + 50),
         cv.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_SECONDARY, 1)

    # Status badge top-right
    _draw_status_badge(canvas, x + w - 28, y + 28, color)

    # Divider
    cv.line(canvas, (x + 16, y + 64), (x + w - 16, y + 64), DIVIDER, 1, cv.LINE_AA)

    # Body lines (label left, value right-aligned mono)
    for i, (k, v) in enumerate(lines):
        row_y = y + 88 + i * 26
        _put(canvas, k, (title_x, row_y),
             cv.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_SECONDARY, 1)
        vw, _ = _text_size(v, cv.FONT_HERSHEY_PLAIN, 1.4, 1)
        _put(canvas, v, (x + w - 16 - vw, row_y),
             cv.FONT_HERSHEY_PLAIN, 1.4, TEXT_PRIMARY, 1)

    # Criteria hint at bottom
    _put(canvas, criteria_hint, (title_x, y + h - 14),
         cv.FONT_HERSHEY_SIMPLEX, 0.42, TEXT_DIM, 1)


def draw_results_panel(metrics, evaluations, overall, subject_name, marked_rested):
    """
    Retorna BGR canvas (PANEL_H x PANEL_W) com 4 tiles 2x2 + banner geral.
    """
    canvas = _bg_canvas()

    _draw_header(
        canvas,
        "RESULTADO",
        "Cada eixo avalia uma dimensao da aptidao para direcao.",
        right_text=subject_name,
    )

    def _fmt(value, fmt, fallback="—"):
        if value is None:
            return fallback
        try:
            return fmt.format(value)
        except Exception:
            return fallback

    perclos_lines = [
        ("PERCLOS",  _fmt(metrics.get("perclos_pct"), "{:.1f}%")),
        ("Piscadas", _fmt(metrics.get("blinks_per_min"), "{:.0f}/min")),
    ]
    rppg_lines = [
        ("BPM", _fmt(metrics.get("bpm"), "{:.0f}")),
        ("RPM", _fmt(metrics.get("rpm"), "{:.0f}")),
        ("SNR", _fmt(metrics.get("snr_db"), "{:.1f} dB")),
    ]
    hrv_lines = [
        ("RMSSD", _fmt(metrics.get("rmssd_ms"), "{:.0f} ms")),
        ("pNN50", _fmt(metrics.get("pnn50_pct"), "{:.1f}%")),
    ]
    pvt_lines = [
        ("Mean 1/RT",    _fmt(metrics.get("pvt_mean_inv_rt"), "{:.2f} resp/s")),
        ("Lapses",       _fmt(metrics.get("pvt_lapses"), "{:d}")
                         if isinstance(metrics.get("pvt_lapses"), int)
                         else _fmt(metrics.get("pvt_lapses"), "{}")),
        ("False starts", _fmt(metrics.get("pvt_false_starts"), "{}")),
        ("Trials",       _fmt(metrics.get("pvt_n_trials"), "{}")),
    ]

    tiles = [
        ("PERCLOS", "perclos",
         "Olhos fechados (P80, janela 180s)",
         perclos_lines,
         "verde <=7.5% • amarelo <=15% • vermelho >15%",
         False),
        ("rPPG", "rppg",
         "Frequencia cardiaca + respiracao",
         rppg_lines,
         "verde 50-100 BPM • amarelo 45-110 • vermelho fora",
         False),
        ("HRV", "hrv",
         "Variabilidade cardiaca — em calibracao",
         hrv_lines,
         "RMSSD >=30ms verde • <30 amarelo • <20 vermelho",
         True),
        ("PVT-B", "pvt",
         "Vigilancia sustentada (Basner 2011)",
         pvt_lines,
         "1/RT >=3.0 verde • lapses <=1 verde",
         False),
    ]

    # Layout: 2x2 grid of 470x230 tiles
    tile_w, tile_h = 470, 230
    margin_x = 32
    top_y = 100
    gap_x = (PANEL_W - 2 * margin_x - 2 * tile_w)
    positions = [
        (margin_x, top_y),
        (margin_x + tile_w + gap_x, top_y),
        (margin_x, top_y + tile_h + 16),
        (margin_x + tile_w + gap_x, top_y + tile_h + 16),
    ]

    for (title, axis_key, subtitle, lines, criteria, exp), (x, y) in zip(tiles, positions):
        status = evaluations.get(axis_key, INVALID)
        _draw_tile(canvas, x, y, tile_w, tile_h, title, subtitle, lines,
                   criteria, status, experimental_warn=exp)

    # Overall banner
    banner_y = top_y + 2 * tile_h + 16 + 16
    banner_h = 50
    overall_color = _color_for(overall)
    cv.rectangle(canvas, (margin_x, banner_y),
                 (PANEL_W - margin_x, banner_y + banner_h),
                 BG_CARD, -1)
    cv.rectangle(canvas, (margin_x, banner_y),
                 (margin_x + 6, banner_y + banner_h),
                 overall_color, -1)
    cv.rectangle(canvas, (margin_x, banner_y),
                 (PANEL_W - margin_x, banner_y + banner_h),
                 BORDER, 1, cv.LINE_AA)

    _put(canvas, "STATUS GERAL", (margin_x + 20, banner_y + 18),
         cv.FONT_HERSHEY_SIMPLEX, 0.42, TEXT_DIM, 1)
    _put(canvas, _overall_label(overall), (margin_x + 20, banner_y + 40),
         cv.FONT_HERSHEY_DUPLEX, 0.8, overall_color, 1)

    hint_text = "qualquer eixo vermelho -> vermelho geral; HRV nao conta enquanto experimental"
    hw, _ = _text_size(hint_text, cv.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    _put(canvas, hint_text,
         (PANEL_W - margin_x - 16 - hw, banner_y + 32),
         cv.FONT_HERSHEY_SIMPLEX, 0.42, TEXT_DIM, 1)

    # Marked-rested line
    rested_y = banner_y + banner_h + 28
    if marked_rested:
        _put(canvas, "[X]", (margin_x, rested_y),
             cv.FONT_HERSHEY_DUPLEX, 0.6, ACCENT_MINT, 1)
        _put(canvas,
             "Sessao marcada como descansado — alimenta baseline pessoal",
             (margin_x + 38, rested_y),
             cv.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_PRIMARY, 1)
    else:
        _put(canvas, "[ ]", (margin_x, rested_y),
             cv.FONT_HERSHEY_DUPLEX, 0.6, TEXT_DIM, 1)
        _put(canvas,
             "Marque como descansado se voce esta alerta — ajuda a calibrar baseline",
             (margin_x + 38, rested_y),
             cv.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_DIM, 1)

    _draw_footer(canvas, [
        ("R", "alternar descansado"),
        ("S", "salvar e sair"),
        ("ESC", "descartar sessao"),
    ])

    return canvas


def show_results(metrics, evaluations, overall, subject_name):
    """
    Loop bloqueante mostrando o painel. Retorna (action, marked_rested):
      action = 'save' | 'discard'
    """
    marked_rested = False
    while True:
        canvas = draw_results_panel(metrics, evaluations, overall, subject_name, marked_rested)
        cv.imshow(WINDOW_NAME, canvas)
        key = cv.waitKey(50) & 0xFF
        if key == ord('r'):
            marked_rested = not marked_rested
        elif key == ord('s'):
            return ("save", marked_rested)
        elif key == 27:
            return ("discard", marked_rested)

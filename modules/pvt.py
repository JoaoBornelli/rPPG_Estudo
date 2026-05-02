"""
PVT-B (3 min, Basner & Dinges 2011) em janela Pygame.

run() é blocante e roda no thread chamador. Em macOS, eventos NSWindow
exigem main thread; recomenda-se chamar run() do main thread enquanto a
captura OpenCV roda em thread separada (orquestração em main.py).

Métricas extraídas no fim:
  n_trials, mean_rt_ms, mean_inv_rt, lapses,
  slowest_10pct_inv_rt, false_starts, mean_rt_ms.

Métrica primária: mean_inv_rt (Basner 2011 maior effect size).

Visual: "operations console" — fundo near-black warm, fontes TTF do sistema
quando disponíveis, mensagens de guia explícitas, mini HUD durante trials,
tela de resumo no fim.
"""
import os
import random
from time import perf_counter

# Parâmetros (PVT-B, Basner & Dinges 2011)
DURATION_SEC = 180.0
ISI_MIN_SEC = 1.0
ISI_MAX_SEC = 4.0
INSTRUCTION_SEC = 5.0
LAPSE_THRESHOLD_MS = 500.0
FALSE_START_PRE_STIM_MS = 100.0  # toque < 100ms após estímulo = false start
SLEEP_ATTACK_MS = 30000.0
TRIAL_TIMEOUT_MS = SLEEP_ATTACK_MS

WINDOW_SIZE = (640, 480)
END_SCREEN_SEC = 3.0

# ---------- Palette (RGB for Pygame) ----------
BG_RGB            = (15, 17, 21)
BG_CARD_RGB       = (25, 28, 32)
DIVIDER_RGB       = (52, 56, 60)
BORDER_RGB        = (65, 70, 75)
TEXT_PRIMARY_RGB  = (232, 230, 224)
TEXT_SECONDARY_RGB= (173, 170, 165)
TEXT_DIM_RGB      = (112, 110, 108)
ACCENT_MINT_RGB   = (121, 192, 118)
ACCENT_AMBER_RGB  = (232, 163, 61)
ACCENT_NEUTRAL_RGB= (90, 140, 180)


def _import_pygame():
    # Lazy import para não exigir display em contextos onde só usamos métricas
    import pygame
    return pygame


def _load_font(size, bold=False):
    import pygame
    candidates = ["SF Pro Display", "Helvetica Neue", "Helvetica",
                  "Avenir Next", "Verdana", "Arial"]
    for name in candidates:
        try:
            font = pygame.font.SysFont(name, size, bold=bold)
            if font is not None:
                return font
        except Exception:
            continue
    return pygame.font.SysFont(None, size, bold=bold)


def _load_mono(size, bold=False):
    import pygame
    candidates = ["SF Mono", "Menlo", "Monaco", "Consolas", "Courier New"]
    for name in candidates:
        try:
            font = pygame.font.SysFont(name, size, bold=bold)
            if font is not None:
                return font
        except Exception:
            continue
    return pygame.font.SysFont(None, size, bold=bold)


def compute_pvt_metrics(trials):
    """
    trials: lista de dicts {"t": float, "rt_ms": float|None, "false_start": bool}
    Retorna dict com métricas agregadas.
    """
    valid_rts = [
        t["rt_ms"] for t in trials
        if t.get("rt_ms") is not None
        and 100.0 <= t["rt_ms"] < 30000.0  # strict; 30000ms timeout = sleep attack, excluded
        and not t.get("false_start", False)
    ]
    n = len(valid_rts)
    false_starts = sum(1 for t in trials if t.get("false_start"))
    if n == 0:
        return {
            "n_trials": 0, "mean_rt_ms": None, "mean_inv_rt": None,
            "lapses": 0, "slowest_10pct_inv_rt": None,
            "false_starts": false_starts,
        }
    lapses = sum(1 for rt in valid_rts if rt > LAPSE_THRESHOLD_MS)
    mean_rt = sum(valid_rts) / n
    mean_inv_rt = sum(1000.0 / rt for rt in valid_rts) / n
    slowest_count = max(1, n // 10)
    slowest = sorted(valid_rts, reverse=True)[:slowest_count]
    slowest_inv = sum(1000.0 / rt for rt in slowest) / len(slowest)
    return {
        "n_trials": n,
        "mean_rt_ms": mean_rt,
        "mean_inv_rt": mean_inv_rt,
        "lapses": lapses,
        "slowest_10pct_inv_rt": slowest_inv,
        "false_starts": false_starts,
    }


def run(duration_sec=DURATION_SEC, on_finish_event=None):
    """
    Executa o teste. Blocante. Retorna lista de trials.

    Se `on_finish_event` (threading.Event) for fornecido, é setado ao retornar.
    """
    pygame = _import_pygame()
    # Centraliza janela
    os.environ["SDL_VIDEO_CENTERED"] = "1"
    pygame.init()
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("PVT-B")
    clock = pygame.time.Clock()

    # Fontes (preferência TTF do sistema)
    f_display = _load_font(56, bold=True)
    f_subtitle = _load_font(18)
    f_body = _load_font(22)
    f_body_dim = _load_font(18)
    f_caption = _load_font(14)
    f_counter = _load_mono(64, bold=True)
    f_hud = _load_mono(16)
    f_hud_label = _load_font(13)
    f_end_label = _load_font(40, bold=True)
    f_end_body = _load_mono(20)
    f_end_label_dim = _load_font(18)

    W, H = WINDOW_SIZE

    def _draw_frame():
        """Background + decorative inset border."""
        screen.fill(BG_RGB)
        pygame.draw.rect(screen, BORDER_RGB,
                         pygame.Rect(16, 16, W - 32, H - 32), width=1)

    def _blit_centered(surf, y):
        rect = surf.get_rect(center=(W // 2, y))
        screen.blit(surf, rect)

    def _draw_progress_bar(progress):
        """Thin 4px bar at very top of frame."""
        progress = max(0.0, min(1.0, progress))
        # Track
        pygame.draw.rect(screen, BG_CARD_RGB, pygame.Rect(0, 0, W, 4))
        # Fill
        fill_w = int(W * progress)
        if fill_w > 0:
            pygame.draw.rect(screen, ACCENT_NEUTRAL_RGB, pygame.Rect(0, 0, fill_w, 4))

    def render_instructions(seconds_left):
        _draw_frame()
        # Title block
        _blit_centered(f_display.render("PVT-B", True, TEXT_PRIMARY_RGB), 110)
        _blit_centered(f_subtitle.render("vigilancia sustentada", True, TEXT_DIM_RGB), 150)

        # Body text
        _blit_centered(
            f_body.render("Quando aparecer um numero", True, TEXT_PRIMARY_RGB), 220)
        _blit_centered(
            f_body.render("no centro, toque a tela", True, TEXT_PRIMARY_RGB), 250)
        _blit_centered(
            f_body.render("o mais rapido possivel.", True, TEXT_PRIMARY_RGB), 280)

        _blit_centered(
            f_body_dim.render("Nao pisque por mais de 1 segundo.",
                              True, TEXT_SECONDARY_RGB), 330)

        # Live countdown
        countdown = max(0, int(seconds_left + 0.999))
        f_count = _load_font(28, bold=True)
        _blit_centered(
            f_count.render(f"Inicio em {countdown}...", True, ACCENT_NEUTRAL_RGB), 410)

        pygame.display.flip()

    def render_trial(elapsed_ms, progress, n_trials_done, time_left_sec):
        _draw_frame()
        _draw_progress_bar(progress)

        # Counter
        counter_text = f"{int(elapsed_ms):03d}"
        surf = f_counter.render(counter_text, True, TEXT_PRIMARY_RGB)
        _blit_centered(surf, H // 2 - 10)

        # Hint below counter
        _blit_centered(f_caption.render("tap", True, TEXT_DIM_RGB), H // 2 + 50)

        # HUD bottom
        # Bottom-left: trials count
        trial_label = f_hud_label.render("trials", True, TEXT_DIM_RGB)
        trial_value = f_hud.render(str(n_trials_done), True, TEXT_SECONDARY_RGB)
        screen.blit(trial_label, (32, H - 44))
        screen.blit(trial_value, (32, H - 28))

        # Bottom-right: time remaining
        mins = int(max(0, time_left_sec) // 60)
        secs = int(max(0, time_left_sec) % 60)
        rem_text = f"{mins}:{secs:02d}"
        rem_label = f_hud_label.render("restante", True, TEXT_DIM_RGB)
        rem_value = f_hud.render(rem_text, True, TEXT_SECONDARY_RGB)
        rl_rect = rem_label.get_rect()
        rv_rect = rem_value.get_rect()
        screen.blit(rem_label, (W - 32 - rl_rect.width, H - 44))
        screen.blit(rem_value, (W - 32 - rv_rect.width, H - 28))

        pygame.display.flip()

    def render_blank_isi():
        _draw_frame()
        # Sutil "..." centralizado pra não parecer travado
        surf = f_caption.render("...", True, TEXT_DIM_RGB)
        _blit_centered(surf, H // 2)
        pygame.display.flip()

    def render_end_screen(metrics):
        _draw_frame()
        _blit_centered(f_end_label.render("Concluido", True, TEXT_PRIMARY_RGB), 90)

        # Body lines (label : value, mono value)
        lines = [
            ("Trials valid",   str(metrics.get("n_trials", 0))),
            ("Mean RT",        (f"{metrics['mean_rt_ms']:.0f} ms"
                                if metrics.get('mean_rt_ms') is not None else "—")),
            ("Mean 1/RT",      (f"{metrics['mean_inv_rt']:.2f} resp/s"
                                if metrics.get('mean_inv_rt') is not None else "—")),
            ("Lapses",         str(metrics.get("lapses", 0)
                                   if metrics.get("lapses") is not None else "—")),
            ("False starts",   str(metrics.get("false_starts", 0))),
        ]
        # Render in a centered block
        block_left = 140
        block_right = W - 140
        y = 170
        for label, value in lines:
            ls = f_body.render(label, True, TEXT_SECONDARY_RGB)
            vs = f_end_body.render(value, True, TEXT_PRIMARY_RGB)
            screen.blit(ls, (block_left, y))
            vrect = vs.get_rect()
            screen.blit(vs, (block_right - vrect.width, y))
            y += 32

        _blit_centered(
            f_end_label_dim.render("Calculando resultado...", True, TEXT_DIM_RGB),
            H - 60)
        pygame.display.flip()

    def poll_events_for_tap(deadline_perf):
        """
        Processa fila pygame até deadline. Retorna ('tap', perf_counter_at_tap)
        se o usuário tocou, ou ('timeout', deadline_perf) se passou do tempo,
        ou ('quit', None) se fechou janela.
        """
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return ("quit", None)
                if event.type in (pygame.MOUSEBUTTONDOWN, pygame.KEYDOWN, pygame.FINGERDOWN):
                    return ("tap", perf_counter())
            if perf_counter() >= deadline_perf:
                return ("timeout", deadline_perf)
            clock.tick(120)  # 120Hz polling

    trials = []

    try:
        # Instruções (5s) com countdown live
        end_instr = perf_counter() + INSTRUCTION_SEC
        last_render = 0.0
        while perf_counter() < end_instr:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    if on_finish_event: on_finish_event.set()
                    return trials
            now = perf_counter()
            if now - last_render > 0.1:
                render_instructions(end_instr - now)
                last_render = now
            clock.tick(60)

        session_start = perf_counter()

        while perf_counter() - session_start < duration_sec:
            isi = random.uniform(ISI_MIN_SEC, ISI_MAX_SEC)
            isi_deadline = perf_counter() + isi
            render_blank_isi()

            # Durante ISI, qualquer toque = false start
            result = poll_events_for_tap(isi_deadline)
            if result[0] == "quit":
                break
            if result[0] == "tap":
                trials.append({
                    "t": result[1] - session_start,
                    "rt_ms": None,
                    "false_start": True,
                })
                # consumir resto do ISI sem registrar mais false starts
                remaining = isi_deadline - perf_counter()
                if remaining > 0:
                    while perf_counter() < isi_deadline:
                        for _e in pygame.event.get():
                            pass
                        clock.tick(120)

            # Estímulo: contador subindo
            if perf_counter() - session_start >= duration_sec:
                break
            stim_t0 = perf_counter()
            tap_deadline = stim_t0 + TRIAL_TIMEOUT_MS / 1000.0
            tapped = False
            n_done = len(trials)
            while perf_counter() < tap_deadline:
                now = perf_counter()
                elapsed_ms = (now - stim_t0) * 1000.0
                progress = (now - session_start) / duration_sec
                time_left_sec = duration_sec - (now - session_start)
                render_trial(elapsed_ms, progress, n_done, time_left_sec)
                # poll non-blocking
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        if on_finish_event: on_finish_event.set()
                        return trials
                    if event.type in (pygame.MOUSEBUTTONDOWN, pygame.KEYDOWN, pygame.FINGERDOWN):
                        rt_ms = (perf_counter() - stim_t0) * 1000.0
                        is_false_start = rt_ms < FALSE_START_PRE_STIM_MS
                        trials.append({
                            "t": stim_t0 - session_start,
                            "rt_ms": rt_ms,
                            "false_start": is_false_start,
                        })
                        tapped = True
                        break
                if tapped:
                    break
                clock.tick(120)
            if not tapped:
                # sleep attack: sem resposta dentro do timeout
                trials.append({
                    "t": stim_t0 - session_start,
                    "rt_ms": TRIAL_TIMEOUT_MS,
                    "false_start": False,
                })
            # tela limpa entre trials
            render_blank_isi()

        # Tela de resumo
        try:
            end_metrics = compute_pvt_metrics(trials)
            render_end_screen(end_metrics)
            end_deadline = perf_counter() + END_SCREEN_SEC
            while perf_counter() < end_deadline:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        break
                clock.tick(60)
        except Exception:
            # Não bloqueia retorno se o resumo falhar
            pass
    finally:
        pygame.quit()
        if on_finish_event: on_finish_event.set()

    return trials

"""
PVT-B (3 min, Basner & Dinges 2011) em janela Pygame.

run() é blocante e roda no thread chamador. Em macOS, eventos NSWindow
exigem main thread; recomenda-se chamar run() do main thread enquanto a
captura OpenCV roda em thread separada (orquestração em main.py).

Métricas extraídas no fim:
  n_trials, mean_rt_ms, mean_inv_rt, lapses,
  slowest_10pct_inv_rt, false_starts, mean_rt_ms.

Métrica primária: mean_inv_rt (Basner 2011 maior effect size).
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
BG_COLOR = (0, 0, 0)
TEXT_COLOR = (240, 240, 240)


def _import_pygame():
    # Lazy import para não exigir display em contextos onde só usamos métricas
    import pygame
    return pygame


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
    font_large = pygame.font.SysFont("monospace", 64, bold=True)
    font_med = pygame.font.SysFont("Arial", 22)
    clock = pygame.time.Clock()

    def render_text(text, font, y_offset=0):
        surface = font.render(text, True, TEXT_COLOR)
        rect = surface.get_rect(center=(WINDOW_SIZE[0]//2, WINDOW_SIZE[1]//2 + y_offset))
        screen.fill(BG_COLOR)
        screen.blit(surface, rect)
        pygame.display.flip()

    def render_counter_ms(elapsed_ms):
        screen.fill(BG_COLOR)
        text = font_large.render(f"{int(elapsed_ms):03d}", True, (255, 255, 255))
        rect = text.get_rect(center=(WINDOW_SIZE[0]//2, WINDOW_SIZE[1]//2))
        screen.blit(text, rect)
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
        # Instruções
        render_text("PVT-B  —  Toque a tela quando o número aparecer",
                    font_med, y_offset=-20)
        render_text("(3 minutos)", font_med, y_offset=20)
        # esperar instrução
        end_instr = perf_counter() + INSTRUCTION_SEC
        while perf_counter() < end_instr:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    if on_finish_event: on_finish_event.set()
                    return trials
            clock.tick(60)

        session_start = perf_counter()
        next_isi_start = session_start

        while perf_counter() - session_start < duration_sec:
            isi = random.uniform(ISI_MIN_SEC, ISI_MAX_SEC)
            isi_deadline = perf_counter() + isi
            screen.fill(BG_COLOR)
            pygame.display.flip()

            # Durante ISI, qualquer toque = false start
            result = poll_events_for_tap(isi_deadline)
            if result[0] == "quit":
                break
            if result[0] == "tap":
                # false start — registra e descarta o resto do ISI
                trials.append({
                    "t": result[1] - session_start,
                    "rt_ms": None,
                    "false_start": True,
                })
                # esperar o ISI restante mesmo assim (sem permitir mais false starts)
                remaining = isi_deadline - perf_counter()
                if remaining > 0:
                    # consumir eventos sem registrar false starts adicionais
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
            while perf_counter() < tap_deadline:
                elapsed_ms = (perf_counter() - stim_t0) * 1000.0
                render_counter_ms(elapsed_ms)
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
            # screen pretinha por 200ms entre trials
            screen.fill(BG_COLOR)
            pygame.display.flip()
    finally:
        pygame.quit()
        if on_finish_event: on_finish_event.set()

    return trials

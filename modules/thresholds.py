"""
Avaliação híbrida de cada eixo (PERCLOS, BPM/rPPG, HRV, PVT) em verde/amarelo/vermelho.

Estratégia:
- Se baseline pessoal tem < MIN_BASELINE_SESSIONS sessões, usa cutoffs da literatura.
- Caso contrário, usa baseline pessoal: vermelho < mean - 1.5*std, amarelo < mean - 0.75*std.
- HRV sempre marcado como 'experimental' até validação contra ground truth.
"""

GREEN = "green"
YELLOW = "yellow"
RED = "red"
EXPERIMENTAL = "experimental"
INVALID = "invalid"

MIN_BASELINE_SESSIONS = 5

# Cutoffs absolutos da literatura
ABSOLUTE = {
    # PERCLOS já tem alerta no código (15% = vermelho); estendendo para 3-níveis
    "perclos_pct": {"green_max": 7.5, "yellow_max": 15.0},
    # rPPG / cardio — adultos saudáveis em repouso
    "bpm": {"green_range": (50, 100), "yellow_range": (45, 110)},
    # HRV (ressalva: variância normativa enorme — usar baseline pessoal assim que possível)
    "rmssd_ms": {"red_below": 20.0, "yellow_below": 30.0},
    # PVT primárias (Basner 2011 não publica cutoffs binários — heurístico inicial)
    "pvt_mean_inv_rt": {"red_below": 2.5, "yellow_below": 3.0},
    "pvt_lapses": {"green_max": 1, "yellow_max": 3},
}

# Métricas que sempre passam por status experimental (não conta no status geral)
EXPERIMENTAL_METRICS = {"rmssd_ms", "pnn50_pct", "sdnn_ms"}


def _absolute_perclos(value):
    if value <= ABSOLUTE["perclos_pct"]["green_max"]: return GREEN
    if value <= ABSOLUTE["perclos_pct"]["yellow_max"]: return YELLOW
    return RED


def _absolute_bpm(value):
    g_lo, g_hi = ABSOLUTE["bpm"]["green_range"]
    y_lo, y_hi = ABSOLUTE["bpm"]["yellow_range"]
    if g_lo <= value <= g_hi: return GREEN
    if y_lo <= value <= y_hi: return YELLOW
    return RED


def _absolute_rmssd(value):
    if value < ABSOLUTE["rmssd_ms"]["red_below"]: return RED
    if value < ABSOLUTE["rmssd_ms"]["yellow_below"]: return YELLOW
    return GREEN


def _absolute_pvt_inv_rt(value):
    if value < ABSOLUTE["pvt_mean_inv_rt"]["red_below"]: return RED
    if value < ABSOLUTE["pvt_mean_inv_rt"]["yellow_below"]: return YELLOW
    return GREEN


def _absolute_pvt_lapses(value):
    if value <= ABSOLUTE["pvt_lapses"]["green_max"]: return GREEN
    if value <= ABSOLUTE["pvt_lapses"]["yellow_max"]: return YELLOW
    return RED


_ABSOLUTE_DISPATCH = {
    "perclos_pct": _absolute_perclos,
    "bpm": _absolute_bpm,
    "rmssd_ms": _absolute_rmssd,
    "pvt_mean_inv_rt": _absolute_pvt_inv_rt,
    "pvt_lapses": _absolute_pvt_lapses,
}


def _personal_lower_is_worse(value, mean, std):
    """Para métricas onde valor BAIXO indica fadiga (mean_inv_rt, rmssd, bpm normal-ish)."""
    if value < mean - 1.5 * std: return RED
    if value < mean - 0.75 * std: return YELLOW
    return GREEN


def _personal_higher_is_worse(value, mean, std):
    """Para métricas onde valor ALTO indica fadiga (lapses, perclos)."""
    if value > mean + 1.5 * std: return RED
    if value > mean + 0.75 * std: return YELLOW
    return GREEN


_PERSONAL_DISPATCH = {
    "perclos_pct": _personal_higher_is_worse,
    "bpm": None,  # sem direção monótona — manter cutoff absoluto sempre
    "rmssd_ms": _personal_lower_is_worse,
    "pvt_mean_inv_rt": _personal_lower_is_worse,
    "pvt_lapses": _personal_higher_is_worse,
}


def evaluate_metric(metric_name, value, profile=None):
    """
    Retorna 'green'/'yellow'/'red'/'invalid'/'experimental' para uma métrica.

    HRV (rmssd_ms) sempre prefixado como 'experimental' por estar pré-calibração.
    """
    if value is None:
        return INVALID

    if metric_name in EXPERIMENTAL_METRICS:
        # Computa o status mas marca como experimental — caller decide como exibir
        # e que não entra no status geral.
        # Pra rmssd_ms ainda calculamos pra mostrar amarelo/vermelho na tile.
        absolute_status = _ABSOLUTE_DISPATCH.get(metric_name, lambda v: GREEN)(value)
        return f"{EXPERIMENTAL}:{absolute_status}"

    if profile is not None:
        n = profile["baseline_sessions_count"].get(metric_name, 0)
        personal_fn = _PERSONAL_DISPATCH.get(metric_name)
        if n >= MIN_BASELINE_SESSIONS and personal_fn is not None:
            mean = profile["baseline_mean"][metric_name]
            std = profile["baseline_std"][metric_name]
            return personal_fn(value, mean, std)

    fn = _ABSOLUTE_DISPATCH.get(metric_name)
    if fn is None:
        return INVALID
    return fn(value)


def overall_status(per_axis_statuses):
    """
    Combina status por eixo em status geral.

    Regra conjuntiva: qualquer RED → RED; qualquer YELLOW → YELLOW; senão GREEN.
    Status 'experimental:*' e 'invalid' não contam.
    """
    contributing = [
        s for s in per_axis_statuses.values()
        if s in (GREEN, YELLOW, RED)
    ]
    if RED in contributing: return RED
    if YELLOW in contributing: return YELLOW
    if GREEN in contributing: return GREEN
    return INVALID

"""
HRV time-domain a partir de buffer IBI (intervalos pico-a-pico em ms).

Filtra outliers fisiológicos (40-150 bpm equivalente) antes de calcular.
Retorna None quando não há amostras suficientes.

Status experimental até calibração contra ground truth (Galaxy Watch / ECG).
"""
import numpy as np

# Faixa fisiológica de IBI: 40-150 bpm
IBI_MIN_MS = 400.0
IBI_MAX_MS = 1500.0

# Mínimo de batidas pra calcular HRV com confiabilidade time-domain
MIN_IBI_SAMPLES = 20


def _filter_outliers(ibi_ms_array):
    arr = np.asarray(ibi_ms_array, dtype=np.float64)
    return arr[(arr >= IBI_MIN_MS) & (arr <= IBI_MAX_MS)]


def compute_hrv(ibi_ms_array):
    """
    Recebe array de IBI em ms (janela rolling de 60s típica).

    Retorna None se sample count insuficiente após outlier filtering, OU
    {"rmssd_ms": float, "pnn50_pct": float, "sdnn_ms": float}.
    """
    if ibi_ms_array is None:
        return None
    clean = _filter_outliers(ibi_ms_array)
    if len(clean) < MIN_IBI_SAMPLES:
        return None
    diffs = np.diff(clean)
    rmssd = float(np.sqrt(np.mean(diffs ** 2)))
    pnn50 = float(np.mean(np.abs(diffs) > 50.0) * 100.0)
    sdnn = float(np.std(clean, ddof=1))
    return {"rmssd_ms": rmssd, "pnn50_pct": pnn50, "sdnn_ms": sdnn}

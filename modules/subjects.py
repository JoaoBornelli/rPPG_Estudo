"""
Gerenciamento de perfis de motorista e baselines pessoais via Welford.

Schema de data/profiles.json:
{
  "version": 1,
  "profiles": {
    "<nome>": {
      "created_at": ISO8601,
      "last_session_at": ISO8601 | null,
      "total_sessions": int,
      "rested_sessions": int,
      "baseline_sessions_count": {<metric>: int},
      "baseline_mean":  {<metric>: float},
      "baseline_std":   {<metric>: float},
      "baseline_M2":    {<metric>: float}
    }
  }
}
"""
import json
import os
from datetime import datetime
from math import sqrt

PROFILES_PATH = os.path.join("data", "profiles.json")
SCHEMA_VERSION = 1

# Métricas que entram no baseline pessoal
BASELINE_METRICS = [
    "perclos_pct",
    "bpm",
    "rmssd_ms",
    "pvt_mean_inv_rt",
    "pvt_lapses",
]


def _empty_profile():
    now = datetime.now().isoformat(timespec="seconds")
    return {
        "created_at": now,
        "last_session_at": None,
        "total_sessions": 0,
        "rested_sessions": 0,
        "baseline_sessions_count": {m: 0 for m in BASELINE_METRICS},
        "baseline_mean": {m: 0.0 for m in BASELINE_METRICS},
        "baseline_std": {m: 0.0 for m in BASELINE_METRICS},
        "baseline_M2": {m: 0.0 for m in BASELINE_METRICS},
    }


def _load_all(path=PROFILES_PATH):
    if not os.path.exists(path):
        return {"version": SCHEMA_VERSION, "profiles": {}}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if data.get("version") != SCHEMA_VERSION:
        raise RuntimeError(
            f"profiles.json versão {data.get('version')} incompatível com esperado {SCHEMA_VERSION}"
        )
    return data


def _save_all_atomic(data, path=PROFILES_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def list_profiles(path=PROFILES_PATH):
    """Retorna lista de nomes ordenada por last_session_at desc (None por último)."""
    data = _load_all(path)
    items = list(data["profiles"].items())
    items.sort(
        key=lambda kv: kv[1].get("last_session_at") or "",
        reverse=True,
    )
    return [name for name, _ in items]


def load_profile(name, path=PROFILES_PATH):
    data = _load_all(path)
    return data["profiles"].get(name)


def create_profile(name, path=PROFILES_PATH):
    """Cria perfil novo. Levanta ValueError se já existe."""
    data = _load_all(path)
    if name in data["profiles"]:
        raise ValueError(f"perfil '{name}' já existe")
    data["profiles"][name] = _empty_profile()
    _save_all_atomic(data, path)
    return data["profiles"][name]


def _welford_update(profile, metric, value):
    n = profile["baseline_sessions_count"][metric] + 1
    prev_mean = profile["baseline_mean"][metric]
    new_mean = prev_mean + (value - prev_mean) / n
    prev_M2 = profile["baseline_M2"][metric]
    new_M2 = prev_M2 + (value - prev_mean) * (value - new_mean)
    profile["baseline_sessions_count"][metric] = n
    profile["baseline_mean"][metric] = new_mean
    profile["baseline_M2"][metric] = new_M2
    profile["baseline_std"][metric] = sqrt(new_M2 / max(1, n - 1))


def save_session(name, metrics, marked_rested, path=PROFILES_PATH):
    """
    Atualiza profile no profiles.json.

    `metrics` é um dict {metric_name: value_or_None} contendo as chaves
    BASELINE_METRICS (mais opcionais). Se marked_rested=True e o valor
    da métrica não é None, atualiza baseline via Welford.
    """
    data = _load_all(path)
    if name not in data["profiles"]:
        raise ValueError(f"perfil '{name}' não existe")
    profile = data["profiles"][name]
    profile["last_session_at"] = datetime.now().isoformat(timespec="seconds")
    profile["total_sessions"] += 1
    if marked_rested:
        profile["rested_sessions"] += 1
        for m in BASELINE_METRICS:
            v = metrics.get(m)
            if v is not None:
                _welford_update(profile, m, float(v))
    _save_all_atomic(data, path)
    return profile

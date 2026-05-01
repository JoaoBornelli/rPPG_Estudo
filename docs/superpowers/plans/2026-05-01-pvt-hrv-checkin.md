# PVT-B + HRV Check-in Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Adicionar avaliação ativa PVT-B (3 min) + HRV time-domain extraído da onda CHROM ao Python desktop (`main.py`), produzindo painel multi-métrica de aptidão para direção (4 eixos: PERCLOS, rPPG, HRV, PVT) com baseline pessoal híbrido.

**Architecture:** Refatorar `main.py` monolítico em pacote `modules/` com módulos focados; rodar PVT em janela Pygame separada concorrente ao loop OpenCV; persistir perfis e baselines em JSON local com Welford incremental; CSV por sessão. Webapp mobile fora de escopo.

**Tech Stack:** Python 3.10+, OpenCV, MediaPipe Face Landmarker, NumPy, SciPy (`signal`), Pygame (PVT window).

**Reference:** Spec completa em `docs/superpowers/specs/2026-05-01-pvt-hrv-checkin-design.md`.

**Sem testes automatizados nessa fase** — verificação manual após cada task. Convenção de commits: Conventional Commits, sem Co-Authored-By.

---

## Task 1: Setup de dependências e estrutura de diretórios

**Files:**
- Modify: `requirements.txt`
- Modify: `.gitignore`
- Create: `modules/__init__.py`
- Create: `data/.gitkeep`
- Create: `data/sessions/.gitkeep`

- [ ] **Step 1: Adicionar dependências em `requirements.txt`**

Substituir conteúdo atual por:

```
numpy>=1.26,<2.0
opencv-python>=4.8,<5.0
mediapipe>=0.10.14,<0.11
scipy>=1.11,<2.0
pygame>=2.5,<3.0
matplotlib>=3.7,<4.0
```

(matplotlib já é usado implicitamente no relatório atual; tornar explícito.)

- [ ] **Step 2: Adicionar entradas em `.gitignore`**

Acrescentar ao final:

```
/data
```

- [ ] **Step 3: Criar diretórios e `__init__.py`**

```bash
mkdir -p modules data/sessions
touch modules/__init__.py data/.gitkeep data/sessions/.gitkeep
```

- [ ] **Step 4: Instalar deps**

Run: `pip install -r requirements.txt`
Expected: instala scipy e pygame sem erro. Versões já presentes (numpy/opencv/mediapipe) já satisfeitas.

- [ ] **Step 5: Smoke test pygame**

Run: `python -c "import pygame; pygame.init(); print(pygame.get_init())"`
Expected: imprime `(6, 0)` (módulos inicializados, 0 falhas) ou similar — sem traceback.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt .gitignore modules/__init__.py data/.gitkeep data/sessions/.gitkeep
git commit -m "chore: scaffold modules/ data/ and add scipy+pygame deps"
```

---

## Task 2: `modules/subjects.py` — perfis e baselines

**Files:**
- Create: `modules/subjects.py`

- [ ] **Step 1: Criar `modules/subjects.py`**

```python
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
```

- [ ] **Step 2: Verificação manual**

Run:
```bash
python -c "
from modules.subjects import create_profile, save_session, load_profile, list_profiles
import os
if os.path.exists('data/profiles.json'): os.remove('data/profiles.json')
create_profile('Teste')
save_session('Teste', {'perclos_pct': 4.0, 'bpm': 70, 'rmssd_ms': 40, 'pvt_mean_inv_rt': 3.5, 'pvt_lapses': 0}, marked_rested=True)
save_session('Teste', {'perclos_pct': 5.0, 'bpm': 72, 'rmssd_ms': 38, 'pvt_mean_inv_rt': 3.4, 'pvt_lapses': 1}, marked_rested=True)
p = load_profile('Teste')
print('mean perclos:', p['baseline_mean']['perclos_pct'])
print('std perclos:', p['baseline_std']['perclos_pct'])
print('total:', p['total_sessions'], 'rested:', p['rested_sessions'])
print('list:', list_profiles())
"
```
Expected: `mean perclos: 4.5`, `std perclos: ~0.707`, `total: 2 rested: 2`, `list: ['Teste']`.

- [ ] **Step 3: Limpeza do artefato de teste**

```bash
rm -f data/profiles.json
```

- [ ] **Step 4: Commit**

```bash
git add modules/subjects.py
git commit -m "feat(subjects): profile CRUD with Welford rolling baseline"
```

---

## Task 3: `modules/thresholds.py` — avaliação híbrida absoluto/pessoal

**Files:**
- Create: `modules/thresholds.py`

- [ ] **Step 1: Criar `modules/thresholds.py`**

```python
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
```

- [ ] **Step 2: Verificação manual**

Run:
```bash
python -c "
from modules.thresholds import evaluate_metric, overall_status
# Sem perfil — cutoff absoluto
print('perclos 5%:', evaluate_metric('perclos_pct', 5.0))      # green
print('perclos 12%:', evaluate_metric('perclos_pct', 12.0))    # yellow
print('perclos 20%:', evaluate_metric('perclos_pct', 20.0))    # red
print('rmssd 40ms:', evaluate_metric('rmssd_ms', 40.0))        # experimental:green
print('pvt 1/RT 2.0:', evaluate_metric('pvt_mean_inv_rt', 2.0)) # red
print('pvt lapses 5:', evaluate_metric('pvt_lapses', 5))        # red
# Status geral
print('overall:', overall_status({'perclos': 'green', 'rppg': 'green', 'pvt': 'red'}))  # red
print('overall mixed:', overall_status({'perclos': 'green', 'rppg': 'yellow', 'pvt': 'green'}))  # yellow
"
```
Expected: cada linha imprime o status entre parênteses no comentário acima.

- [ ] **Step 3: Commit**

```bash
git add modules/thresholds.py
git commit -m "feat(thresholds): hybrid absolute/personal axis evaluation"
```

---

## Task 4: `modules/hrv.py` — RMSSD/pNN50/SDNN do buffer IBI

**Files:**
- Create: `modules/hrv.py`

- [ ] **Step 1: Criar `modules/hrv.py`**

```python
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
```

- [ ] **Step 2: Verificação manual**

Run:
```bash
python -c "
from modules.hrv import compute_hrv
import numpy as np
# IBI sintético: 60bpm constante (1000ms) com ruído 50ms — RMSSD esperado ~50ms
np.random.seed(0)
ibi = 1000 + np.random.normal(0, 50, 60)
result = compute_hrv(ibi)
print('rmssd_ms:', round(result['rmssd_ms'], 1))   # ~70ms (sqrt(2)*50 por causa do diff)
print('pnn50_pct:', round(result['pnn50_pct'], 1)) # ~60%
print('sdnn_ms:', round(result['sdnn_ms'], 1))     # ~50ms
# Janela curta — None
print('curto:', compute_hrv([1000, 1010, 990]))    # None
# Outliers — filtra
ibi_dirty = [1000, 1010, 200, 5000] + [1000]*30
print('com outliers:', compute_hrv(ibi_dirty) is not None)  # True (filtra os outliers)
"
```
Expected: rmssd ~70ms, pnn50 ~60%, sdnn ~50ms, curto: None, com outliers: True.

- [ ] **Step 3: Commit**

```bash
git add modules/hrv.py
git commit -m "feat(hrv): time-domain RMSSD/pNN50/SDNN from IBI buffer"
```

---

## Task 5: `modules/capture.py` — extração de captura + landmarks + draw

**Files:**
- Create: `modules/capture.py`

- [ ] **Step 1: Criar `modules/capture.py`**

Conteúdo extrai linhas 364-372 (init), 323-358 (draw), e provê uma classe `Capture` que encapsula o loop de leitura. Funções utilitárias `get_landmark_px` e `landmark_points` ficam aqui também.

```python
"""
Captura de webcam + MediaPipe Face Landmarker.

Encapsula:
- VideoCapture loop (next_frame → (rgb, landmarks, timestamp_ms))
- desenho dos landmarks (mesh, iris)
- helpers de landmark→pixel
"""
import cv2 as cv
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils, drawing_styles


def get_landmark_px(face_landmarks, idx, w, h):
    lm = face_landmarks[idx]
    return np.array([lm.x * w, lm.y * h])


def landmark_points(face_landmarks, idx_list, w, h):
    pts = []
    for idx in idx_list:
        lm = face_landmarks[idx]
        x = int(np.clip(lm.x * w, 0, w - 1))
        y = int(np.clip(lm.y * h, 0, h - 1))
        pts.append([x, y])
    return np.array(pts, dtype=np.int32)


def draw_landmarks_on_image(rgb_image, detection_result):
    annotated_image = np.copy(rgb_image)
    if not detection_result.face_landmarks:
        return annotated_image

    for face_landmarks in detection_result.face_landmarks:
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style(),
        )
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style(),
        )
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style(),
        )
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style(),
        )
    return annotated_image


FRAME_DURATION_MS = 33  # 30 fps


class Capture:
    def __init__(self, model_path="face_landmarker.task", source=0):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
        self.cap = cv.VideoCapture(source)
        self.timestamp_ms = 0

    def next_frame(self):
        """
        Retorna (rgb_frame, detection_result, timestamp_ms) ou None se câmera falhar.
        Avança timestamp_ms em FRAME_DURATION_MS.
        """
        ret, frame_bgr = self.cap.read()
        if not ret:
            return None
        rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        detection = self.detector.detect_for_video(mp_image, self.timestamp_ms)
        ts = self.timestamp_ms
        self.timestamp_ms += FRAME_DURATION_MS
        return rgb, detection, ts

    def release(self):
        self.cap.release()
```

- [ ] **Step 2: Verificação manual**

Run:
```bash
python -c "
from modules.capture import Capture
cap = Capture()
result = cap.next_frame()
if result is None:
    print('FAIL: câmera não retornou frame')
else:
    rgb, det, ts = result
    print('frame shape:', rgb.shape, 'ts:', ts, 'face_detected:', bool(det.face_landmarks))
cap.release()
"
```
Expected: imprime shape `(H, W, 3)`, ts=0, face_detected=True (se o usuário estiver na frente da câmera).

- [ ] **Step 3: Commit**

```bash
git add modules/capture.py
git commit -m "feat(capture): extract MediaPipe + VideoCapture into module"
```

---

## Task 6: `modules/perclos.py` — EAR, blinks, PERCLOS janela 180s

**Files:**
- Create: `modules/perclos.py`

- [ ] **Step 1: Criar `modules/perclos.py`**

Encapsula a lógica de calibração + monitoramento PERCLOS (linhas 16-40, 73-93, 503-619 do main.py atual). Exposta como classe `PerclosTracker`.

```python
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
```

- [ ] **Step 2: Verificação manual**

Run:
```bash
python -c "
from modules.perclos import PerclosTracker
t = PerclosTracker()
# Calibração simulada
for _ in range(50): t.feed_calibration_open(0.30)
t.finish_calibration_open()
for i in range(50): t.feed_calibration_closed(0.10, i*100)
t.finish_calibration_closed(0)
print('open:', t.ear_open, 'closed:', t.ear_closed, 'thresh:', t.perclos_threshold)
# Monitoramento simulado
state = t.update(0.30, 0.30, 1.0)
print('estado descansado pct_r:', state['pct_r'], 'perclos:', state['perclos_pct'])
state = t.update(0.10, 0.10, 2.0)
print('olhos fechados pct_r:', state['pct_r'], 'alert:', state['alert_r'])
"
```
Expected: open ~0.30, closed ~0.10, thresh ~0.14; pct_r ~100 e ~0; alerta ativa quando fechado.

- [ ] **Step 3: Commit**

```bash
git add modules/perclos.py
git commit -m "feat(perclos): extract EAR + PERCLOS tracker into module"
```

---

## Task 7: `modules/rppg.py` — CHROM, BPM/RPM, e exposição de IBI buffer

**Files:**
- Create: `modules/rppg.py`

- [ ] **Step 1: Criar `modules/rppg.py`**

Extrai linhas 42-62 (constantes), 99-320 (CHROM/Welch/HR/RR) do main.py, e ADICIONA `get_ibi_buffer()` para alimentar HRV.

```python
"""
rPPG via CHROM (de Haan & Jeanne 2013) sobre ROI testa+bochechas.

Pipeline:
  feed_frame(rgb_frame, landmarks, timestamp_ms)
    → acumula R/G/B médios em buffers temporais
    → recompute() periódico (a cada 1s):
        - CHROM signal
        - bandpass (FFT ideal — TODO: Butterworth zero-phase)
        - Welch PSD → BPM (banda 0.8-3.2 Hz)
        - filtro EMA por SNR
        - peaks no domínio do tempo → IBI buffer (NOVO, para HRV)
"""
import cv2 as cv
import numpy as np
from scipy.signal import find_peaks

from modules.capture import landmark_points

# ROI landmarks (testa + bochechas bilaterais)
FOREHEAD_IDX = [54, 10, 67, 103, 109, 338, 297, 332, 284]
LEFT_CHEEK_IDX = [117, 118, 50, 205, 187, 147, 213, 192]
RIGHT_CHEEK_IDX = [346, 347, 280, 425, 411, 376, 433, 416]

HEART_WINDOW_SEC = 12.0
MIN_HEART_WINDOW_SEC = 6.0
HEART_BAND_HZ = (0.8, 3.2)
WELCH_SEG_SEC_HEART = 5.0
WELCH_OVERLAP_HEART = 0.5
ALPHA_WINDOW_SEC = 1.6
MAX_RPPG_BUFFER_SEC = 65.0  # estende para 65s pra acomodar janela HRV de 60s
EMA_ALPHA_HEART = 0.15
SNR_MIN_DB = 2.0

RESP_WINDOW_SEC = 30.0
MIN_RESP_WINDOW_SEC = 12.0
RESP_BAND_HZ = (0.1, 0.5)
WELCH_SEG_SEC_RESP = 20.0
WELCH_OVERLAP_RESP = 0.75
EMA_ALPHA_RESP = 0.10

# IBI / HRV
IBI_WINDOW_SEC = 60.0
PEAK_MIN_DISTANCE_SEC = 0.4   # equivalente a 150bpm
PEAK_PROMINENCE_REL = 0.5     # fração do desvio padrão do sinal


# ============== Construção da ROI e extração RGB ==============
def build_roi_mask(rgb_frame, face_landmarks):
    h, w = rgb_frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    forehead = landmark_points(face_landmarks, FOREHEAD_IDX, w, h)
    c_x = np.mean(forehead[:, 0])
    c_y = np.mean(forehead[:, 1])
    forehead = np.stack(
        [c_x + (forehead[:, 0] - c_x) * 1.12,
         c_y + (forehead[:, 1] - c_y) * 1.18],
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


def extract_combined_roi(rgb_frame, face_landmarks):
    mask = build_roi_mask(rgb_frame, face_landmarks)
    px = rgb_frame[mask == 255]
    if px.size == 0:
        return None
    return np.array([np.mean(px[:, 0]), np.mean(px[:, 1]), np.mean(px[:, 2])])


# ============== DSP ==============
def chrom_signal(r, g, b, fs):
    n = len(r)
    alpha_win = max(4, round(ALPHA_WINDOW_SEC * fs))
    signal = np.zeros(n)
    for start in range(0, n, alpha_win):
        end = min(start + alpha_win, n)
        if end - start < 2:
            break
        mu_r = max(np.mean(r[start:end]), 1e-6)
        mu_g = max(np.mean(g[start:end]), 1e-6)
        mu_b = max(np.mean(b[start:end]), 1e-6)
        rn = r[start:end] / mu_r - 1
        gn = g[start:end] / mu_g - 1
        bn = b[start:end] / mu_b - 1
        xs = 3 * rn - 2 * gn
        ys = 1.5 * rn + gn - 1.5 * bn
        alpha = np.std(xs) / (np.std(ys) + 1e-8)
        signal[start:end] = xs - alpha * ys
    s = np.std(signal)
    if s < 1e-8:
        return np.zeros(n)
    return (signal - np.mean(signal)) / s


def detrend_linear(signal):
    n = len(signal)
    if n < 3:
        return signal
    x = np.arange(n)
    coeffs = np.polyfit(x, signal, 1)
    return signal - np.polyval(coeffs, x)


def bandpass_fft(signal, fs, low_hz, high_hz):
    n = len(signal)
    if n < 4:
        return signal
    spec = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    spec[~mask] = 0
    return np.fft.irfft(spec, n=n)


def welch_psd(signal, fs, seg_sec, overlap):
    n = len(signal)
    if n < 16:
        return None
    nperseg = max(32, min(round(seg_sec * fs), n))
    if n < nperseg:
        return None
    step = max(1, round(nperseg * (1 - overlap)))
    win = 0.5 * (1 - np.cos(2 * np.pi * np.arange(nperseg) / (nperseg - 1)))
    win_pow = np.sum(win ** 2) + 1e-12
    freqs = np.fft.rfftfreq(nperseg, d=1.0 / fs)
    acc = np.zeros(len(freqs))
    count = 0
    for start in range(0, n - nperseg + 1, step):
        seg = signal[start:start + nperseg].copy()
        seg -= np.mean(seg)
        seg *= win
        spec = np.fft.rfft(seg)
        acc += (np.abs(spec) ** 2) / win_pow
        count += 1
    if count == 0:
        return None
    return freqs, acc / count


def estimate_rate(welch_result, band_hz):
    if welch_result is None:
        return None
    freqs, psd = welch_result
    low, high = band_hz
    mask = (freqs >= low) & (freqs <= high)
    if not np.any(mask):
        return None
    band_psd = psd[mask]
    band_freqs = freqs[mask]
    peak_idx = np.argmax(band_psd)
    peak_hz = band_freqs[peak_idx]
    peak_pow = band_psd[peak_idx]
    total_pow = np.sum(band_psd)
    noise_pow = (total_pow - peak_pow) / max(len(band_psd) - 1, 1)
    snr_db = 10 * np.log10((peak_pow + 1e-12) / (noise_pow + 1e-12))
    return {"bpm": peak_hz * 60, "snr_db": snr_db, "peak_hz": peak_hz}


# ============== Tracker ==============
class RPPGTracker:
    def __init__(self):
        self.r_buf = []
        self.g_buf = []
        self.b_buf = []
        self.ts_buf = []  # ms
        self.smooth_bpm = None
        self.last_heart = None
        self.smooth_rpm = None
        self.last_resp = None
        self.last_compute_ms = 0
        self.compute_interval_ms = 1000
        self._last_signal = None  # último sinal CHROM filtrado (pra extração de IBI)
        self._last_signal_fs = None

    def feed_frame(self, rgb_frame, face_landmarks, timestamp_ms):
        roi = extract_combined_roi(rgb_frame, face_landmarks)
        if roi is None:
            return
        self.r_buf.append(roi[0])
        self.g_buf.append(roi[1])
        self.b_buf.append(roi[2])
        self.ts_buf.append(timestamp_ms)
        # Trim
        while (len(self.ts_buf) > 1 and
               (self.ts_buf[-1] - self.ts_buf[0]) > MAX_RPPG_BUFFER_SEC * 1000):
            self.r_buf.pop(0); self.g_buf.pop(0)
            self.b_buf.pop(0); self.ts_buf.pop(0)

    def maybe_recompute(self, timestamp_ms):
        """Recompute BPM/RPM se passou o intervalo. Retorna True se recomputou."""
        if timestamp_ms - self.last_compute_ms <= self.compute_interval_ms:
            return False
        self.last_compute_ms = timestamp_ms

        hr_signal_fs = self._compute_heart()
        if hr_signal_fs is not None:
            self._last_signal, self._last_signal_fs = hr_signal_fs

        self._compute_resp()
        return True

    def _window_buffers(self, window_sec, min_window_sec):
        if len(self.ts_buf) < 2:
            return None
        dur_sec = (self.ts_buf[-1] - self.ts_buf[0]) / 1000.0
        if dur_sec < min_window_sec:
            return None
        cutoff_ms = self.ts_buf[-1] - window_sec * 1000
        start_idx = 0
        while start_idx < len(self.ts_buf) - 1 and self.ts_buf[start_idx] < cutoff_ms:
            start_idx += 1
        r = np.array(self.r_buf[start_idx:], dtype=np.float64)
        g = np.array(self.g_buf[start_idx:], dtype=np.float64)
        b = np.array(self.b_buf[start_idx:], dtype=np.float64)
        ts = self.ts_buf[start_idx:]
        if len(ts) < 2:
            return None
        diffs = np.diff(ts) / 1000.0
        avg_dt = np.mean(diffs)
        if avg_dt <= 0:
            return None
        fs = 1.0 / avg_dt
        if fs < 1:
            return None
        return r, g, b, fs, ts

    def _compute_heart(self):
        win = self._window_buffers(HEART_WINDOW_SEC, MIN_HEART_WINDOW_SEC)
        if win is None:
            return None
        r, g, b, fs, _ts = win
        sig = chrom_signal(r, g, b, fs)
        sig = detrend_linear(sig)
        sig = bandpass_fft(sig, fs, HEART_BAND_HZ[0], HEART_BAND_HZ[1])
        s = np.std(sig)
        if s > 1e-8:
            sig = (sig - np.mean(sig)) / s
        psd_result = welch_psd(sig, fs, WELCH_SEG_SEC_HEART, WELCH_OVERLAP_HEART)
        result = estimate_rate(psd_result, HEART_BAND_HZ)
        if result is not None:
            self.last_heart = result
            if result["snr_db"] >= SNR_MIN_DB:
                if self.smooth_bpm is None:
                    self.smooth_bpm = result["bpm"]
                else:
                    self.smooth_bpm = (
                        EMA_ALPHA_HEART * result["bpm"]
                        + (1 - EMA_ALPHA_HEART) * self.smooth_bpm
                    )
        return sig, fs

    def _compute_resp(self):
        win = self._window_buffers(RESP_WINDOW_SEC, MIN_RESP_WINDOW_SEC)
        if win is None:
            return
        _r, g, _b, fs, _ts = win
        sig = g.copy()
        s = np.std(sig)
        if s < 1e-8:
            return
        sig = (sig - np.mean(sig)) / s
        sig = detrend_linear(sig)
        sig = bandpass_fft(sig, fs, RESP_BAND_HZ[0], RESP_BAND_HZ[1])
        s = np.std(sig)
        if s > 1e-8:
            sig = (sig - np.mean(sig)) / s
        psd_result = welch_psd(sig, fs, WELCH_SEG_SEC_RESP, WELCH_OVERLAP_RESP)
        result = estimate_rate(psd_result, RESP_BAND_HZ)
        if result is not None:
            self.last_resp = result
            if result["snr_db"] >= -5.0:
                rpm = result["bpm"]
                if self.smooth_rpm is None:
                    self.smooth_rpm = rpm
                else:
                    self.smooth_rpm = (
                        EMA_ALPHA_RESP * rpm
                        + (1 - EMA_ALPHA_RESP) * self.smooth_rpm
                    )

    def get_ibi_buffer(self, window_sec=IBI_WINDOW_SEC):
        """
        Detecta picos no sinal CHROM filtrado mais recente e retorna
        array de IBI em ms. Retorna None se não houver sinal disponível
        ou número de picos insuficiente.

        STATUS: experimental — precisão sub-frame não implementada (TODO:
        parabolic interpolation Gasior 2004). Uso atual só para protótipo
        de HRV até validação contra ground truth.
        """
        # Recomputar com janela específica de IBI (60s) — pode diferir da janela cardíaca (12s)
        win = self._window_buffers(window_sec, min_window_sec=20.0)
        if win is None:
            return None
        r, g, b, fs, ts = win
        sig = chrom_signal(r, g, b, fs)
        sig = detrend_linear(sig)
        sig = bandpass_fft(sig, fs, HEART_BAND_HZ[0], HEART_BAND_HZ[1])
        s = np.std(sig)
        if s < 1e-8:
            return None
        sig = (sig - np.mean(sig)) / s

        min_dist_samples = max(1, int(PEAK_MIN_DISTANCE_SEC * fs))
        peaks, _props = find_peaks(
            sig,
            distance=min_dist_samples,
            prominence=PEAK_PROMINENCE_REL,
        )
        if len(peaks) < 3:
            return None
        peak_times_ms = np.array(ts)[peaks]
        ibi_ms = np.diff(peak_times_ms.astype(np.float64))
        return ibi_ms

    def buffer_seconds(self):
        if len(self.ts_buf) < 2:
            return 0.0
        return (self.ts_buf[-1] - self.ts_buf[0]) / 1000.0
```

- [ ] **Step 2: Verificação manual (sinal sintético)**

Run:
```bash
python -c "
import numpy as np
from modules.rppg import chrom_signal, detrend_linear, bandpass_fft, welch_psd, estimate_rate, HEART_BAND_HZ
# Gera sinal sintético: 70bpm = 1.167Hz
fs = 30.0
n = int(15 * fs)
t = np.arange(n) / fs
r = 0.5 + 0.05 * np.sin(2*np.pi*1.167*t) + 0.01*np.random.randn(n)
g = 0.5 + 0.10 * np.sin(2*np.pi*1.167*t) + 0.01*np.random.randn(n)
b = 0.5 + 0.03 * np.sin(2*np.pi*1.167*t) + 0.01*np.random.randn(n)
sig = chrom_signal(r,g,b,fs)
sig = detrend_linear(sig)
sig = bandpass_fft(sig, fs, *HEART_BAND_HZ)
psd = welch_psd(sig, fs, 5.0, 0.5)
r = estimate_rate(psd, HEART_BAND_HZ)
print('BPM estimado:', round(r['bpm'], 1), 'SNR:', round(r['snr_db'], 1))
"
```
Expected: BPM ~70 (com tolerância de ±6 BPM por causa da resolução), SNR > 5dB.

- [ ] **Step 3: Commit**

```bash
git add modules/rppg.py
git commit -m "feat(rppg): extract CHROM/Welch tracker + add IBI buffer for HRV"
```

---

## Task 8: `modules/pvt.py` — janela Pygame com trials e RT

**Files:**
- Create: `modules/pvt.py`

- [ ] **Step 1: Criar `modules/pvt.py`**

```python
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
        and 100.0 <= t["rt_ms"] <= 30000.0
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
```

- [ ] **Step 2: Verificação manual (run reduzido)**

Run:
```bash
python -c "
from modules import pvt
# Reduz duração para 15s pra testar manualmente
trials = pvt.run(duration_sec=15.0)
metrics = pvt.compute_pvt_metrics(trials)
print('n_trials:', metrics['n_trials'])
print('mean_inv_rt:', round(metrics['mean_inv_rt'], 2) if metrics['mean_inv_rt'] else None)
print('lapses:', metrics['lapses'])
print('false_starts:', metrics['false_starts'])
"
```
Expected: janela Pygame abre, mostra instruções, depois roda ~15s de trials. Toque algumas vezes. Ao fim, imprime n_trials > 0 e mean_inv_rt razoável.

**Note:** se travar em macOS com aviso NSWindow off main thread, é problema de plataforma. Aqui chamamos `run()` do main thread então não deve dar — em integração futura (Task 10), garantir que `pvt.run()` seja chamado do main thread e a captura OpenCV vá pra thread.

- [ ] **Step 3: Commit**

```bash
git add modules/pvt.py
git commit -m "feat(pvt): pygame PVT-B with RT, lapses, false starts"
```

---

## Task 9: `modules/ui_panel.py` — painel multi-métrica + tela de seleção de perfil

**Files:**
- Create: `modules/ui_panel.py`

- [ ] **Step 1: Criar `modules/ui_panel.py`**

Esse módulo desenha:
1. tela de seleção de perfil (lista + "novo perfil")
2. painel de resultados (4 tiles + status geral) ao final do check-in

Ambos via OpenCV (sem janela extra). Interação por teclado.

```python
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
```

- [ ] **Step 2: Verificação manual**

Run:
```bash
python -c "
from modules import ui_panel
# Tela de seleção
choice = ui_panel.select_profile(['Joao', 'Maria'])
print('escolha:', choice)
# Painel de resultados sintético
metrics = {
    'perclos_pct': 4.5, 'blinks_per_min': 18,
    'bpm': 72, 'rpm': 14, 'snr_db': 6.3,
    'rmssd_ms': 32, 'pnn50_pct': 8.1,
    'pvt_n_trials': 36, 'pvt_mean_inv_rt': 3.42, 'pvt_lapses': 1, 'pvt_false_starts': 0,
}
evals = {'perclos': 'green', 'rppg': 'green', 'hrv': 'experimental:yellow', 'pvt': 'green'}
result = ui_panel.show_results(metrics, evals, 'green', 'Test')
print('action:', result)
"
```
Expected: tela de seleção abre, navegação funciona; painel mostra 4 tiles, R alterna descansado, S/ESC fecha.

- [ ] **Step 3: Commit**

```bash
git add modules/ui_panel.py
git commit -m "feat(ui_panel): profile select + multi-axis results panel"
```

---

## Task 10: Refatorar `main.py` em orquestrador

**Files:**
- Modify: `main.py` (substituição completa)

- [ ] **Step 1: Substituir `main.py` pelo orquestrador novo**

Cria fluxo: seleção de perfil → calibração PERCLOS (já no `perclos.py` — só conduz) → check-in 3 min com PVT em main thread + captura em thread auxiliar — wait. **Decisão crítica:** Pygame em macOS exige main thread; então **invertemos**: captura OpenCV em thread auxiliar, PVT no main thread. Isso é o oposto do spec (spec dizia PVT em thread separada), mas é a única forma estável em macOS.

Atualização ao spec: PVT roda no main thread (Pygame), captura roda em thread auxiliar. Documentar isso no README também.

```python
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

            ibi = rppg.get_ibi_buffer(window_sec=60.0)
            if ibi is not None:
                hrv = compute_hrv(ibi)
                if hrv is not None:
                    snapshot.update(hrv)
        cv.imshow("Check-in - Captura (passivo)",
                  cv.cvtColor(rgb, cv.COLOR_RGB2BGR))
        cv.waitKey(1)


def main():
    # 1. Seleção de perfil
    profiles = subjects.list_profiles()
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

    stop_event.set()
    passive_thread.join(timeout=2.0)
    cap.release()
    cv.destroyWindow("Check-in - Captura (passivo)")

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
```

- [ ] **Step 2: Backup do main.py original e substituir**

```bash
cp main.py main_legacy.py.bak
# (depois de verificar que tudo funciona, remover main_legacy.py.bak antes do commit)
```

Substituir `main.py` pelo conteúdo acima.

- [ ] **Step 3: Smoke run completo**

Run: `python main.py`

Expected fluxo:
1. Janela "Check-in - Selecao de perfil" abre.
2. Cria/escolhe perfil.
3. Janela "Check-in - Calibracao" abre. Faz 5s aberto + 5s fechado.
4. Janela "Check-in - Captura (passivo)" + janela Pygame "PVT-B" abrem.
5. Faz PVT por 3 min (ou cancela com fechamento da janela Pygame).
6. Janela "Check-in - Resultado" mostra os 4 tiles + status geral.
7. R alterna descansado, S salva, ESC descarta.
8. Após salvar: arquivo em `data/sessions/<TS>_<nome>.csv` e `data/profiles.json` atualizado.

Se algo travar (especialmente em macOS), checar console por warnings sobre NSWindow.

- [ ] **Step 4: Limpar backup se tudo ok**

```bash
rm main_legacy.py.bak
```

- [ ] **Step 5: Commit**

```bash
git add main.py
git commit -m "feat(main): orchestrator wiring profile→calibration→PVT+passive→panel"
```

---

## Task 11: Atualizar README com a nova feature

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Adicionar seção no `README.md`**

Inserir uma nova seção após a "Visão Geral do Pipeline (Python)" e antes de "Requisitos":

```markdown
## Check-in Multi-Métrica de Aptidão (PVT + HRV + PERCLOS + rPPG)

A partir de 2026-05, o `main.py` executa um **check-in de 3 minutos** que combina quatro eixos independentes de fadiga:

| Eixo | Tipo | Janela | Métrica primária |
|------|------|--------|------------------|
| PERCLOS | passivo | 180s | % de tempo com olhos fechados (P80) |
| rPPG | passivo | 12s | BPM (CHROM + Welch) |
| HRV ⚠ experimental | passivo | 60s | RMSSD time-domain extraído da onda CHROM |
| PVT-B | ativo | 3 min | Mean 1/RT (Basner & Dinges 2011) |

**Fluxo:**
1. Seleção de perfil (lista local em `data/profiles.json`).
2. Calibração PERCLOS (5s olhos abertos + 5s fechados).
3. Check-in 3 min: PVT-B em janela Pygame (main thread) + captura OpenCV + PERCLOS/rPPG/HRV em thread auxiliar.
4. Painel de resultados com 4 tiles, status verde/amarelo/vermelho por eixo e status geral por regra conjuntiva.
5. Opção de marcar sessão como "descansado" antes de salvar — só nesse caso a sessão alimenta o **baseline pessoal** (Welford rolling).
6. Cada sessão é gravada em `data/sessions/YYYY-MM-DD_HHMM_<nome>.csv`.

**Avaliação híbrida absoluto/pessoal:** as primeiras 5 sessões usam cutoffs da literatura. Após 5 sessões marcadas como "descansado", os thresholds de cada eixo passam a ser pessoais (vermelho < mean − 1.5·std, amarelo < mean − 0.75·std).

**Por que essa combinação:** PERCLOS captura o eixo ocular passivo, rPPG o cardio, HRV o autonômico (parassimpático), PVT-B o cortical (atenção sustentada — gold standard de fadiga aguda, Basner 2011). UFOV foi avaliado e descartado por falta de validação em smartphone e baixo poder discriminativo em motoristas jovens.

**Status experimental do HRV:** RMSSD/pNN50 extraídos da onda CHROM via detecção de picos `scipy.signal.find_peaks`. A 30 fps a resolução temporal de IBI é ~33ms, o que limita a precisão de RMSSD. Antes de tratar HRV como medição publicável, é necessário (i) trocar o bandpass FFT ideal atual por Butterworth zero-phase, (ii) adicionar parabolic peak interpolation (Gasior 2004), (iii) avaliar bump para 60 fps e (iv) validar contra ground truth (Galaxy Watch ou Polar H10) com Bland-Altman. Até lá, HRV é exibido com tag "experimental" e **não conta** para o status geral.

**Spec completo:** `docs/superpowers/specs/2026-05-01-pvt-hrv-checkin-design.md`.

**Plataforma:** Python desktop nesta fase. Port para a webapp mobile (`webapp/`) é fase 2.
```

- [ ] **Step 2: Adicionar referências no fim**

Acrescentar ao bloco de Referências:

```
[16] D. F. Dinges and J. W. Powell, "Microcomputer analyses of performance on a portable,
     simple visual RT task during sustained operations," Behav. Res. Methods Instrum.
     Comput., vol. 17, no. 6, pp. 652-655, 1985.

[17] M. Basner and D. F. Dinges, "Maximizing sensitivity of the PVT to sleep loss,"
     Sleep, vol. 34, no. 5, pp. 581-591, 2011. DOI: 10.1093/sleep/34.5.581

[18] M. Basner, D. Mollicone, and D. F. Dinges, "Validity and sensitivity of a brief
     psychomotor vigilance test (PVT-B)," Acta Astronautica, vol. 69, no. 11-12,
     pp. 949-959, 2011. DOI: 10.1016/j.actaastro.2011.07.015

[19] P. F. Brunet et al., "sleep-2-Peak: a smartphone application that allows the
     assessment of vigilance with the PVT," Behav. Res. Methods, vol. 49,
     pp. 1907-1916, 2017. DOI: 10.3758/s13428-016-0802-5

[20] J. Vicente et al., "Drowsiness detection using heart rate variability,"
     Med. Biol. Eng. Comput., vol. 54, pp. 927-937, 2016.

[21] M. Munoz et al., "Validity of (ultra-)short recordings for HRV measurements,"
     PLoS ONE, vol. 10, no. 9, p. e0138921, 2015.

[22] Task Force of ESC and NASPE, "Heart rate variability: Standards of measurement,
     physiological interpretation and clinical use," Circulation, vol. 93, no. 5,
     pp. 1043-1065, 1996.

[23] D. Nunan, G. R. H. Sandercock, and D. A. Brodie, "A quantitative systematic
     review of normal values for short-term heart rate variability in healthy adults,"
     PACE, vol. 33, pp. 1407-1417, 2010.
```

- [ ] **Step 3: Verificação manual**

Run: `head -100 README.md` — ver se a nova seção aparece corretamente.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs(readme): document multi-axis check-in feature (PVT + HRV)"
```

---

## Self-Review Checklist (após executar todas as tasks)

- [ ] `python main.py` roda sem traceback do início ao fim em sessão descansada.
- [ ] `data/profiles.json` é criado e atualizado corretamente.
- [ ] `data/sessions/*.csv` é gravado com todas as colunas declaradas.
- [ ] Os 4 tiles aparecem com cores corretas no painel.
- [ ] HRV exibido com tag experimental e não move o status geral.
- [ ] Após ≥5 sessões "descansado", o evaluator transita pra baseline pessoal (verificar com `python -c "from modules.subjects import load_profile; print(load_profile('<nome>'))"`).
- [ ] PVT em janela Pygame é responsivo (toques registram RT corretamente). Em macOS, sem warnings críticos sobre NSWindow.

---

## Notas finais e gotchas conhecidos

1. **macOS + Pygame off-main-thread:** o plano coloca PVT no main thread por essa razão. Se em outra plataforma quiser inverter (PVT em thread, OpenCV no main), o spec original previa isso e não muda a interface dos módulos.

2. **Pygame e múltiplos displays / Retina:** `WINDOW_SIZE = (640, 480)` é fixo. Em telas Retina pode aparecer pequeno. Não implementar redimensionamento agora — YAGNI.

3. **OpenCV `waitKey(1)` em thread auxiliar:** funciona na maioria das plataformas, mas é sabido que em macOS o GUI loop quer estar no main thread. Se travar, mover `cv.imshow` da thread auxiliar pra um polling no main thread durante o PVT é uma alternativa (PVT pode chamar callback periódico). Não implementar antecipadamente.

4. **Calibração de HRV:** ver Task 4 e Task 7. Os caveats estão registrados no README. Próxima fase do projeto deveria ser exatamente essa validação Bland-Altman vs Galaxy Watch.

5. **`face_profiles.json` legado:** mantido intocado para `quick_rppg_experiment.py`. O check-in usa `data/profiles.json` (caminho diferente).

6. **Logs do main.py antigo:** o `logs/` (gitignored) gerado pelo flow anterior continua válido para invocações de `quick_rppg_experiment.py` ou similares; o novo flow grava em `data/sessions/`. Não há colisão.

7. **Reset manual de baseline:** não implementado v1. Se precisar (mudou de webcam/iluminação), editar `data/profiles.json` à mão zerando os campos `baseline_*` daquele perfil.

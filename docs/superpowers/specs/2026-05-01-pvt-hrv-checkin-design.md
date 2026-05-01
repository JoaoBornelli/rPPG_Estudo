# Check-in Multi-Métrica de Aptidão para Direção (PVT-B + HRV + PERCLOS + rPPG)

**Data:** 2026-05-01
**Autor:** João Bornelli (com brainstorming colaborativo)
**Status:** Design aprovado, aguardando plano de implementação
**Escopo:** Python desktop (`main.py`); webapp mobile fora de escopo nesta fase

---

## 1. Contexto e motivação

O projeto `rPPG_Estudo` hoje implementa em Python desktop (`main.py`) um pipeline de detecção de fadiga via PERCLOS (fechamento ocular, janela de 180s, alerta em PERCLOS≥15%) combinado com rPPG (CHROM, BPM/RPM com EMA + filtro por SNR). Existe também uma webapp mobile (Vercel) que roda apenas rPPG.

A motivação inicial foi avaliar se o teste **UFOV (Useful Field of View)** poderia ser incluído como avaliação ativa concorrente, aproveitando que o motorista olha para o celular durante um "check-in" pré-direção.

A pesquisa de literatura concluiu que:

- **UFOV não é apropriado**: zero validação peer-reviewed em smartphone (geometria mobile invalida normas clínicas), e baixo poder discriminativo em motoristas jovens/saudáveis (validação primária em idosos / declínio cognitivo crônico).
- **PVT-B (Psychomotor Vigilance Test, 3 min)** é o gold standard de fadiga aguda, com versões smartphone validadas (Brunet 2017 sleep-2-Peak; Kay 2016) e sensibilidade demonstrada a privação de sono em adultos jovens (Basner & Dinges 2011).
- **HRV time-domain (RMSSD/pNN50)** pode ser extraído da onda CHROM rPPG já existente, com janela de 60s suficiente (Munoz 2015), capturando dimensão autonômica (Vicente 2016) — incremental ao BPM/RPM já calculado.
- A combinação **PERCLOS + rPPG + HRV + PVT-B** cobre quatro eixos independentes da fadiga (ocular passivo, cardio, autonômico, cortical/atenção sustentada), todos sustentados pela literatura.

## 2. Decisões de design (recap das escolhas do brainstorm)

| # | Tópico | Escolha | Justificativa |
|---|---|---|---|
| 1 | Métodos | PVT-B 3min + HRV-RMSSD (descarta UFOV) | Ver pesquisa acima |
| 2 | Plataforma | Python desktop primeiro; webapp depois | `main.py` já tem PERCLOS+rPPG, é o lab natural |
| 3 | Tempo total do check-in | ~3 min ativo + passivo simultâneos | PVT-B é 3 min; passivo roda em paralelo |
| 4 | Output | Painel multi-métrica, 4 eixos com semáforo | Selander 2019 / Hird 2016: regra clínica > score fundido |
| 5 | Thresholds | Híbrido: literatura no dia 1, baseline pessoal após N≥5 sessões "descansado" | Threshold absoluto vira ruído em jovens; intra-sujeito é mais sensível |
| 6 | Identificação | Seleção manual de perfil em JSON local; sem face recognition | Bate com remoção de face recognition no commit 773b908 |
| 7 | PVT variant | PVT-B (3 min, Basner 2011) | Gold standard publicado |
| 8 | Janela do PVT | Pygame em thread separada | Timing sub-frame defensável; SDL backend |
| 9 | Arquitetura | Híbrido pragmático: extrai módulos só do que precisa de tratamento especial | Isola PVT/HRV sem refatorar o que já funciona |

## 3. Fluxo do check-in

```
0. APP STARTUP
   ├─ load profiles.json
   └─ tela inicial: lista de perfis + "novo perfil"
       └─ usuário seleciona → carrega baseline pessoal se existir,
          senão usa thresholds da literatura

1. CALIBRAÇÃO PERCLOS (~10s, já existe)
   └─ olhos abertos 5s → olhos fechados 5s

2. CHECK-IN (3 min total)
   ├─ Thread principal (OpenCV):
   │   ├─ captura @ 30fps, MediaPipe Face Landmarker
   │   ├─ PERCLOS (EAR, janela 180s)
   │   ├─ rPPG CHROM → BPM, RPM
   │   └─ HRV-RMSSD (IBI da CHROM, janela rolling 60s)
   │
   └─ Thread PVT (Pygame, janela separada):
       ├─ instrução 5s
       ├─ ~30-40 trials, ISI uniforme 1.0-4.0s (PVT-B padrão Basner 2011)
       ├─ registra RT por trial via time.perf_counter()
       └─ ~3 min total → sinaliza fim por threading.Event

3. PAINEL DE RESULTADOS
   ├─ 4 tiles: PERCLOS · rPPG · HRV · PVT
   ├─ status semáforo por eixo + status geral
   ├─ números crus
   └─ checkbox "marcar sessão como descansado" + botão "salvar"
       └─ CSV + atualização Welford de baseline em profiles.json
```

**Coordenação entre threads:** main thread inicia o PVT thread no começo da fase 2; PVT thread sinaliza término via `threading.Event`. Main thread agrega métricas quando o evento dispara. Nenhuma dependência de tempo absoluto entre eles.

**"Estado descansado":** o motorista marca via checkbox antes de salvar. Se marcado, métricas dessa sessão entram no baseline pessoal (Welford rolling). Senão, sessão fica no CSV mas não polui o baseline.

## 4. Arquitetura de arquivos

```
rPPG_Estudo/
├── main.py                    ← entry point + OpenCV main loop (orquestrador, ~8-10K)
├── modules/
│   ├── __init__.py
│   ├── capture.py             ← MediaPipe + frame loop (extraído do main.py)
│   ├── perclos.py             ← EAR + PERCLOS janela 180s (extraído)
│   ├── rppg.py                ← CHROM + Welch PSD → BPM/RPM (extraído)
│   ├── hrv.py                 ← IBI extraction + RMSSD/pNN50 (NOVO)
│   ├── pvt.py                 ← thread Pygame, trials, RT (NOVO)
│   ├── subjects.py            ← profiles.json + baseline pessoal (NOVO)
│   ├── thresholds.py          ← cutoffs literatura + lógica híbrida (NOVO)
│   └── ui_panel.py            ← painel 4 tiles + semáforo (NOVO)
│
├── data/                      ← gitignored
│   ├── profiles.json          ← perfis + baselines
│   └── sessions/
│       └── YYYY-MM-DD_HHMM_<nome>.csv
│
├── face_landmarker.task       ← (existente)
├── face_profiles.json         ← (existente, mantido pra compatibilidade)
├── quick_rppg_experiment.py   ← (intocado)
├── quick_fft_compare.py       ← (intocado)
├── README.md                  ← atualizar
└── webapp/                    ← (intocado nessa fase)
```

**Interfaces mínimas dos módulos:**

| Módulo | API pública |
|---|---|
| `capture.py` | `next_frame() → (frame, landmarks)` |
| `perclos.py` | `update(landmarks) → {ear, perclos, fatigue_alert}` |
| `rppg.py` | `update(frame, landmarks) → {bpm, rpm, snr}`; `get_ibi_buffer(window_sec=60) → np.ndarray` |
| `hrv.py` | `compute(ibi_ms_window) → {rmssd_ms, pnn50_pct, sdnn_ms}` ou `None` se janela insuficiente |
| `pvt.py` | `run(duration_sec=180) → list[trial]`; síncrono no PVT thread |
| `subjects.py` | `load_profile(name)`, `create_profile(name)`, `list_profiles()`, `save_session(name, metrics, marked_rested)` |
| `thresholds.py` | `evaluate(metrics, profile) → {axis: 'green'/'yellow'/'red'/'experimental'}` |
| `ui_panel.py` | `draw_panel(frame, evaluation, raw_metrics) → frame` |

**Princípio:** cada módulo tem uma responsabilidade clara, depende só do que precisa, e é fácil de raciocinar isoladamente. `main.py` deixa de ser monolítico de 29.5K e vira orquestrador (~8-10K).

## 5. PVT-B (detalhe)

### 5.1 Parâmetros (Basner & Dinges 2011, *Acta Astronautica*)

| Parâmetro | Valor |
|---|---|
| Duração total | 180s (3 min) |
| ISI | uniforme 1.0-4.0s |
| Estímulo | contador de ms subindo (000 → 999...), fonte grande monoespaçada |
| Resposta | toque/clique em qualquer lugar da janela Pygame |
| Lapse threshold | RT > 500ms |
| False start | toque < 100ms após estímulo, ou toque sem estímulo |
| Sleep attack | RT > 30s ou sem resposta |
| Trials esperados | ~30-40 (variável c/ ISI aleatório) |

### 5.2 Métricas

```python
def compute_pvt_metrics(trials):
    rts = [t["rt_ms"] for t in trials if 100 <= t["rt_ms"] <= 30000]
    lapses = sum(1 for rt in rts if rt > 500)
    mean_rt = mean(rts)
    mean_inv_rt = mean([1000.0 / rt for rt in rts])           # primária (Basner 2011)
    slowest_10 = sorted(rts, reverse=True)[:max(1, len(rts)//10)]
    slowest_10_inv_rt = mean([1000.0 / rt for rt in slowest_10])
    false_starts = count_false_starts(trials)
    return {
        "n_trials": len(trials),
        "mean_rt_ms": mean_rt,
        "mean_inv_rt": mean_inv_rt,
        "lapses": lapses,
        "slowest_10pct_inv_rt": slowest_10_inv_rt,
        "false_starts": false_starts,
    }
```

**Métrica primária:** `mean_inv_rt` (Basner 2011 mostrou maior effect size que `lapses` ou `mean_rt` em TSD/PSD).

### 5.3 Cutoffs absolutos (literatura, dia 1)

| Métrica | Verde | Amarelo | Vermelho |
|---|---|---|---|
| `mean_inv_rt` (resp/s) | ≥ 3.0 | 2.5–3.0 | < 2.5 |
| `lapses` (em 3 min) | 0–1 | 2–3 | ≥ 4 |
| `false_starts` | 0–2 | 3–5 | ≥ 6 |

### 5.4 Cutoffs pessoais (após N≥5 sessões "descansado")

```
mean_inv_rt:
  vermelho se atual < (baseline_mean - 1.5 * baseline_std)
  amarelo  se atual < (baseline_mean - 0.75 * baseline_std)
  verde    caso contrário

lapses:
  vermelho se atual > (baseline_mean_lapses + 2)
  amarelo  se atual > (baseline_mean_lapses + 1)
```

### 5.5 Notas de implementação

- **Timing:** `time.perf_counter()` (Python nano-precision) marca cada evento. Pygame renderiza, mas a fonte da verdade é `perf_counter()`. Latência de touch/click do OS entra como ruído inevitável (~5-15ms tipicamente em macOS/desktop) — aceito como floor.
- **Janela fullscreen sem decoração:** evita o motorista clicar em outra coisa por engano.
- **Sem feedback sonoro:** evita distração.

## 6. HRV pipeline

### 6.1 Fluxo

```
onda CHROM filtrada (já produzida por rppg.py; banda atual 0.8-3.2 Hz, fs=30Hz)
    ↓ (a banda e o tipo de filtro são alvo de calibração — ver 6.6)
detecção de picos (scipy.signal.find_peaks com prominence/distance/width)
    ↓
parabolic peak interpolation (Gasior 2004) — sub-frame
    ↓
IBI[i] = (t_pico[i] - t_pico[i-1]) * 1000  # ms
    ↓
filtro outliers (rejeitar IBI fora de [400ms, 1500ms])
    ↓
janela rolling de 60s
    ↓
RMSSD, pNN50, SDNN
```

### 6.2 Métricas (Task Force 1996)

```python
def compute_hrv(ibi_ms_window):
    if len(ibi_ms_window) < 20:
        return None
    diffs = np.diff(ibi_ms_window)
    rmssd = np.sqrt(np.mean(diffs**2))
    pnn50 = np.mean(np.abs(diffs) > 50.0) * 100
    sdnn = np.std(ibi_ms_window, ddof=1)
    return {"rmssd_ms": rmssd, "pnn50_pct": pnn50, "sdnn_ms": sdnn}
```

**Primária:** RMSSD (tônus parassimpático/vagal; cai com fadiga — Vicente 2016, Patel 2011). LF/HF fora de escopo (precisa ≥2 min e é mais sensível a artefato em rPPG; Iozzia 2016).

### 6.3 Cutoffs absolutos (sanity check inicial)

Literatura tem **enorme variância normativa** (Nunan 2010 meta-análise: RMSSD adulto saudável ~19-75ms). Cutoffs absolutos servem só de fallback antes do baseline pessoal.

| Métrica | Verde | Amarelo | Vermelho |
|---|---|---|---|
| RMSSD (ms) | ≥ 30 | 20–30 | < 20 |
| pNN50 (%) | ≥ 10 | 3–10 | < 3 |

### 6.4 Cutoffs pessoais

Mesma lógica do PVT (1.5 std → vermelho, 0.75 std → amarelo).

### 6.5 Caveats explícitos (a documentar no README + painel)

1. **PRV ≠ HRV verdadeiro.** rPPG mede pulse rate variability — diferente de RR-ECG por jitter de pulse transit time (Yu et al. 2021). Pra fadiga *relativa intra-sujeito*, suficiente. Pra valor absoluto comparável a wearable de pulso, não.
2. **SNR baixo polui IBI.** Se `rppg.py` reporta SNR < 2dB, descartar a janela HRV correspondente.
3. **Movimento destrói HRV.** Se PERCLOS detectar perda de face/landmarks por > 10% do tempo da janela 60s, marcar HRV inválido (status cinza no painel).

### 6.6 Calibração necessária antes de confiar em HRV

A 30 fps, a resolução temporal de IBI é ~33ms. RMSSD opera sobre `diff(IBI)`, ainda mais ruidoso. Antes de mostrar RMSSD ao usuário sem ressalva, é necessário:

1. **Trocar bandpass FFT ideal → Butterworth zero-phase** (`scipy.signal.sosfiltfilt`, ordem 4-6, banda 0.7-3.0 Hz). O filtro ideal atual introduz Gibbs ringing que distorce posição de picos. Já listado como TODO no README do projeto.
2. **Peak detection com `find_peaks` parametrizado:** `prominence`, `distance` (mínimo 0.4s = 150bpm máx), `width`. Tuning empírico contra ground truth.
3. **Parabolic peak interpolation** pra precisão sub-frame (Gasior 2004 [15] do README, aplicável também a peaks no tempo). Reduz erro de ~33ms para ~5-10ms.
4. **Avaliar bump de fps** (60 fps webcam → resolução IBI ~16ms; combinada com parabolic, ~3-5ms).
5. **Validação empírica obrigatória:** 5-10 sessões side-by-side com Galaxy Watch (PRV) ou Polar H10 (RR-ECG verdadeiro). Calcular bias + LoA Bland-Altman. Se erro > ~10ms RMSSD, não publicar a métrica.

**Status no painel até validação concluída:** HRV exibe valor com tag "EXPERIMENTAL — não calibrado vs ground truth" e **não conta para o status geral**.

### 6.7 Impacto em `rppg.py` existente

Mínimo. Acrescentar export do buffer de picos detectados (já é necessário internamente). Função pública nova: `get_ibi_buffer(window_sec=60) → np.ndarray`. Resto inalterado.

## 7. Painel multi-métrica e lógica de decisão

### 7.1 Layout

```
┌─────────────────────────────────────────────────────────┐
│  CHECK-IN — João Bornelli — 2026-05-01 14:32            │
├──────────────────────┬──────────────────────────────────┤
│  PERCLOS             │  rPPG                            │
│  ●●● VERDE           │  ●●● VERDE                       │
│  3.2% (180s window)  │  BPM: 72  RPM: 14                │
│  Blink rate: 18/min  │  SNR: 6.3 dB                     │
├──────────────────────┼──────────────────────────────────┤
│  HRV  ⚠ experimental │  PVT-B                           │
│  ●●● AMARELO         │  ●●● VERDE                       │
│  RMSSD: 24ms         │  Mean 1/RT: 3.42 resp/s          │
│  pNN50: 8.1%         │  Lapses: 1   False starts: 0     │
│                      │  Trials: 36                      │
├──────────────────────┴──────────────────────────────────┤
│  STATUS GERAL: AMARELO                                  │
│  (regra: ≥1 amarelo e 0 vermelhos)                      │
│                                                         │
│  [ marcar como sessão "descansado" ]  [ salvar ]        │
└─────────────────────────────────────────────────────────┘
```

### 7.2 Regra de decisão geral (conjuntiva)

```
qualquer eixo VERMELHO   → STATUS GERAL = VERMELHO
≥ 1 eixo AMARELO         → STATUS GERAL = AMARELO
todos VERDE              → STATUS GERAL = VERDE
HRV em "experimental"    → não conta pro status geral
```

Bate com Selander 2019 / Hird 2016: regra clínica > score fundido.

### 7.3 Lógica híbrida absoluto → pessoal

```python
def evaluate_axis(metric_name, current_value, profile):
    n_baseline = profile["baseline_sessions_count"].get(metric_name, 0)
    if n_baseline < 5:
        return apply_absolute_thresholds(metric_name, current_value)
    else:
        mean = profile["baseline_mean"][metric_name]
        std  = profile["baseline_std"][metric_name]
        return apply_personal_thresholds(current_value, mean, std)
```

### 7.4 Atualização Welford de baseline

```python
profile["baseline_sessions_count"][m] += 1
n = profile["baseline_sessions_count"][m]
prev_mean = profile["baseline_mean"][m]
new_mean = prev_mean + (current_value - prev_mean) / n
profile["baseline_mean"][m] = new_mean
profile["baseline_M2"][m] += (current_value - prev_mean) * (current_value - new_mean)
profile["baseline_std"][m] = sqrt(profile["baseline_M2"][m] / max(1, n - 1))
```

Welford garante atualização sem precisar guardar histórico inteiro.

## 8. Persistência

### 8.1 `data/profiles.json`

```json
{
  "version": 1,
  "profiles": {
    "João Bornelli": {
      "created_at": "2026-05-01T14:00:00",
      "last_session_at": "2026-05-01T14:32:00",
      "total_sessions": 12,
      "rested_sessions": 7,
      "baseline_sessions_count": {
        "perclos_pct": 7, "bpm": 7, "rmssd_ms": 7,
        "pvt_mean_inv_rt": 7, "pvt_lapses": 7
      },
      "baseline_mean": { "perclos_pct": 4.1, "bpm": 68.3, "rmssd_ms": 38.2,
                         "pvt_mean_inv_rt": 3.51, "pvt_lapses": 0.7 },
      "baseline_std":  { "perclos_pct": 1.8, "bpm": 4.5, "rmssd_ms": 8.7,
                         "pvt_mean_inv_rt": 0.22, "pvt_lapses": 0.9 },
      "baseline_M2":   { "perclos_pct": 19.4, "bpm": 121.5, "rmssd_ms": 454.8,
                         "pvt_mean_inv_rt": 0.29, "pvt_lapses": 4.86 }
    }
  }
}
```

**Notas:**

- `version: 1` permite migração futura.
- `baseline_M2` (soma de quadrados) é guardado pro Welford operar sem histórico bruto.
- `total_sessions` ≠ `rested_sessions` — só sessões marcadas "descansado" alimentam baseline.

### 8.2 Escrita atômica

```python
def save_profiles(profiles_dict, path="data/profiles.json"):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(profiles_dict, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)
```

Evita corromper o arquivo se o app crasha no meio da escrita.

### 8.3 CSV por sessão

`data/sessions/2026-05-01_1432_joao.csv`:

| coluna | descrição |
|---|---|
| `timestamp` | ISO 8601 |
| `subject` | nome do perfil |
| `perclos_pct`, `bpm`, `rpm`, `snr_db` | passivos |
| `rmssd_ms`, `pnn50_pct`, `sdnn_ms` | HRV |
| `pvt_n_trials`, `pvt_mean_inv_rt`, `pvt_lapses`, `pvt_false_starts`, `pvt_mean_rt` | PVT |
| `pvt_trials_json` | array JSON inline com cada `{t, rt_ms}` (análise post-hoc) |
| `marked_rested` | bool |
| `status_per_axis_json` | `{"perclos": "green", ...}` |
| `status_overall` | green/yellow/red |
| `notes` | string livre, opcional |

### 8.4 Edge cases tratados

1. **Primeira sessão de um perfil**: `evaluate_axis` cai no ramo de cutoffs absolutos.
2. **Crash durante o PVT**: sessão não é salva, baseline preservado.
3. **Nome duplicado**: `create_profile` recusa, mostra erro.
4. **Métrica HRV inválida** (SNR baixo ou movimento): grava como vazio no CSV, baseline não atualiza pra HRV nessa sessão.
5. **PVT abortado** antes de 3 min: se `n_trials < 10`, descarta PVT, marca como None. PERCLOS+rPPG+HRV continuam válidos.
6. **Mudança de webcam/iluminação**: não automatizado v1; flag manual "resetar baseline".

## 9. Dependências novas

| Lib | Tamanho | Justificativa |
|---|---|---|
| `pygame` | ~10MB | Janela do PVT com timing SDL/vsync. Único método com publicação científica direta (PsychoPy backend). |
| `scipy.signal` | já instalado via numpy/mediapipe stack | `find_peaks`, `sosfiltfilt`, `butter` |

Adicionar a `requirements.txt`:

```
opencv-python
mediapipe
numpy
scipy
pygame
```

## 10. Fora de escopo desta fase

- **Webapp mobile**: portar PERCLOS/PVT/HRV pra JavaScript. Fica pra depois de validar no Python.
- **LF/HF frequency-domain HRV**: precisa ≥2 min e é mais sensível a artefato em rPPG. Time-domain (RMSSD/pNN50) é suficiente para v1.
- **Pupilometria / PLR**: candidato promissor (Solyman 2016, Karwoska 2022) mas fora de escopo v1; voltar quando tiver validação HRV concluída.
- **PVT-BA adaptativo**: defere; PVT-B é suficiente.
- **Score combinado fundido**: descartado conscientemente.
- **Multi-baseline (descansado / cansado simulado)**: por ora só o baseline "descansado" alimenta o evaluator.
- **Detecção automática de "descansado vs cansado"** sem input do motorista: deixa a marcação manual; automatizar abre frente de validação grande.
- **Sincronização de profiles entre máquinas**: local-only.

## 11. Critérios de sucesso

1. **Funcional:** check-in completo de 3 min termina sem crash, salva CSV + atualiza profile.
2. **Métricas razoáveis em sujeito descansado:** PERCLOS verde, BPM/RPM realistas, PVT mean_inv_rt ≥ 3.0, lapses ≤ 1.
3. **Sensibilidade a sleep deprivation:** após uma noite de < 4h sono em sujeito jovem, PVT mean_inv_rt e lapses devem ser piores que baseline pessoal por ≥ 0.75 std (sair de verde).
4. **Baseline estabiliza após 5-7 sessões descansado:** desvios padrão dos baselines de PVT mean_inv_rt e RMSSD devem convergir (variação < 10% entre sessão N e N+1).
5. **Calibração HRV comparada a Galaxy Watch:** bias < 10ms RMSSD em LoA Bland-Altman (ou marcar HRV como não-publicável).

## 12. Referências

Já presentes no README do projeto: [1] Verkruysse 2008, [2] de Haan & Jeanne 2013, [3] Wang 2017, [4] Poh 2010, [5] Welch 1967, [6] Tarvainen 2002, [7-8] BlazeFace/MediaPipe, [9] Kwon 2015, [10] Benezeth 2021, [11] McDuff 2023, [12] rPPG-Toolbox, [13] pyVHR, [14] iPhys, [15] Gasior 2004.

**Adicionais para esta fase:**

- Basner & Dinges (2011), "Maximizing sensitivity of the PVT to sleep loss," *Sleep* 34. DOI:10.1093/sleep/34.5.581
- Basner, Mollicone & Dinges (2011), "Validity and sensitivity of a brief PVT," *Acta Astronautica* 69. DOI:10.1016/j.actaastro.2011.07.015
- Brunet et al. (2017), "sleep-2-Peak: smartphone PVT," *Behav Res Methods* 49. DOI:10.3758/s13428-016-0802-5
- Kay et al. (2016), "Smartphone/tablet PVT validation," *Behav Res Methods*. DOI:10.3758/s13428-016-0763-8
- Vicente et al. (2016), "Drowsiness detection using HRV," *Med Biol Eng Comput*.
- Munoz et al. (2015), "Validity of (ultra-)short HRV recordings," *PLoS ONE* 10:e0138921.
- Yu et al. (2021), "PRV vs HRV from camera PPG," (validação rPPG-HRV).
- Task Force ESC/NASPE (1996), "Heart rate variability standards," *Circulation* 93.
- Nunan, Sandercock & Brodie (2010), "Quantitative review of HRV in healthy adults," *PACE*.
- Selander et al. (2019), "Driving assessment battery norms," *Scand J Occup Ther*. DOI:10.1080/11038128.2019.1614214
- Hird et al. (2016), "Standardized on-road tests in cognitive impairment — systematic review," PMID 27253511.

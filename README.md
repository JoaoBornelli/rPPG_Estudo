# rPPG Experiment — Remote Photoplethysmography via Webcam

Pipeline experimental em tempo real para estimativa de frequência cardíaca (BPM) e frequência respiratória (RPM) a partir de vídeo facial, utilizando técnicas de Remote Photoplethysmography (rPPG).

## Índice

- [Visão Geral do Pipeline](#visão-geral-do-pipeline)
- [Requisitos](#requisitos)
- [Uso](#uso)
- [Metodologia](#metodologia)
  - [1. Detecção Facial e Seleção de ROI](#1-detecção-facial-e-seleção-de-roi)
  - [2. Extração de Sinal — Métodos rPPG](#2-extração-de-sinal--métodos-rppg)
  - [3. Processamento de Sinal](#3-processamento-de-sinal)
  - [4. Estimativa de Frequência via Welch PSD](#4-estimativa-de-frequência-via-welch-psd)
  - [5. Visualização CHROM Espacial](#5-visualização-chrom-espacial)
- [Peer Review — Análise de Validade Científica](#peer-review--análise-de-validade-científica)
- [Limitações e Testes](#limitações-e-testes)
- [Parâmetros de Configuração](#parâmetros-de-configuração)
- [Referências](#referências)

---

## Visão Geral do Pipeline

```
Webcam (30fps)
  |
  v
MediaPipe Face Landmarker (478 landmarks)
  |
  v
Seleção de ROI (testa + bochechas bilaterais)
  |
  v
Média espacial RGB por frame → buffers temporais R(t), G(t), B(t)
  |
  v
Método de extração (GREEN | CHROM | POS) → sinal pulsátil 1D
  |
  v
Processamento: normalize → detrend linear → bandpass FFT → normalize
  |
  v
Welch PSD → pico espectral na banda cardíaca/respiratória
  |
  v
Estimativa de BPM, RPM e SNR
```

A interface exibe três tiles em tempo real (facial landmarks, heatmap CHROM das ROIs, resultado final com métricas), acompanhados de gráficos do sinal temporal e do espectro Welch PSD.

---

## Requisitos

```
python >= 3.10
opencv-python
mediapipe
numpy
```

Alem disso, o modelo `face_landmarker.task` deve estar no diretorio raiz do projeto (disponivel em [MediaPipe Models](https://developers.google.com/mediapipe/solutions/vision/face_landmarker#models)).

## Uso

```bash
python quick_rppg_experiment.py
```

- Pressione `ESC` para encerrar.
- Para usar um arquivo de video em vez da webcam, altere `SOURCE = 0` para o caminho do arquivo.

---

## Metodologia

Esta secao correlaciona cada etapa critica do pipeline com a fundamentacao teorica e as praticas recomendadas pela literatura.

### 1. Detecção Facial e Seleção de ROI

**Codigo**: linhas 44-46 (definicao dos landmarks), 79-104 (`build_roi_mask`)

#### 1.1 Detector: MediaPipe Face Landmarker

O pipeline utiliza o MediaPipe Face Landmarker [7, 8], que combina o detector BlazeFace (sub-milissegundo em GPUs mobile) com uma rede de regressao de 478 landmarks 3D. Opera no modo `VIDEO` com rastreamento de face unica (`num_faces=1`), eliminando a necessidade de re-deteccao frame a frame.

```python
# Linha 53-54
running_mode=vision.RunningMode.VIDEO,
num_faces=1,
```

#### 1.2 Regiões de Interesse (ROI)

Tres ROIs anatomicas sao utilizadas:

| ROI | Landmarks | Justificativa Fisiologica |
|-----|-----------|--------------------------|
| **Testa** (supraorbital) | `[54, 10, 67, 103, 109, 338, 297, 332, 284]` | Alta densidade capilar superficial, pele fina (~1.2 mm), minima oclusao por pelos faciais |
| **Bochecha esquerda** (malar) | `[117, 118, 50, 205, 187, 147, 213, 192]` | Regiao malar com perfusao cutanea elevada |
| **Bochecha direita** (malar) | `[346, 347, 280, 425, 411, 376, 433, 416]` | Simetrica a esquerda, aumenta area amostral |

A escolha de testa + bochechas bilaterais e validada por Kwon et al. [9], que sistematicamente avaliaram 7 regioes faciais e encontraram que testa e bochechas apresentam o maior SNR para rPPG. Benezeth et al. [10] estenderam a analise para 39 regioes com MediaPipe e 7 algoritmos, confirmando que as regioes malar direita, glabela e testa medial sao as TOP-5 — regioes com pele mais fina (1.191 um medio) fornecem sinal mais forte.

#### 1.3 Expansao da Testa

```python
# Linhas 87-93
forehead = np.stack([
    c_x + (forehead[:, 0] - c_x) * 1.12,   # 12% expansao horizontal
    c_y + (forehead[:, 1] - c_y) * 1.18,    # 18% expansao vertical
], axis=1)
```

A expansao radial a partir do centroide amplia a area amostral da testa, que tende a ser subestimada pelos landmarks que seguem a linha da sobrancelha. Os fatores 1.12x/1.18y sao heuristicos — a expansao vertical (18%) mais agressiva busca capturar mais area da testa em direcao ao couro cabeludo, embora isso carregue o risco de incluir pixels de cabelo em sujeitos com hairline baixo. O codigo aplica `np.clip` (linhas 94-95) para manter os pontos dentro dos limites do frame.

#### 1.4 Mascara Convexa

```python
# Linhas 101-103
cv.fillConvexPoly(mask, cv.convexHull(forehead), 255)
cv.fillConvexPoly(mask, cv.convexHull(left_cheek), 255)
cv.fillConvexPoly(mask, cv.convexHull(right_cheek), 255)
```

A envoltoria convexa (`convexHull`) sobre o conjunto esparso de landmarks produz poligonos que cobrem as ROIs. Embora a envoltoria convexa possa incluir alguns pixels fora da area cutanea ideal, as regioes escolhidas (testa e bochechas) sao suficientemente convexas para que a diferenca pratica em relacao a poligonos exatos seja pequena [10].

---

### 2. Extração de Sinal — Métodos rPPG

O pipeline implementa tres metodos classicos. Todos operam sobre as series temporais de medias espaciais R(t), G(t), B(t) extraidas da ROI.

#### 2.1 Metodo GREEN (Verkruysse et al. 2008)

**Codigo**: linhas 148-149 (`signal_green`)

```python
def signal_green(r, g, b):
    return normalize(g)
```

**Principio**: O canal verde (500-600 nm) coincide com o pico de absorcao da oxihemoglobina e desoxihemoglobina, resultando na maior amplitude de sinal pulsátil entre os tres canais RGB [1]. Verkruysse et al. [1] demonstraram pela primeira vez que sinais pletismograficos podem ser detectados remotamente usando camera digital e luz ambiente, com o canal G apresentando SNR superior.

**Limitacoes**: Sensivel a artefatos de movimento (o canal G captura tanto o sinal pulsátil quanto variacao de iluminacao) e a variacao de cor de pele (alta melanina atenua todos os canais proporcionalmente) [11].

#### 2.2 Metodo CHROM (de Haan & Jeanne 2013)

**Codigo**: linhas 163-170 (`signal_chrom`)

```python
def signal_chrom(r, g, b):
    r_n = normalize(r)      # z-score temporal
    g_n = normalize(g)
    b_n = normalize(b)
    x = 3.0 * r_n - 2.0 * g_n           # crominancia X
    y = 1.5 * r_n + g_n - 1.5 * b_n     # crominancia Y
    alpha = std(x) / (std(y) + eps)
    return normalize(x - alpha * y)      # projecao ortogonal
```

**Principio**: O CHROM [2] modela a reflectancia da pele como mistura de componentes especular e difuso. Ao projetar o sinal RGB normalizado nas direcoes de crominancia (Xs = 3R - 2G; Ys = 1.5R + G - 1.5B), o componente especular (iluminacao) e suprimido e o pulso e isolado como S = Xs - alpha * Ys, onde alpha = std(Xs)/std(Ys) adapta a projecao a iluminacao corrente.

**Correlacao com o paper original** [2]:
- **Coeficientes**: X = 3R - 2G e Y = 1.5R + G - 1.5B estao corretos (Eq. 5 e 6 do paper)
- **Combinacao**: S = X - alpha * Y (subtracao) esta correta
- **Alpha adaptativo**: alpha = std(X)/std(Y) esta correto (Eq. 7)

**Divergencia na normalizacao**: O paper original normaliza cada canal dividindo pela media espacial do frame atual: `R_n[t] = R[t] / mean(R)` — uma normalizacao por frame que remove o componente DC variavel. A implementacao utiliza z-score temporal (subtrai media e divide pelo desvio padrao sobre toda a janela de 12s). A diferenca e que o z-score altera a amplitude relativa entre canais, potencialmente afetando alpha. Alem disso, o paper calcula alpha dentro de janelas curtas (~1.6s) recalculando a cada janela [2, 14], enquanto esta implementacao calcula alpha sobre a janela completa de 12s, reduzindo a adaptabilidade a variacao de iluminacao dentro da janela.

**Performance**: Em testes com 117 sujeitos estaticos, CHROM obteve RMSE e desvio padrao 2x menores que metodos ICA/PCA [2]. Em benchmarks modernos no dataset UBFC-rPPG, atinge MAE de ~1-3 BPM [13].

#### 2.3 Metodo POS (Wang et al. 2017)

**Codigo**: linhas 152-160 (`signal_pos`)

```python
def signal_pos(r, g, b):
    rgb = np.vstack([r, g, b])          # shape: [3, N]
    means = np.mean(rgb, axis=1, keepdims=True)
    c = (rgb / means) - 1.0             # normalizacao temporal
    x = c[1] - c[2]                     # G - B
    y = c[1] + c[2] - 2.0 * c[0]       # G + B - 2R
    alpha = std(x) / (std(y) + eps)
    return normalize(x + alpha * y)      # projecao: SOMA
```

**Principio**: O POS [3] define um plano ortogonal ao vetor de tom de pele no espaco RGB normalizado. A matriz de projecao e derivada de primeiros principios geometricos:

```
P = [[0,  1, -1],     # e1: projeta em G-B
     [-2, 1,  1]]     # e2: projeta em -2R+G+B
S = e1 + alpha * e2   # SOMA (diferente do CHROM que usa subtracao)
```

**Divergencias identificadas na implementacao**:

1. **Normalizacao temporal vs. por frame**: O paper original normaliza cada frame individualmente pela media espacial dos pixels da ROI naquele frame: `C_n = diag(mean(C))^-1 @ C` [3, 14]. A implementacao calcula `means = np.mean(rgb, axis=1)` — a media temporal de cada canal sobre toda a janela — que e semanticamente diferente. A normalizacao por frame remove variacoes de iluminacao frame a frame; a temporal nao.

2. **Indexacao dos canais**: O `np.vstack` cria `rgb[0]=R, rgb[1]=G, rgb[2]=B`. Portanto `x = c[1]-c[2]` corresponde a `G-B` e `y = c[1]+c[2]-2*c[0]` corresponde a `G+B-2R`. O paper define `S1 = G-B` e `S2 = -2R+G+B` usando a matriz `[[0,1,-1],[-2,1,1]]` [3, 14] — os sinais estao equivalentes.

3. **Combinacao**: A implementacao usa `x + alpha * y` (soma), que esta correto para POS. O CHROM usa subtracao [2], POS usa adicao [3] — confirmado nas implementacoes de referencia do iPhys Toolbox [14].

---

### 3. Processamento de Sinal

O processamento e aplicado na funcao `compute_psd_standard` (linhas 238-255), que executa quatro etapas em sequencia.

#### 3.1 Normalizacao Z-score

**Codigo**: linhas 61-66 (`normalize`), aplicada na linha 249

```python
def normalize(x):
    return (x - mean(x)) / std(x)
```

**Fundamentacao**: A normalizacao z-score (media zero, variancia unitaria) remove o offset DC e equaliza a escala de amplitude, permitindo comparacao entre sinais de diferentes magnitudes. E uma pratica padrao em pipelines de rPPG antes de processamento espectral [2, 3, 11].

#### 3.2 Remocao de Tendencia Linear

**Codigo**: linhas 191-197 (`detrend_linear`), aplicada na linha 250

```python
def detrend_linear(signal):
    p = np.polyfit(x, signal, deg=1)
    return signal - (p[0] * x + p[1])
```

**Fundamentacao**: Remove deriva lenta causada por variacao gradual de iluminacao, movimento lento da cabeca ou auto-exposicao da camera. O detrending linear (polinomio de grau 1 via minimos quadrados) e a abordagem mais simples.

**Alternativa recomendada pela literatura**: Tarvainen et al. [6] propuseram um metodo baseado em smoothness priors (filtro passa-alta regularizado) que e mais adequado para drifts nao-lineares: `z_stat = (I - (I + lambda^2 * D2' * D2)^-1) * z`, com lambda tipico de 100-300 para rPPG a 30 FPS. Para janelas curtas (12s) em condicoes estaticas, o detrend linear e uma aproximacao aceitavel.

#### 3.3 Filtro Passa-banda via FFT

**Codigo**: linhas 200-207 (`bandpass_fft`), aplicada na linha 251

```python
def bandpass_fft(signal, fs, low_hz, high_hz):
    spec = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), d=1.0/fs)
    spec[(freqs < low_hz) | (freqs > high_hz)] = 0.0
    return np.fft.irfft(spec, n=len(signal))
```

**Fundamentacao**: Restringe o sinal a banda de interesse fisiologico antes da estimativa espectral, removendo ruido fora da banda. As bandas utilizadas sao:

| Sinal | Banda | Faixa fisiologica | Referencia |
|-------|-------|-------------------|------------|
| Cardiaco | 0.8 - 3.2 Hz | 48 - 192 bpm | [1, 2] |
| Respiratorio | 0.1 - 0.5 Hz | 6 - 30 rpm | [4] |

A banda cardíaca 0.8-3.2 Hz e consistente com a literatura (tipicamente 0.65-4.0 Hz) [2]. O limite inferior de 0.8 Hz reduz contaminacao pelo componente respiratorio (0.1-0.5 Hz).

**Nota tecnica**: Esta implementacao usa um filtro ideal (retangular) no dominio da frequência, que equivale a convolucao com uma funcao sinc no tempo. Isso pode introduzir artefatos de Gibbs (ringing) nas bordas de transientes [14]. A pratica padrao em rPPG e utilizar filtro Butterworth de ordem 3-6, zero-phase (`scipy.signal.sosfiltfilt`) [2, 14]:

```python
# Alternativa recomendada (nao implementada):
sos = butter(N=6, Wn=[0.7, 2.5], btype='bandpass', fs=fps, output='sos')
filtered = sosfiltfilt(sos, signal)
```

O filtro FFT ideal e aceitavel quando utilizado exclusivamente como pre-processamento antes do Welch PSD, onde o objetivo e analise espectral e nao reconstrucao do sinal no dominio do tempo.

#### 3.4 Normalizacao Final

A segunda chamada a `normalize()` (linha 252) re-normaliza o sinal filtrado antes de alimentar o Welch PSD, garantindo media zero e variancia unitaria para a estimativa espectral.

---

### 4. Estimativa de Frequencia via Welch PSD

#### 4.1 Implementacao de Welch

**Codigo**: linhas 210-235 (`welch_psd`)

```python
win = np.hanning(nperseg)                    # janela de Hann
win_pow = np.sum(win**2)                     # normalizacao de potencia
for start in range(0, n - nperseg + 1, step):
    seg = signal[start:start + nperseg]
    seg = seg - np.mean(seg)                 # remove DC por segmento
    p = (|fft(seg * win)|^2) / win_pow       # periodograma janelado
    acc += p                                 # acumula
psd = acc / count                            # media dos periodogramas
```

**Fundamentacao**: O metodo de Welch [5] estima a Power Spectral Density (PSD) dividindo o sinal em segmentos sobrepostos, aplicando janelamento, calculando o periodograma de cada segmento e fazendo a media. Isso reduz a variancia da estimativa espectral ao custo de resolucao em frequência — trade-off fundamental para sinais fisiologicos curtos.

**Análise dos parametros utilizados**:

| Parametro | Valor (cardiaco) | Valor (respiratorio) | Efeito |
|-----------|-------------------|----------------------|--------|
| `seg_sec` | 5.0 s | 20.0 s | Resolucao: df = 1/seg_sec |
| `overlap` | 50% | 75% | Mais segmentos = menor variancia |
| Janela | Hann | Hann | Sidelobes -32 dB |

Para o sinal cardiaco com janela de 12s e segmentos de 5s com 50% overlap:
- Resolucao espectral: df = 1/5 = 0.2 Hz = **12 BPM**
- Numero de segmentos: ~3-4

A resolucao de 12 BPM e aceitavel para monitoramento grosseiro mas insuficiente para aplicacoes clinicas (onde se espera +-2 BPM). A literatura recomenda segmentos de 8-10s para resolucao de 6-7.5 BPM [11, 14], o que requer janela total de 15-20s.

**Janela de Hann**: Escolha padrao e adequada — sidelobes de -32 dB sao suficientes para a separacao entre componentes cardiaco e respiratorio. Blackman (-58 dB) ou Kaiser (parametrizavel) ofereceriam melhor isolamento de harmonicos, mas a diferenca e negligenciavel para rPPG [14].

#### 4.2 Deteccao de Pico e Estimativa de BPM

**Codigo**: linhas 258-278 (`estimate_rate`)

```python
i = np.argmax(s)                             # bin de maior potencia
peak_hz = f[i]
rate = peak_hz * 60.0                        # conversao Hz → BPM/RPM
```

**Fundamentacao**: O argmax no espectro Welch retorna a frequência do bin com maior potencia dentro da banda fisiologica. E o metodo mais simples e robusto para sinais com pico espectral dominante.

**Resolucao**: Limitada pela largura do bin FFT. Para fs=30 Hz e nperseg=150: df = 0.2 Hz = 12 BPM. Tecnicas mais precisas incluem:

- **Interpolacao parabolica** [15]: estima posicao sub-bin usando os 3 bins ao redor do pico: `p = 0.5*(alpha-gamma)/(alpha-2*beta+gamma)`. Ganho tipico de 1 ordem de magnitude em resolucao, sem custo computacional significativo.
- **Centroide espectral** (media ponderada): `f_hr = sum(f_i * PSD_i) / sum(PSD_i)` — mais robusto a ruido que argmax mas menos preciso com pico limpo.

#### 4.3 Calculo de SNR

```python
peak_power = s[i]
noise_power = (sum(s) - peak_power) / (len(s) - 1)
snr_db = 10 * log10(peak_power / noise_power)
```

**Fundamentacao**: SNR intra-banda — razao entre potencia do pico e media dos demais bins dentro da banda de interesse. A definicao de de Haan & Jeanne [2] e McDuff et al. [11] usa uma janela estreita ao redor do pico (+-0.15 Hz) como sinal e o restante da banda como ruido, incluindo harmonicos. A implementacao aqui e uma estimativa mais conservadora (SNR menor, pois todo o ruido intra-banda entra no denominador).

---

### 5. Visualização CHROM Espacial

**Codigo**: linhas 288-352 (`apply_chrom_spatial`, `build_roi_chrom_tile`)

```python
x_ch = 3.0 * r - 2.0 * g              # CHROM X por pixel
y_ch = 1.5 * r + g - 1.5 * b          # CHROM Y por pixel
chrom = x_ch - alpha * y_ch            # projecao com alpha temporal
```

**Natureza**: Esta visualizacao aplica os coeficientes CHROM pixel a pixel em um unico frame, utilizando o `alpha` calculado temporalmente (linha 543). Isso e uma **heuristica de visualizacao** — nao uma aplicacao canonica do metodo CHROM, que opera sobre series temporais de medias espaciais [2].

O alpha temporal (razao de variancias temporais de X e Y ao longo de 12s) e usado para combinar as crominancias espaciais, o que assume que a relacao temporal entre variancias e transferivel para o dominio espacial. Nao ha fundamentacao publicada para essa transferencia. O resultado e util como ferramenta de inspecao visual (mostra contraste de crominancia com cancelamento de iluminacao), mas nao representa uma estimativa de sinal rPPG por pixel.

A normalizacao por percentil 2%-98% (linhas 299-301) remove outliers para a visualizacao sem saturacao extrema, aplicando colormap `INFERNO`.

---

## Peer Review — Análise de Validade Cientifica

### Tabela de Conformidade

| Componente | Status | Impacto | Detalhe |
|:-----------|:------:|:-------:|:--------|
| Landmarks ROI (testa + bochechas) | CORRETO | Baixo | Regioes validadas por [9, 10] |
| Expansao da testa (1.12x/1.18y) | PARCIAL | Baixo | Heuristica sem referencia publicada; risco de capturar cabelo |
| Mascara convexa | PARCIAL | Baixo | Pode incluir pixels nao-cutaneos entre landmarks |
| Metodo GREEN | CORRETO | — | Conforme Verkruysse et al. [1] |
| Metodo CHROM (coeficientes) | CORRETO | — | Coeficientes X=3R-2G, Y=1.5R+G-1.5B corretos [2] |
| Metodo CHROM (normalizacao) | PARCIAL | Medio | Z-score temporal vs. divisao pela media por frame [2] |
| Metodo POS (projecao) | CORRETO | — | Indices e sinais equivalentes ao paper [3] |
| Metodo POS (normalizacao) | PARCIAL | Medio | Media temporal vs. por frame [3] |
| `normalize()` z-score | CORRETO | — | Pratica padrao [2, 3] |
| `detrend_linear()` | PARCIAL | Baixo | Suficiente para condicoes estaticas; Tarvainen [6] seria mais robusto |
| `bandpass_fft()` ideal | PARCIAL | Medio | Risco de Gibbs; Butterworth zero-phase preferivel [2, 14] |
| `welch_psd()` implementacao | CORRETO | — | Metodo de Welch [5] implementado corretamente |
| Resolucao espectral (12 BPM) | PARCIAL | Medio | Insuficiente para aplicacao clinica; interpolacao parabolica [15] melhoraria |
| SNR intra-banda | PARCIAL | Baixo | Definicao mais conservadora que o padrao [2, 11] |
| `HEART_BAND_HZ` (0.8-3.2 Hz) | CORRETO | — | Consistente com [1, 2] |
| `RESP_BAND_HZ` (0.1-0.5 Hz) | CORRETO | — | Faixa fisiologica padrao [4] |
| CHROM espacial (visualizacao) | HEURISTICA | — | Util para inspecao visual, sem base publicada |

### Problemas Nao Tratados

| Aspecto | Impacto | Mitigacao Recomendada |
|---------|---------|----------------------|
| **Artefatos de movimento** | Alto | Deteccao de frames ruins via variancia inter-frame; estimativa de movimento rigido via landmarks [4] |
| **Auto-exposicao/AWB da camera** | Medio | Desabilitar via `cv.VideoCapture.set(CAP_PROP_AUTO_EXPOSURE, 0)` |
| **Jitter de amostragem** | Baixo-Medio | Resample para grade uniforme antes do Welch; a media de dt (linha 188) mascara jitter |
| **Ausencia de deteccao de pele** | Baixo | Limiar HSV adaptativo ou Elliptical Skin Model para mascarar pixels nao-pele |

---

## Limitacoes e Testes

### Status de Validacao

> **IMPORTANTE**: Os testes deste codigo foram limitados e nao constituem validacao clinica ou academica formal.

A validacao foi realizada de forma **comparativa informal**: as leituras de BPM produzidas pelo pipeline rPPG foram contrastadas com as medicoes simultaneas de um **Samsung Galaxy Watch 7** (sensor optico PPG de contato no pulso). Essas comparacoes foram feitas em um numero restrito de sessoes, sob condicoes controladas (sujeito estatico, iluminacao artificial estavel), e nao seguiram protocolo de pesquisa padronizado (sem IRB, sem amostra estatistica, sem dataset de referencia).

### Ressalvas

- **Nenhum dataset padrao** (UBFC-rPPG, PURE, MMPD) foi utilizado para validacao quantitativa.
- O Galaxy Watch 7 e um dispositivo de consumo, **nao um gold standard clinico** (como ECG ou oximetro de pulso certificado). Suas leituras estao sujeitas a erros proprios (movimento do pulso, ajuste da pulseira).
- Nao foram coletadas metricas quantitativas formais (MAE, RMSE, correlacao de Pearson, limites de Bland-Altman).
- Os testes foram realizados em **um unico sujeito** sob condicoes estaticas favoraveis — nao representam a diversidade de tons de pele, condicoes de iluminacao ou niveis de atividade encontrados em estudos robustos.
- Para validacao cientifica adequada, recomenda-se utilizar os benchmarks padronizados do **rPPG-Toolbox** [12] ou **pyVHR** [13] sobre datasets publicos com ground truth de ECG.

---

## Parametros de Configuracao

| Parametro | Valor Padrao | Descricao |
|-----------|:------------:|-----------|
| `HEART_METHOD` | `"CHROM"` | Metodo de extracao cardíaca (GREEN, POS, CHROM) |
| `RESP_METHOD` | `"GREEN"` | Metodo de extracao respiratória |
| `HEART_WINDOW_SEC` | `12.0` | Janela temporal para analise cardíaca (s) |
| `MIN_HEART_WINDOW_SEC` | `6.0` | Janela minima para iniciar estimativa (s) |
| `RESP_WINDOW_SEC` | `30.0` | Janela temporal para analise respiratória (s) |
| `HEART_BAND_HZ` | `(0.8, 3.2)` | Banda de frequência cardíaca (Hz) = 48-192 bpm |
| `RESP_BAND_HZ` | `(0.1, 0.5)` | Banda de frequência respiratória (Hz) = 6-30 rpm |
| `WELCH_SEG_SEC_HEART` | `5.0` | Comprimento do segmento Welch para HR (s) |
| `WELCH_OVERLAP_HEART` | `0.5` | Sobreposicao dos segmentos Welch (HR) |
| `TARGET_TILE_HEIGHT` | `360` | Altura dos tiles de visualizacao (px) |

---

## Referencias

```
[1]  W. Verkruysse, L. O. Svaasand, and J. S. Nelson, "Remote plethysmographic
     imaging using ambient light," Optics Express, vol. 16, no. 26, pp. 21434-21445,
     2008. DOI: 10.1364/OE.16.021434

[2]  G. de Haan and V. Jeanne, "Robust pulse rate from chrominance-based rPPG,"
     IEEE Trans. Biomed. Eng., vol. 60, no. 10, pp. 2878-2886, 2013.
     DOI: 10.1109/TBME.2013.2266196

[3]  W. Wang, A. C. den Brinker, S. Stuijk, and G. de Haan, "Algorithmic principles
     of remote PPG," IEEE Trans. Biomed. Eng., vol. 64, no. 7, pp. 1479-1491, 2017.
     DOI: 10.1109/TBME.2016.2609282

[4]  M.-Z. Poh, D. J. McDuff, and R. W. Picard, "Non-contact, automated cardiac
     pulse measurements using video imaging and blind source separation," Optics
     Express, vol. 18, no. 10, pp. 10762-10774, 2010. DOI: 10.1364/OE.18.010762

[5]  P. D. Welch, "The use of fast Fourier transform for the estimation of power
     spectra: A method based on time averaging over short, modified periodograms,"
     IEEE Trans. Audio Electroacoust., vol. 15, no. 2, pp. 70-73, 1967.
     DOI: 10.1109/TAU.1967.1161901

[6]  M. P. Tarvainen, P. O. Ranta-aho, and P. A. Karjalainen, "An advanced
     detrending method with application to HRV analysis," IEEE Trans. Biomed. Eng.,
     vol. 49, no. 2, pp. 172-175, 2002. DOI: 10.1109/10.979357

[7]  V. Bazarevsky, Y. Kartynnik, A. Vakunov, K. Raveendran, and M. Grundmann,
     "BlazeFace: Sub-millisecond neural face detection on mobile GPUs,"
     arXiv:1907.05047, 2019.

[8]  Y. Kartynnik, A. Ablavatski, I. Grishchenko, and M. Grundmann, "Real-time
     facial surface geometry from monocular video on mobile GPUs,"
     arXiv:1907.06724, 2019.

[9]  S. Kwon, J. Kim, D. Lee, and K. S. Park, "ROI analysis for remote
     photoplethysmography on facial video," in Proc. IEEE EMBC, Milan, Italy,
     2015, pp. 4938-4941. DOI: 10.1109/EMBC.2015.7319499

[10] Y. Benezeth et al., "Assessment of ROI selection for facial video-based rPPG,"
     Sensors, vol. 21, no. 23, p. 7923, 2021. DOI: 10.3390/s21237923

[11] D. McDuff, "Camera measurement of physiological vital signs," ACM Comput.
     Surv., vol. 55, no. 9, Art. 176, pp. 1-40, 2023. DOI: 10.1145/3558518

[12] X. Liu et al., "rPPG-Toolbox: Deep remote PPG toolbox," in Proc. NeurIPS
     2023 Datasets and Benchmarks Track. arXiv:2210.00716.
     GitHub: https://github.com/ubicomplab/rPPG-Toolbox

[13] G. Boccignone et al., "pyVHR: A Python framework for remote
     photoplethysmography," PeerJ Comput. Sci., vol. 8, p. e929, 2022.
     DOI: 10.7717/peerj-cs.929
     GitHub: https://github.com/phuselab/pyVHR

[14] D. McDuff, "iPhys: An Open Non-Contact Imaging-Based Physiological
     Measurement Toolbox," in Proc. IEEE EMBC, 2019.
     GitHub: https://github.com/danmcduff/iphys-toolbox

[15] M. Gasior and J. L. Gonzalez, "Improving FFT frequency measurement resolution
     by parabolic and Gaussian spectrum interpolation," AIP Conf. Proc., vol. 732,
     pp. 276-285, 2004. CERN: AB-Note-2004-021
```

# Estimativa de Consumo de Recursos Computacionais

Documento de analise e estimativa de recursos necessarios para execucao completa do pipeline. As estimativas originais (secoes 1-7) cobriam o fluxo "treino unico" do esquema 64/16/20 e seguem validas para um treino isolado. **O protocolo atual inclui (a) splits 80/10/10 + CV 5-fold e (b) `RandomizedSearchCV` por modelo antes dos 3 regimes de avaliacao**, o que multiplica a carga total. Ver a secao final **Adendo: orcamento da nova metodologia (search-then-evaluate)**.

---

## 1. Hardware disponivel

| Componente | Especificacao |
|------------|---------------|
| CPU | Intel Core i7-11370H @ 3.30 GHz (4 nucleos / 8 threads, Tiger Lake) |
| RAM | 16 GB DDR4 (~12 GB disponiveis para o pipeline) |
| GPU | NVIDIA GeForce GTX 1650 Mobile (Turing, 1024 CUDA cores) |
| VRAM | 4 GB GDDR6 (~3.9 GB utilizaveis) |
| FP16 throughput | ~3.5-4.2 TFLOPS |
| TDP GPU | 50 W (variante mobile) |
| CUDA | 13.0 |
| Disco | SSD (leitura ~500 MB/s) |

---

## 2. Corpus e volumes de dados

### 2.1 Estatisticas do corpus

| Metrica | Valor |
|---------|-------|
| Total de artigos | 167.053 |
| Classe `mercado` (positivo) | 20.970 (12,55%) |
| Classe `outros` (negativo) | 146.083 (87,45%) |
| Arquivo CSV | 503,6 MB |
| DataFrame em memoria (text + category) | 676,7 MB |

### 2.2 Comprimento dos textos

| Metrica | Caracteres | Palavras |
|---------|-----------|----------|
| Media | 2.715 | 445 |
| Mediana | 2.409 | 392 |
| Percentil 95 | 5.782 | 947 |
| Maximo | 61.154 | 10.434 |

### 2.3 Tokenizacao BERT estimada

Estimativa de tokens WordPiece (fator ~1,3x sobre contagem de palavras para portugues):

| Metrica | Tokens estimados |
|---------|-----------------|
| Media | 578 |
| Mediana | 510 |
| Percentil 95 | 1.231 |
| Textos truncados em `max_length=256` | 139.083 (83,3%) |
| Textos truncados em `max_length=512` | 83.192 (49,8%) |

**Nota:** Com `max_length=256`, a maioria dos textos sera truncada. Isso e aceitavel conforme justificado no `REQUIREMENTS.md` (~95% dos textos sem truncamento se refere a caracteres, nao tokens). A truncagem de tokens descarta porcoes significativas do texto — isso pode impactar recall para artigos longos cujos sinais economicos aparecem apos os primeiros ~200 palavras.

### 2.4 Volumes por split

| Split | Linhas | Mercado (%) | Uso |
|-------|--------|-------------|-----|
| Treino bruto | ~106.914 | 12,55% | Base para balanceamento |
| Treino balanceado | ~33.552 | 50,0% | Treino de modelos supervisionados |
| Validacao | ~26.728 | 12,55% | Ajuste, calibracao, stacking |
| Teste | ~33.411 | 12,55% | Avaliacao final (execucao unica) |

---

## 3. Estimativa por metodo

### 3.1 Metodos TF-IDF (M1, M2, M3)

#### 3.1.1 Vetorizacao TF-IDF (comum aos tres)

Parametros do vetorizador:

| Parametro | Valor | Impacto |
|-----------|-------|---------|
| `max_features` | 50.000 | Vocabulario final truncado |
| `ngram_range` | (1, 2) | Unigramas + bigramas |
| `sublinear_tf` | True | Escala log — sem impacto em memoria |
| `min_df` | 2 | Remove hapax legomena |
| `max_df` | 0.95 | Remove termos quase universais |

Estimativa da matriz TF-IDF esparsa:

| Dataset | Dimensao | Non-zero/linha (est.) | Tamanho em memoria (CSR) |
|---------|----------|----------------------|--------------------------|
| Treino balanceado | 33.552 x 50.000 | ~600-800 | ~250-320 MB |
| Validacao | 26.728 x 50.000 | ~600-800 | ~200-260 MB |
| Teste | 33.411 x 50.000 | ~600-800 | ~250-320 MB |

A vetorizacao `fit_transform` no treino e a etapa mais pesada em CPU. O `ngram_range=(1,2)` gera um vocabulario bruto da ordem de centenas de milhares de n-gramas antes do corte em 50.000 — essa construcao intermediaria consome memoria adicional.

**Pico de RAM durante vetorizacao:** ~2,0-3,0 GB (DataFrame + vocabulario intermediario + matriz esparsa).

#### 3.1.2 M1 — TF-IDF + Regressao Logistica

| Recurso | Estimativa |
|---------|-----------|
| Algoritmo | `LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)` |
| Tempo de treino | ~30-60 segundos |
| Tempo de inferencia (val) | ~5-10 segundos |
| Pico de RAM | ~2,5-3,5 GB |
| GPU | Nao utilizada |
| CPU | Multi-thread (lbfgs usa BLAS — escala com nucleos) |
| Modelo serializado (joblib) | ~30-80 MB |

O solver `lbfgs` utiliza rotinas BLAS otimizadas e escala razoavelmente com o numero de threads disponiveis (8 neste hardware). Para uma matriz esparsa de 33.552 x 50.000 com classificacao binaria, convergencia em <100 iteracoes e tipica.

#### 3.1.3 M2 — TF-IDF + LinearSVC + CalibratedClassifierCV

| Recurso | Estimativa |
|---------|-----------|
| Algoritmo | `CalibratedClassifierCV(LinearSVC(C=1.0), cv=3)` |
| Tempo de treino | ~2-4 minutos |
| Tempo de inferencia (val) | ~5-10 segundos |
| Pico de RAM | ~3,0-4,0 GB |
| GPU | Nao utilizada |
| CPU | Multi-thread (liblinear + sigmoid/isotonic calibration) |
| Modelo serializado (joblib) | ~50-120 MB |

**Nota importante:** `CalibratedClassifierCV(cv=3)` treina o `LinearSVC` **3 vezes** internamente (uma por fold), alem de ajustar os calibradores. O tempo total e ~3-4x maior que um `LinearSVC` isolado. O modelo serializado tambem e maior, pois armazena os 3 calibradores.

#### 3.1.4 M3 — TF-IDF + Multinomial Naive Bayes

| Recurso | Estimativa |
|---------|-----------|
| Algoritmo | `MultinomialNB(alpha=1.0)` |
| Tempo de treino | ~10-20 segundos |
| Tempo de inferencia (val) | ~3-5 segundos |
| Pico de RAM | ~2,0-3,0 GB |
| GPU | Nao utilizada |
| CPU | Single-thread (passagem unica) |
| Modelo serializado (joblib) | ~20-50 MB |

O `MultinomialNB` e o mais rapido dos tres: uma unica passagem sobre os dados para calcular probabilidades condicionais. Nao requer otimizacao iterativa.

#### 3.1.5 Resumo TF-IDF

| Metodo | Treino | Inferencia (val) | RAM pico | Disco (modelo) |
|--------|--------|------------------|----------|-----------------|
| M1 LogReg | ~45s | ~8s | ~3 GB | ~50 MB |
| M2 LinearSVC | ~3 min | ~8s | ~3,5 GB | ~80 MB |
| M3 NB | ~15s | ~4s | ~2,5 GB | ~30 MB |
| **Total TF-IDF** | **~5 min** | **~20s** | **~3,5 GB** | **~160 MB** |

---

### 3.2 Metodos BERT (M4a, M4b, M4c)

Esta e a etapa critica do pipeline em termos de tempo e recursos.

#### 3.2.1 Parametros de treino (identicos para M4a, M4b, M4c)

| Parametro | Valor | Impacto em recursos |
|-----------|-------|---------------------|
| `model_name` | BERT-base (~110M parametros) | Define VRAM base |
| `max_length` | 256 | Define tamanho dos tensores de entrada |
| `per_device_train_batch_size` | 1 | Minimo possivel — VRAM limitada |
| `gradient_accumulation_steps` | 8 | Batch efetivo = 8 |
| `per_device_eval_batch_size` | 1 | Conservador — pode ser aumentado |
| `num_train_epochs` | 3 | Maximo (early stopping pode parar em 2) |
| `fp16` | True (auto, se CUDA disponivel) | Reduz VRAM e acelera compute |
| `learning_rate` | 2e-5 | Sem impacto em recursos |
| `weight_decay` | 0.01 | Sem impacto significativo |
| `warmup_ratio` | 0.1 | Sem impacto em recursos |
| `eval_strategy` | `"epoch"` | 3 avaliacoes por treino |
| `save_strategy` | `"epoch"` | 3 checkpoints salvos (disco) |
| `load_best_model_at_end` | True | Requer manter checkpoints |

#### 3.2.2 Consumo de VRAM durante treino (FP16)

Decomposicao para BERT-base (110M parametros), `batch_size=1`, `max_length=256`:

| Componente | Bytes por parametro | Total |
|------------|-------------------|-------|
| Parametros do modelo (FP16) | 2 | ~220 MB |
| Copia master FP32 (para optimizer) | 4 | ~440 MB |
| Estados do AdamW: momento (FP32) | 4 | ~440 MB |
| Estados do AdamW: variancia (FP32) | 4 | ~440 MB |
| Gradientes (FP16) | 2 | ~220 MB |
| **Subtotal modelo + otimizador** | | **~1.760 MB** |

| Componente | Estimativa |
|------------|-----------|
| Ativacoes para backprop (12 layers, batch=1, seq=256) | ~200-400 MB |
| Overhead PyTorch + CUDA context | ~300-500 MB |
| **Total VRAM estimado** | **~2.300-2.700 MB** |
| **VRAM livre restante** | **~1.200-1.600 MB** |

**Veredicto: cabe na GTX 1650 (4 GB) com FP16, porem com margem apertada.** Se houver fragmentacao de memoria CUDA ou picos inesperados, um `OutOfMemoryError` e possivel. Estrategias de mitigacao:

1. `torch.cuda.empty_cache()` antes do treino
2. Gradient checkpointing (`gradient_checkpointing=True` no `TrainingArguments`) — troca tempo por VRAM, reduzindo ativacoes em ~60% ao custo de ~20% mais tempo
3. Reducao de `max_length` para 128 (reduz ativacoes pela metade)

#### 3.2.3 Consumo de VRAM durante inferencia

Sem gradientes, optimizer ou ativacoes de backprop:

| Componente | Estimativa |
|------------|-----------|
| Modelo (FP16) | ~220 MB |
| Ativacoes forward (batch=1, seq=256) | ~50-100 MB |
| Overhead CUDA | ~300 MB |
| **Total** | **~600-700 MB** |

**O `per_device_eval_batch_size` pode ser aumentado para 8-16** sem risco de OOM, reduzindo significativamente o tempo de avaliacao (recomendacao: ajustar para 8).

#### 3.2.4 Estimativa de tempo de treino

Velocidade estimada no GTX 1650 Mobile (FP16, batch_size=1, seq_len=256):

| Operacao | Tempo por amostra |
|----------|-------------------|
| Forward pass | ~30-50 ms |
| Backward pass | ~60-100 ms |
| **Total (forward + backward)** | **~90-150 ms** |

Estimativa por epoca:

| Componente | Calculo | Tempo |
|------------|---------|-------|
| Treino (33.552 amostras) | 33.552 x 0,12s | ~4.026s ≈ **67 min** |
| Avaliacao no val (26.728 amostras) | 26.728 x 0,04s | ~1.069s ≈ **18 min** |
| **Total por epoca** | | **~85 min** |

Estimativa total por modelo:

| Cenario | Epocas | Tempo treino | Tempo eval | **Total** |
|---------|--------|-------------|------------|-----------|
| Completo (3 epocas) | 3 | ~201 min | ~54 min | **~255 min ≈ 4,2 h** |
| Early stopping (2 epocas) | 2 | ~134 min | ~36 min | **~170 min ≈ 2,8 h** |

Estimativa total para os 3 modelos BERT:

| Cenario | M4a (BERTimbau) | M4b (FinBERT-PT-BR) | M4c (DeB3RTa) | **Total** |
|---------|----------------|---------------------|---------------|-----------|
| 3 epocas | ~4,2 h | ~4,2 h | ~3,0 h | **~11,4 h** |
| 2 epocas (early stop) | ~2,8 h | ~2,8 h | ~2,0 h | **~7,6 h** |

**Nota:** Os treinos devem ser executados sequencialmente (VRAM insuficiente para paralelizar). DeB3RTa-base (~71M params) e mais leve que BERTimbau/FinBERT-PT-BR (~110M cada), portanto roda mais rapido.

#### 3.2.5 Consumo de RAM durante treino BERT

| Componente | Estimativa |
|------------|-----------|
| DataFrame do corpus em memoria | ~700 MB |
| Dataset HuggingFace tokenizado (treino) | ~150-250 MB |
| Dataset HuggingFace tokenizado (val) | ~120-200 MB |
| Modelo em CPU (antes de `.to(cuda)`) | ~440 MB |
| Overhead Trainer + logging | ~200-400 MB |
| **Pico de RAM total** | **~2,0-3,0 GB** |

A RAM do sistema (16 GB) nao sera gargalo.

#### 3.2.6 Consumo de disco (checkpoints)

| Artefato | Por modelo | 3 modelos |
|----------|-----------|-----------|
| Checkpoints intermediarios (3 epocas) | ~1,3 GB | ~3,9 GB |
| Modelo final salvo | ~440 MB | ~1,3 GB |
| **Total durante treino** | **~1,7 GB** | **~5,2 GB** |
| **Total permanente (apos limpeza)** | **~440 MB** | **~1,3 GB** |

Os checkpoints intermediarios podem ser removidos apos o treino. Se disco for restricao, configurar `save_total_limit=1` no `TrainingArguments` para manter apenas o melhor checkpoint.

---

### 3.3 Estrategias de ensemble (E1-E4)

As operacoes de ensemble sao computacionalmente triviais, operando sobre as predicoes ja geradas.

| Estrategia | Operacao | Tempo | RAM adicional |
|-----------|----------|-------|---------------|
| E1 Votacao majoritaria | Soma de 7 vetores binarios | <1 s | ~5 MB |
| E2 Votacao ponderada | Media ponderada de 7 vetores float | <1 s | ~5 MB |
| E3 Stacking | LogReg sobre matriz 26.728 x 7 (treino no val) | <1 s | ~5 MB |
| E4 Concordancia | Tabela de contingencia | <1 s | ~5 MB |

O stacking (E3) treina uma `LogisticRegression` sobre uma matriz de 7 features x 26.728 amostras — execucao instantanea.

---

## 4. Estimativa consolidada do pipeline completo

### 4.1 Tempo de execucao

Execucao sequencial, do notebook 01 ao 43:

| Etapa | Descricao | Tempo estimado |
|-------|-----------|---------------|
| 01 | Carga do corpus, splits, persistencia | ~2-3 min |
| 11 | TF-IDF + LogReg (treino + eval no val) | ~2-3 min |
| 12 | TF-IDF + LinearSVC (treino + eval no val) | ~4-6 min |
| 13 | TF-IDF + MultinomialNB (treino + eval no val) | ~1-2 min |
| 21 | BERT BERTimbau (treino + eval no val) | ~3-4 h |
| 21 | BERT FinBERT-PT-BR (treino + eval no val) | ~3-4 h |
| 21 | DeBERTa DeB3RTa-base (treino + eval no val) | ~2-3 h |
| 43 | Inferencia no teste (6 metodos) | ~30-45 min |
| 43 | Ensembles + McNemar + figuras | ~5-10 min |
| **Total** | | **~10-14 horas** |

**O gargalo absoluto e o treino dos 3 modelos BERT, que representa ~90% do tempo total.**

### 4.2 Pico de consumo por recurso

| Recurso | Pico maximo | Fase critica | Limite do hardware |
|---------|-------------|-------------|-------------------|
| RAM | ~3,5-4,0 GB | TF-IDF vetorizacao | 16 GB (OK) |
| VRAM | ~2,5-2,7 GB | BERT treino (FP16) | 3,9 GB (apertado) |
| CPU (threads) | 8/8 | TF-IDF treino (BLAS) | 8 (OK) |
| GPU utilization | ~80-95% | BERT treino | 100% (OK) |
| Disco (temporario) | ~6-7 GB | Checkpoints BERT | SSD (OK) |
| Disco (permanente) | ~2,0-2,5 GB | Modelos finais | SSD (OK) |

### 4.3 Perfil de consumo ao longo do tempo

```
Fase          CPU    RAM     GPU    VRAM    Disco
────────────  ─────  ──────  ─────  ──────  ──────
01 Splits     medio  ~1 GB   idle   0       ~50 MB
11 M1 LogReg  alto   ~3 GB   idle   0       +50 MB
12 M2 SVC     alto   ~3.5GB  idle   0       +80 MB
13 M3 NB      medio  ~2.5GB  idle   0       +30 MB
21 M4a BERT   baixo  ~3 GB   ALTO   ~2.5GB  +1.7 GB
21 M4b BERT   baixo  ~3 GB   ALTO   ~2.5GB  +1.7 GB
21 M4c BERT   baixo  ~3 GB   ALTO   ~2.5GB  +1.7 GB
31 LLM HF     baixo  ~3 GB   ALTO   ~14 GB  +0 GB (pesos cache HF)
43 Eval+Ens.  medio  ~2 GB   medio  ~700MB  +100 MB
```

---

## 5. Riscos e mitigacoes

### 5.1 VRAM insuficiente durante treino BERT

**Risco:** `torch.cuda.OutOfMemoryError` — a margem de ~1,2 GB pode ser consumida por fragmentacao CUDA ou picos de alocacao.

**Mitigacoes (em ordem de preferencia):**

1. **Gradient checkpointing** — adicionar `gradient_checkpointing=True` ao `TrainingArguments`. Reduz VRAM de ativacoes em ~60%, ao custo de ~20-25% mais tempo.
2. **Reducao de `max_length`** — de 256 para 192 ou 128. Reduz ativacoes proporcionalmente, mas aumenta truncagem.
3. **Limpar cache CUDA** — chamar `torch.cuda.empty_cache()` antes de cada treino.
4. **Fechar aplicacoes** — garantir que nenhum outro processo use a GPU (o Xorg usa ~4 MB, negligivel).

### 5.2 Tempo excessivo de treino BERT

**Risco:** 10-14 horas de execucao total, com risco de interrupcao.

**Mitigacoes:**

1. **Salvar checkpoints por epoca** (ja configurado) — permite retomar treino interrompido.
2. **Treinar um modelo por vez** — editar `MODELS = [...]` em `21_bert.ipynb` para rodar um BERT por sessao.
3. **Aumentar `per_device_eval_batch_size`** — de 1 para 8, reduzindo tempo de avaliacao de ~18 min para ~3 min por epoca (economia de ~45 min por modelo).
4. **Considerar Google Colab** (GPU T4, 15 GB VRAM) para os treinos BERT se a maquina local for instavel.

### 5.3 Disco

**Risco:** Acumulo de checkpoints pode consumir ~5-7 GB.

**Mitigacao:** Usar `save_total_limit=2` nos `TrainingArguments` para manter apenas os 2 melhores checkpoints. Limpar checkpoints apos selecionar o modelo final.

### 5.4 Estabilidade termica

**Risco:** A GPU mobile (50W TDP) pode sofrer thermal throttling em sessoes longas (~4h de treino continuo), reduzindo performance em ~10-20%.

**Mitigacao:** Garantir boa ventilacao. Considerar pausas de 10 min entre treinos BERT. Monitorar temperatura com `nvidia-smi` (alvo: <85C).

---

## 6. Recomendacoes de configuracao

### 6.1 Ajustes recomendados para BertTrainingConfig

```python
# Em bert.py ou no notebook, ajustar:
BertTrainingConfig(
    per_device_eval_batch_size=8,    # atualmente 1 — subotimo
    # Considerar se houver OOM:
    # gradient_checkpointing=True,   # via TrainingArguments
)

# Em TrainingArguments, considerar adicionar:
#   save_total_limit=2,             # limitar checkpoints em disco
#   dataloader_num_workers=2,       # paralelizar carregamento de dados
```

### 6.2 Ordem de execucao sugerida

1. Executar notebooks 01, 11, 12, 13 primeiro (TF-IDF) — ~15 min total. Isso valida todo o pipeline sem depender da GPU.
2. Executar `21_bert.ipynb` para os 3 BERTs (um por sessao se necessario), verificando estabilidade.
3. Executar `31_llm_hf.ipynb` para LLMs (zero-shot + few-shot).
4. Executar `41_eda_resultados.ipynb` (inclui tabela final + McNemar pareado) e `43_ensemble.ipynb` por ultimo.

### 6.3 Monitoramento durante treino BERT

```bash
# Em terminal separado, monitorar GPU:
watch -n 5 nvidia-smi

# Monitorar RAM:
watch -n 10 free -h
```

---

## 7. Resumo executivo

| Metrica | Valor |
|---------|-------|
| **Tempo total estimado** | **10-14 horas** |
| Tempo TF-IDF (3 modelos) | ~10-15 minutos |
| Tempo BERT (3 modelos) | ~9-12 horas |
| Tempo ensembles | ~5-10 minutos |
| **Pico de RAM** | **~3,5-4,0 GB** (de 16 GB) |
| **Pico de VRAM** | **~2,5-2,7 GB** (de 3,9 GB) |
| **Disco total (permanente)** | **~2,0-2,5 GB** |
| Disco total (com checkpoints) | ~6-7 GB |
| Viabilidade no hardware atual | **Sim, com margem apertada na VRAM** |

O pipeline e viavel no hardware disponivel. O principal risco e a VRAM limitada (4 GB) durante treino BERT, que opera com margem de ~1,2 GB. As mitigacoes descritas (gradient checkpointing, aumento de eval batch size) sao suficientes para garantir execucao estavel. O tempo total de ~10-14 horas e dominado pelo treino dos 3 modelos BERT e pode ser distribuido em sessoes separadas sem perda de progresso.

---

## Adendo: orcamento da nova metodologia (search-then-evaluate)

A reformulacao introduz duas mudancas que multiplicam a carga total:

1. **Splits 80/10/10 + CV 5-fold** — cada modelo agora roda 3 regimes (`fixed_split`, `cv_5fold`, `test_set`) por tarefa (binario + multiclasse). O `cv_5fold` sozinho ja sao 5 fits.
2. **`RandomizedSearchCV` antes da avaliacao** — TF-IDF 60 trials x 5 inner folds = 300 fits adicionais por modelo por tarefa; BERT 25 trials (val unico) por modelo por tarefa.

### Carga TF-IDF total (por modelo, por tarefa)

| Etapa | Fits | Tempo (i7-11370H) |
|-------|------|-------------------|
| Search (60 trials x 5 folds) | 300 | ~30-60 min |
| `fixed_split` | 1 | ~10-30 s |
| `cv_5fold` (5 folds) | 5 | ~1-3 min |
| `test_set` | 1 | ~10-30 s |
| **Subtotal por (modelo, tarefa)** | **307** | **~35-65 min** |

Para os 3 classificadores TF-IDF x 2 tarefas: ~3-7 horas (CPU local). No Colab CPU: similar.

### Carga BERT total (por modelo, por tarefa)

| Etapa | Fits | Tempo (T4) | Tempo (A100) |
|-------|------|------------|--------------|
| Search (25 trials, val unico) | 25 | ~12-25 h | ~3-6 h |
| `fixed_split` | 1 | ~30-60 min | ~10-20 min |
| `cv_5fold` (5 folds) | 5 | ~2.5-5 h | ~50 min - 2 h |
| `test_set` | 1 | ~30-60 min | ~15-25 min |
| **Subtotal por (modelo, tarefa)** | **32** | **~16-32 h** | **~5-10 h** |

Para os 3 modelos BERT x 2 tarefas: ~96-192 h (T4) ou ~30-60 h (A100). **A100 e a recomendacao** para concluir em 1-3 dias.

### Tabela final (todos os modelos, todas as tarefas)

| Componente | Hardware | Tempo total |
|-----------|----------|-------------|
| TF-IDF (M1+M2+M3) x 2 tarefas | i7 local | 3-7 h |
| BERT (M4a+M4b+M4c) x 2 tarefas | A100 Colab | 30-60 h (1-3 dias) |
| BERT (M4a+M4b+M4c) x 2 tarefas | T4 Colab | 96-192 h (4-8 dias) |
| Ensembles + comparacao | i7 local | 10-30 min |

### Mitigacoes para reduzir o orcamento

Se o tempo total for inviavel:

1. **Reduzir `N_ITER_BERT`** de 25 para 10-15 (perde resolucao da busca, mantem o restante).
2. **Rodar so binario** (comente celulas multiclasse no notebook 21). Reduz pela metade.
3. **Rodar so 1-2 modelos BERT** (edite `MODELS = [...]`). FinBERT e DeB3RTa podem ser ablacao.
4. **`N_ITER_TFIDF`** de 60 para 30 (pouco impacto pratico — TF-IDF satura rapido).

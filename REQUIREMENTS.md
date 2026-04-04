# Requisitos Tecnicos — Economy Classifier

## 1. Proposito do repositorio

Este repositorio tem um objetivo unico e delimitado: **construir, avaliar e escolher um modelo de classificacao binaria** (`mercado` vs `outros`) para textos jornalisticos em portugues brasileiro. O corpus de referencia e a base da Folha de Sao Paulo (167.053 artigos), onde a categoria editorial `mercado` serve como ground truth.

O repositorio **nao e** o produto final da pesquisa. Ele e a bancada experimental onde multiplos metodos sao treinados sob condicoes identicas, comparados com metricas padronizadas e combinados em ensembles — ate que um modelo (ou combinacao) seja selecionado para aplicacao em dados externos. O modelo escolhido sera exportado e utilizado fora deste repositorio.

---

## 2. Modelos e configuracoes

### 2.1 Inventario de metodos

Sete metodos sao avaliados. Os seis primeiros sao supervisionados e compartilham o mesmo split de treino balanceado. O ultimo e baseado em regras e nao requer treino.

| ID | Metodo | Familia | Requer treino |
|----|--------|---------|---------------|
| M1 | TF-IDF + Regressao Logistica | Linear, discriminativo | Sim |
| M2 | TF-IDF + LinearSVC | Linear, margem maxima | Sim |
| M3 | TF-IDF + Multinomial Naive Bayes | Probabilistico, generativo | Sim |
| M4a | BERT fine-tuned (BERTimbau) | Neural, Transformer | Sim |
| M4b | BERT fine-tuned (FinBERT) | Neural, Transformer | Sim |
| M4c | BERT fine-tuned (FinBERT-PT-BR) | Neural, Transformer | Sim |
| M5 | Heuristica lexical ponderada | Baseado em regras | Nao |

### 2.2 Configuracoes dos metodos TF-IDF (M1, M2, M3)

Os tres metodos TF-IDF compartilham o mesmo vetorizador. Apenas o classificador final muda.

**Vetorizador comum:**

| Parametro | Valor | Justificativa |
|-----------|-------|---------------|
| `max_features` | 50.000 | Cobertura lexical ampla |
| `ngram_range` | (1, 2) | Unigramas + bigramas para collocations financeiras |
| `sublinear_tf` | True | Escala logaritmica contra termos dominantes |
| `min_df` | 2 | Remove hapax legomena |
| `max_df` | 0.95 | Remove termos quase universais |

**Classificadores:**

| Metodo | Classificador | Parametros especificos |
|--------|--------------|----------------------|
| M1 | `LogisticRegression` | `C=1.0`, `solver="lbfgs"`, `max_iter=1000` |
| M2 | `LinearSVC` | `C=1.0`, `dual="auto"` |
| M3 | `MultinomialNB` | `alpha=1.0` |

**Nota sobre scores continuos:** M1 e M3 produzem `predict_proba`. M2 produz `decision_function` (sem probabilidade nativa). Para metricas que exigem score continuo (AUC-ROC, votacao ponderada), o `decision_function` de M2 deve ser normalizado para [0, 1] via `CalibratedClassifierCV` ou min-max scaling no split de validacao.

### 2.3 Configuracao dos BERTs (M4a, M4b, M4c)

Tres modelos BERT sao avaliados sob condicoes identicas de fine-tuning. A unica diferenca entre eles e o checkpoint pre-treinado utilizado.

| Variante | `model_name` | Origem | Justificativa |
|----------|-------------|--------|---------------|
| M4a | `neuralmind/bert-base-portuguese-cased` | BERTimbau | Pre-treino generalista em PT-BR |
| M4b | `ProsusAI/finbert` | FinBERT | Pre-treino em textos financeiros em ingles |
| M4c | `lucas-leme/FinBERT-PT-BR` | FinBERT-PT-BR | Pre-treino financeiro adaptado para PT-BR |

**Hiperparametros compartilhados (identicos para M4a, M4b, M4c):**

| Parametro | Valor | Justificativa |
|-----------|-------|---------------|
| `max_length` | 256 | ~95% dos textos sem truncamento |
| `learning_rate` | 2e-5 | Padrao para fine-tuning (Devlin et al., 2019) |
| `batch_size` | 8 (efetivo, via gradient accumulation) | Limitacao de VRAM |
| `epochs` | 3 | Fine-tuning curto |
| `weight_decay` | 0.01 | Regularizacao L2 |
| `warmup_ratio` | 0.1 | Aquecimento linear do learning rate |
| `eval_strategy` | `"epoch"` | Avaliacao ao final de cada epoca |
| `save_strategy` | `"epoch"` | Checkpoint por epoca |
| `load_best_model_at_end` | True | Selecao automatica do melhor checkpoint |
| `metric_for_best_model` | `"f1"` | F1 como criterio de selecao |

**Early stopping:** Baseado no F1 do split de validacao. Patience de 1 epoca (com 3 epocas, isso significa que o treino para se a epoca 2 nao melhorar em relacao a epoca 1).

**Nota sobre FinBERT (M4b):** O FinBERT foi pre-treinado em textos financeiros em ingles. Embora o dominio seja relevante, o idioma difere do corpus. O fine-tuning permite avaliar se o conhecimento financeiro transfere entre linguas.

**Nota sobre FinBERT-PT-BR (M4c):** Combina dominio financeiro e idioma PT-BR. Espera-se que seja o mais adequado para a tarefa, mas a avaliacao empirica confirmara.

### 2.4 Configuracao da heuristica (M5)

| Componente | Especificacao |
|------------|---------------|
| Catalogo | 195 termos em 7 temas |
| Niveis de sinal | nuclear=4, setorial=3, contextual=2, fraco=1 |
| Penalidades de contexto | automotivo, editorial, tecnico, politico-social |
| Normalizacao | `log1p(word_count)` |
| Limiar mercado (estrito) | `score >= 2.6` AND presenca de sinal forte |
| Limiar ambiguo | `score >= 1.0` |

**Modos de avaliacao:**

| Modo | Definicao de positivo | Uso |
|------|----------------------|-----|
| Estrito | Somente banda `mercado` | Metrica principal para comparacao |
| Leniente | Bandas `mercado` + `ambiguo` | Analise de sensibilidade |

### 2.5 Estrategias de ensemble

Quatro estrategias sao avaliadas sobre as predicoes dos 7 metodos base.

| ID | Estrategia | Entrada | Calibracao |
|----|-----------|---------|------------|
| E1 | Votacao majoritaria | Predicoes binarias (0/1) | Limiares: >=4/7, >=5/7, >=6/7 |
| E2 | Votacao ponderada | Scores continuos [0,1] | Pesos = F1 de cada metodo no val |
| E3 | Stacking (meta-classificador) | Scores continuos [0,1] | LogReg treinado no val |
| E4 | Concordancia como confianca | Predicoes binarias (0/1) | Sem limiar — produz nivel de confianca |

**Regras de integridade:**

- E1-E4 sao calibrados exclusivamente no split de **validacao**.
- E3 (stacking): o meta-classificador e treinado nas predicoes que os modelos base fizeram sobre o split de validacao. Os modelos base **nao** viram o split de validacao durante treino.
- E4 nao produz uma classificacao final, mas uma tabela de contingencia (nivel de concordancia x classe real) que informa confianca.

---

## 3. Organizacao dos splits e dados

### 3.1 Estrategia de particao

```
Corpus completo (167.053 artigos)
    │
    ├── Teste (20%) ─── ~33.411 linhas ─── INTOCAVEL ate avaliacao final
    │
    └── Treino+Val (80%) ─── ~133.642 linhas
            │
            ├── Validacao (20% do restante = 16% do total) ─── ~26.728 linhas
            │
            └── Treino bruto (64% do total) ─── ~106.914 linhas
                    │
                    └── Treino balanceado (downsample) ─── ~33.552 linhas (50/50)
```

### 3.2 Parametros fixos

| Parametro | Valor | Motivo |
|-----------|-------|--------|
| `seed` | 42 | Reproducibilidade entre execucoes |
| Estratificacao | Por `label` | Preservar proporcao ~12.5% mercado em cada split |
| Balanceamento | Downsample da classe majoritaria | Somente no treino |

### 3.3 Invariantes dos splits

Estas condicoes devem ser verificadas por testes unitarios:

1. **Disjuncao total**: `treino ∩ val = ∅`, `treino ∩ teste = ∅`, `val ∩ teste = ∅`.
2. **Cobertura total**: `|treino| + |val| + |teste| = |corpus|`.
3. **Proporcao estratificada**: cada split preserva ~12.5% de `mercado` (tolerancia ±0.5pp).
4. **Balanceamento isolado**: `val` e `teste` nunca sao balanceados.
5. **Determinismo**: mesma `seed` produz indices identicos em qualquer execucao.

### 3.4 Formato de persistencia dos splits

Os splits devem ser persistidos como artefatos para garantir que todos os metodos usem exatamente os mesmos dados:

```
artifacts/splits/
    split_metadata.json     ← seed, proporcoes, contagens, hash do corpus
    train_indices.csv       ← indices do DataFrame original (coluna: index)
    val_indices.csv
    test_indices.csv
```

O `split_metadata.json` contem:

```json
{
    "seed": 42,
    "corpus_sha256": "abc123...",
    "corpus_rows": 167053,
    "train_rows": 106914,
    "val_rows": 26728,
    "test_rows": 33411,
    "train_mercado_pct": 12.55,
    "val_mercado_pct": 12.55,
    "test_mercado_pct": 12.55,
    "balanced_train_rows": 33552,
    "generated_at": "2026-04-03T...",
    "git_commit": "abc1234"
}
```

---

## 4. Controle de artefatos e reproducibilidade

### 4.1 Estrutura de artefatos

Cada execucao de treino ou avaliacao gera um diretorio de run com metadados completos:

```
artifacts/
    splits/                           ← Particoes do dataset (geradas uma unica vez)
    runs/
        {timestamp}-{stage}-{slug}/   ← Um diretorio por execucao
            run_metadata.json         ← Metadados completos do run
            README.md                 ← Resumo legivel por humanos
            model/                    ← Artefatos do modelo (quando aplicavel)
            predictions.csv           ← Predicoes no split avaliado
            metrics.json              ← Metricas computadas
            figures/                  ← Imagens geradas
    latest/
        {stage}/
            {manifest}.json           ← Ponteiro estavel para o run mais recente
```

### 4.2 Metadados de run (`run_metadata.json`)

Cada run deve registrar:

| Campo | Descricao |
|-------|-----------|
| `run_id` | Identificador unico com timestamp |
| `stage` | Tipo de operacao (ex: `tfidf-training`, `bert-training`, `evaluation`, `ensemble`) |
| `git_commit` | Hash curto do commit no momento da execucao |
| `generated_at` | Timestamp UTC |
| `parameters` | Todos os hiperparametros usados (reproduzir a execucao) |
| `inputs` | Caminhos e proveniencia dos dados de entrada |
| `outputs` | Caminhos dos artefatos gerados |
| `summary` | Metricas resumidas e contagens |
| `timing` | Tempo de treino e inferencia em segundos |

### 4.3 Rastreabilidade de linhagem

Cada artefato de saida deve apontar para seus artefatos de entrada. Isso permite reconstruir a cadeia:

```
splits → treino M1 → predicoes M1 ─┐
splits → treino M2 → predicoes M2 ─┤
splits → treino M3 → predicoes M3 ─┼─→ ensemble → avaliacao final
splits → treino M4a → predicoes M4a ─┤
splits → treino M4b → predicoes M4b ─┤
splits → treino M4c → predicoes M4c ─┤
         heuristica → predicoes M5 ─┘
```

### 4.4 Versionamento

| Item | Estrategia |
|------|-----------|
| Codigo | Git (commits com mensagens descritivas) |
| Dependencias | `uv.lock` determinista, `pyproject.toml` com versoes minimas |
| Dados brutos | Nao versionados no git (`.gitignore`), documentados em manifests TOML |
| Artefatos de modelo | Nao versionados no git, rastreados por `run_metadata.json` |
| Splits | Persistidos em `artifacts/splits/`, hash do corpus garante consistencia |
| Seed global | `42` em todos os pontos de aleatoriedade |

### 4.5 O que deve estar no `.gitignore`

```
data/
artifacts/runs/
artifacts/splits/
*.pyc
__pycache__/
.venv/
*.egg-info/
```

O que **deve** estar no git:

```
src/                    ← Codigo fonte
tests/                  ← Testes
notebooks/              ← Notebooks (sem output — usar nbstripout ou similar)
pyproject.toml          ← Dependencias
uv.lock                 ← Lock determinista
AGENTS.md               ← Personas
REQUIREMENTS.md         ← Este documento
data/manifests/         ← Descritores TOML dos datasets
```

---

## 5. Organizacao dos testes

### 5.1 Estrutura

```
tests/
    conftest.py                 ← Fixtures compartilhadas (DataFrames sinteticos, splits mockados)
    test_datasets.py            ← Splits, balanceamento, invariantes
    test_tfidf.py               ← Pipeline TF-IDF (treino, predicao, serializacao)
    test_bert.py                ← Pipeline BERT (configuracao, tokenizacao, predicao para M4a/M4b/M4c)
    test_heuristics.py          ← Scoring, bandas, cobertura
    test_evaluation.py          ← Metricas, McNemar, AUC-ROC
    test_ensemble.py            ← Votacao, stacking, concordancia
    test_integration.py         ← Fluxo completo com dados sinteticos
```

### 5.2 Tipos de teste

| Tipo | Escopo | Exemplo |
|------|--------|---------|
| Unitario | Uma funcao isolada | `compute_binary_metrics` retorna valores corretos para TP/FP/FN/TN conhecidos |
| Contrato | Interface entre modulos | `train_tfidf_classifier` retorna dict com chaves `y_pred`, `y_score`, `metrics` |
| Invariante | Propriedades que nunca devem violar | Splits sao disjuntos e cobrem todo o corpus |
| Integracao | Pipeline completo | Treinar M1 em dados sinteticos, gerar predicoes, avaliar, exportar |

### 5.3 Fixtures obrigatorias

```python
@pytest.fixture
def synthetic_corpus() -> pd.DataFrame:
    """DataFrame com ~200 linhas (25 mercado, 175 outros) para testes rapidos."""

@pytest.fixture
def synthetic_splits(synthetic_corpus) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Treino/val/teste derivados do corpus sintetico com seed=42."""

@pytest.fixture
def known_predictions() -> tuple[pd.Series, pd.Series]:
    """y_true e y_pred com valores escolhidos para metricas verificaveis manualmente."""
```

### 5.4 Testes criticos (nao podem faltar)

| Modulo | Teste | Verifica |
|--------|-------|----------|
| `datasets` | `test_split_disjunction` | Nenhum indice aparece em mais de um split |
| `datasets` | `test_split_stratification` | Proporcao de mercado em cada split esta em 12.0-13.0% |
| `datasets` | `test_split_determinism` | Duas chamadas com `seed=42` produzem indices identicos |
| `datasets` | `test_balanced_train_size` | Treino balanceado tem 50% de cada classe |
| `evaluation` | `test_metrics_known_values` | F1/precision/recall batem com calculo manual |
| `evaluation` | `test_mcnemar_identical_predictions` | Predicoes identicas → p-valor = 1.0 |
| `ensemble` | `test_majority_vote_tie` | Comportamento definido em empate (numero impar de metodos, verificar limiar) |
| `ensemble` | `test_stacking_no_leakage` | Meta-classificador treinado em val, nao em treino |
| `tfidf` | `test_pipeline_roundtrip` | Salvar e carregar pipeline produz predicoes identicas |
| `heuristics` | `test_known_terms_score` | Termos nucleares produzem score >= limiar mercado |

### 5.5 Execucao

```bash
# Todos os testes
uv run pytest

# Somente unitarios (rapidos)
uv run pytest tests/ -m "not integration"

# Com cobertura
uv run pytest --cov=economy_classifier --cov-report=term-missing
```

---

## 6. Saidas visuais para relatorios

### 6.1 Principios de geracao de figuras

1. **Toda figura e gerada por codigo.** Nenhuma figura e criada manualmente. Isso garante que o relatorio final pode ser reconstruido a partir do codigo.

2. **Formato dual.** Cada figura e salva em dois formatos:
   - **PNG** (300 DPI) para inclusao direta em documentos e visualizacao rapida.
   - **PDF** vetorial para inclusao em LaTeX sem perda de qualidade.

3. **Padrao visual consistente.** Todas as figuras usam o mesmo estilo:
   ```python
   import matplotlib.pyplot as plt
   import seaborn as sns

   plt.rcParams.update({
       "figure.figsize": (8, 5),
       "figure.dpi": 150,
       "savefig.dpi": 300,
       "savefig.bbox_inches": "tight",
       "font.size": 11,
       "axes.titlesize": 13,
       "axes.labelsize": 11,
   })
   sns.set_style("whitegrid")
   ```

4. **Nomes descritivos.** Arquivos de figura seguem o padrao:
   ```
   {run_dir}/figures/{metodo}_{tipo_de_grafico}.{png,pdf}
   ```
   Exemplo: `logreg_confusion_matrix.png`, `comparativo_f1_barplot.pdf`.

5. **Funcao de persistencia.** Toda figura e salva por uma funcao utilitaria que garante os dois formatos:
   ```python
   def save_figure(fig: Figure, path: Path, name: str) -> dict[str, Path]:
       """Salva figura em PNG e PDF. Retorna caminhos gerados."""
   ```

### 6.2 Figuras obrigatorias por metodo individual

| Figura | Descricao | Metodos |
|--------|-----------|---------|
| Matriz de confusao | Heatmap com contagens absolutas e percentuais | Todos (M1-M5) |
| Curva precision-recall | Precision vs recall em funcao do limiar | M1, M3, M4a, M4b, M4c (com `predict_proba`) |
| Curva ROC | TPR vs FPR com AUC anotada | M1, M3, M4a, M4b, M4c |
| Top termos | Barplot horizontal dos 20 termos mais discriminativos | M1 (coeficientes do LogReg) |

### 6.3 Figuras obrigatorias comparativas

| Figura | Descricao |
|--------|-----------|
| Barplot comparativo de F1 | Todos os metodos + ensembles lado a lado |
| Barplot comparativo de precision e recall | Barras agrupadas por metodo |
| Heatmap de concordancia | Cohen's Kappa entre cada par de classificadores |
| Tabela de contingencia de concordancia | Nivel de concordancia (1/7 a 7/7) vs classe real |
| Curvas PR sobrepostas | Curvas de todos os metodos com `predict_proba` no mesmo grafico |

### 6.4 Figuras opcionais (se informativas)

| Figura | Descricao |
|--------|-----------|
| Distribuicao de scores | Histograma dos scores continuos por classe real |
| Venn de erros | Quais exemplos cada metodo erra exclusivamente |
| Calibracao | Reliability diagram (frequencia observada vs probabilidade predita) |

---

## 7. Formatos de dados para processamento posterior

### 7.1 Principio

Todo dado numerico gerado pelo pipeline deve estar disponivel em formato **tabular e legivel por maquina**, independentemente das figuras. As figuras sao derivadas dos dados, nunca o contrario.

### 7.2 Formatos de saida

| Artefato | Formato | Justificativa |
|----------|---------|---------------|
| Predicoes por metodo | CSV | Uma linha por exemplo, colunas: `index`, `y_true`, `y_pred`, `y_score` |
| Metricas por metodo | JSON | Dict com precision, recall, F1, accuracy, AUC, tempo |
| Tabela comparativa | CSV | Uma linha por metodo, colunas: todas as metricas |
| Matriz de confusao | CSV | 2x2, com labels como cabeçalho |
| Concordancia pareada | CSV | Matriz NxN de Cohen's Kappa |
| Resultado de McNemar | JSON | Para cada par de metodos: chi2, p-valor, significancia |
| Metadados do run | JSON | Parametros, inputs, outputs, timing |

### 7.3 Predicoes — formato padrao

Todos os metodos devem produzir predicoes no mesmo formato CSV:

```csv
index,y_true,y_pred,y_score,method
0,1,1,0.923,logreg
1,0,0,0.087,logreg
2,1,0,0.412,logreg
```

| Coluna | Tipo | Descricao |
|--------|------|-----------|
| `index` | int | Indice original do DataFrame do corpus |
| `y_true` | int (0/1) | Label real |
| `y_pred` | int (0/1) | Predicao binaria |
| `y_score` | float [0,1] | Score continuo (probabilidade ou score normalizado) |
| `method` | str | Identificador do metodo (ex: `logreg`, `linearsvc`, `nb`, `bertimbau`, `finbert`, `finbert_ptbr`, `heuristic_strict`) |

**Para a heuristica**, `y_score` e o score heuristico normalizado para [0,1]. `y_pred` depende do modo (estrito ou leniente).

### 7.4 Metricas — formato padrao

```json
{
    "method": "logreg",
    "split": "test",
    "metrics": {
        "precision": 0.9245,
        "recall": 0.8901,
        "f1": 0.9070,
        "accuracy": 0.9734,
        "auc_roc": 0.9812
    },
    "timing": {
        "train_seconds": 12.4,
        "inference_seconds": 0.8
    },
    "n_samples": 33411,
    "n_positive": 4189,
    "n_negative": 29222
}
```

### 7.5 Tabela comparativa final

A tabela comparativa final e o artefato principal do repositorio. Deve ser salva como CSV e como JSON:

```
artifacts/runs/{run_avaliacao_final}/
    comparative_table.csv
    comparative_table.json
    figures/
        comparativo_f1_barplot.png
        comparativo_f1_barplot.pdf
```

A tabela tem uma linha por metodo/ensemble e as seguintes colunas:

| Coluna | Tipo |
|--------|------|
| `method` | str |
| `type` | str (`individual` / `ensemble`) |
| `precision` | float |
| `recall` | float |
| `f1` | float |
| `accuracy` | float |
| `auc_roc` | float ou null |
| `train_seconds` | float ou null |
| `inference_seconds` | float |
| `mcnemar_vs_best` | float (p-valor) ou null |

---

## 8. Estrutura de notebooks

### 8.1 Convencoes

1. **Notebooks sao orquestradores.** Toda logica reutilizavel esta em `src/economy_classifier/`. Notebooks importam funcoes, definem configuracao e exibem resultados.

2. **Numeracao sequencial.** Notebooks sao numerados na ordem de execucao logica. Nao precisa ser a ordem de implementacao.

3. **Celulas de cabecalho.** Todo notebook comeca com:
   - Celula markdown: titulo, objetivo, dependencias
   - Celula de imports
   - Celula de configuracao (caminhos, hiperparametros)

4. **Sem output no git.** Notebooks sao commitados sem outputs (usar `nbstripout` ou limpar antes do commit).

### 8.2 Mapa de notebooks

| Notebook | Responsabilidade | Metodos |
|----------|-----------------|---------|
| `01_preparacao_dados.ipynb` | Carga do corpus, binarizacao, geracao dos splits, persistencia | — |
| `02_tfidf_logreg.ipynb` | Treino e avaliacao de M1 no val | M1 |
| `03_tfidf_linearsvc.ipynb` | Treino e avaliacao de M2 no val | M2 |
| `04_tfidf_multinomialnb.ipynb` | Treino e avaliacao de M3 no val | M3 |
| `05_bert_bertimbau.ipynb` | Treino e avaliacao de M4a no val | M4a |
| `05b_bert_finbert.ipynb` | Treino e avaliacao de M4b no val | M4b |
| `05c_bert_finbert_ptbr.ipynb` | Treino e avaliacao de M4c no val | M4c |
| `06_heuristica.ipynb` | Avaliacao de M5 no val (estrito + leniente) | M5 |
| `07_avaliacao_comparativa.ipynb` | Comparacao no val, ensembles, avaliacao final no teste | Todos + E1-E4 |

---

## 9. Modulos Python (`src/economy_classifier/`)

### 9.1 Contratos de interface

Todos os modulos de treino devem expor funcoes que retornam predicoes no formato padrao:

```python
# Treino: retorna dict com modelo e metricas de validacao
def train_*(train_df, val_df, *, run_dir, config) -> dict

# Predicao: retorna DataFrame com colunas [y_pred, y_score]
def predict_*(texts, *, model_dir) -> pd.DataFrame

# Avaliacao: retorna dict com metricas
def evaluate_*(y_true, y_pred, y_score) -> dict
```

### 9.2 Modulos e responsabilidades

| Modulo | Funcoes principais |
|--------|--------------------|
| `datasets.py` | `build_train_val_test_split()`, `build_balanced_training_frame()` |
| `tfidf.py` | `train_tfidf_classifier()`, `load_tfidf_pipeline()`, `predict_texts()` |
| `bert.py` | `train_bert_classifier()`, `predict_texts()` |
| `heuristics.py` | `score_text()`, `classify_score()` |
| `evaluation.py` | `compute_binary_metrics()`, `compute_mcnemar_test()`, `compute_roc_auc()` |
| `ensemble.py` | `majority_vote()`, `weighted_vote()`, `train_stacking_classifier()`, `compute_agreement_matrix()` |
| `project.py` | `create_run_directory()`, `build_run_metadata()`, `persist_run_artifacts()` |
| `visualization.py` | `plot_confusion_matrix()`, `plot_pr_curve()`, `plot_roc_curve()`, `plot_comparative_barplot()`, `save_figure()` |

---

## 10. Dependencias

### 10.1 Runtime

| Pacote | Uso |
|--------|-----|
| `pandas` | Manipulacao de dados |
| `scikit-learn` | TF-IDF, LogReg, SVC, NB, metricas, splits |
| `transformers` | BERTimbau |
| `torch` | Backend BERT |
| `accelerate` | Treino distribuido BERT |
| `datasets` | Integracao HuggingFace |
| `matplotlib` | Figuras |
| `seaborn` | Estilo e heatmaps |
| `numpy` | Operacoes numericas |

### 10.2 Desenvolvimento

| Pacote | Uso |
|--------|-----|
| `pytest` | Testes |
| `pytest-cov` | Cobertura |
| `ruff` | Linting e formatacao |
| `ipykernel` | Notebooks |

---

## 11. Checklist de entrega

Antes de considerar o pipeline completo, verificar:

- [ ] Splits gerados e persistidos com metadados e hash do corpus
- [ ] 7 metodos treinados (ou avaliados) nos mesmos splits
- [ ] Predicoes de todos os metodos no split de teste em formato CSV padrao
- [ ] Metricas individuais em JSON para cada metodo
- [ ] 4 estrategias de ensemble avaliadas no teste
- [ ] Tabela comparativa final (CSV + JSON)
- [ ] Figuras obrigatorias geradas (PNG 300 DPI + PDF)
- [ ] Testes unitarios passando (`uv run pytest`)
- [ ] McNemar entre os dois melhores metodos com p-valor reportado
- [ ] `run_metadata.json` em cada diretorio de run
- [ ] Nenhum metodo viu o split de teste antes da avaliacao final

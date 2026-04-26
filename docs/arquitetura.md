# Arquitetura do Projeto

## Visao geral

O economy-classifier e um pipeline de ML reprodutivel que constroi, avalia e seleciona um classificador binario de textos (`mercado` vs `outros`) para textos jornalisticos em portugues brasileiro. O corpus de referencia e a base da Folha de Sao Paulo com 167.053 artigos, onde a categoria editorial `mercado` serve como ground truth.

O repositorio nao e o produto final da pesquisa. Ele e a bancada experimental onde multiplos metodos sao treinados sob condicoes identicas, comparados com metricas padronizadas e combinados em ensembles, ate que um modelo (ou combinacao) seja selecionado para aplicacao em dados externos.

---

## Modulos Python

O codigo-fonte esta organizado em 8 modulos dentro de `src/economy_classifier/`:

| Modulo | Responsabilidade |
|--------|-----------------|
| `project.py` | Gerenciamento de runs: criacao de diretorios, metadados, `result_card` (com `hyperparameter_search`) |
| `datasets.py` | Split estratificado 80/10/10 e `StratifiedKFold(5)` sobre o pool train+val (90%) |
| `evaluation.py` | Metricas binarias e multiclasse, teste de McNemar, AUC-ROC, matriz de confusao, `summarize_cv_metrics`, `compute_cost_metrics` |
| `tfidf.py` | 3 classificadores lineares (LogReg, CalibratedClassifierCV(LinearSVC), MultinomialNB) + variantes multiclasse (native + OvR) |
| `bert.py` | Fine-tuning de checkpoints transformer: BERTimbau, FinBERT-PT-BR, DeB3RTa-base |
| `hyperparameter_search.py` | `RandomizedSearchCV` para TF-IDF (sklearn) e busca aleatoria custom para BERT (loop com HF Trainer) |
| `ensemble.py` | 4 estrategias de ensemble: votacao majoritaria, votacao ponderada, stacking, concordancia |
| `visualization.py` | Geracao de figuras em PNG (300 DPI) e PDF vetorial |

---

## Inventario de metodos

### Metodos individuais (M1-M4c)

| ID | Metodo | Familia | Requer treino |
|----|--------|---------|---------------|
| M1 | TF-IDF + Regressao Logistica | Linear, discriminativo | Sim |
| M2 | TF-IDF + LinearSVC (calibrado) | Linear, margem maxima | Sim |
| M3 | TF-IDF + Multinomial Naive Bayes | Probabilistico, generativo | Sim |
| M4a | BERT fine-tuned (BERTimbau) | Neural, Transformer | Sim |
| M4b | BERT fine-tuned (FinBERT-PT-BR) | Neural, Transformer | Sim |
| M4c | DeBERTa fine-tuned (DeB3RTa-base) | Neural, Transformer | Sim |

Todos os metodos sao supervisionados, compartilham os mesmos splits e folds, e passam pela mesma busca de hiperparametros (RandomizedSearchCV) antes da avaliacao.

### Estrategias de ensemble (E1-E4)

| ID | Estrategia | Entrada | Calibracao |
|----|-----------|---------|------------|
| E1 | Votacao majoritaria | Predicoes binarias (0/1) | Limiares: >=4/7, >=5/7, >=6/7 |
| E2 | Votacao ponderada | Scores continuos [0,1] | Pesos = F1 de cada metodo na validacao |
| E3 | Stacking (meta-classificador) | Scores continuos [0,1] | LogisticRegression treinado na validacao |
| E4 | Concordancia como confianca | Predicoes binarias (0/1) | Sem limiar — produz nivel de confianca |

As estrategias de ensemble sao calibradas exclusivamente no split de validacao. O meta-classificador do E3 e treinado nas predicoes que os modelos base fizeram sobre a validacao, sem vazamento de dados. O E4 nao produz uma classificacao final, mas uma tabela de contingencia (nivel de concordancia x classe real).

---

## Busca de hiperparametros

Cada modelo passa por uma busca aleatoria sobre o pool train+val (90%) **antes** dos 3 regimes de avaliacao. Os melhores hiperparametros encontrados sao reusados nos 3 regimes (binario e multiclasse independentes).

| Familia | Mecanismo | Inner | Default n_iter | Espaco |
|---------|-----------|-------|----------------|--------|
| TF-IDF | `RandomizedSearchCV` (sklearn) | `StratifiedKFold(5, seed=43)` | 60 | `ngram_range`, `min_df`, `max_df`, `max_features`, `sublinear_tf`, `C`/`alpha`, `class_weight`/`fit_prior` |
| BERT | Loop custom (HF Trainer) | val unico | 25 | `learning_rate`, `per_device_train_batch_size`, `num_train_epochs`, `weight_decay`, `warmup_ratio`, `gradient_accumulation_steps` |

A assimetria — TF-IDF inner-CV vs BERT val unico — e deliberada: o custo de inner-CV em BERT seria proibitivo mesmo em A100. Esta declarada no campo `hyperparameter_search.scoring` de cada `result_card.json` e deve constar na sessao de discussao do artigo.

O log completo da busca (todos os trials, scores, duracoes) fica em `artifacts/runs/{model_id}_search_{task}/search_result.json`. O `result_card.json` carrega so o payload compacto (`best_params`, `best_score`, `n_trials`, `search_seconds`, `scoring`, `search_space`).

O inner-CV TF-IDF usa `cv_seed=43` deliberadamente diferente do `cv_folds.json` (`seed=42`) para que o regime `cv_5fold` reportado seja uma estimativa de variancia em particoes **independentes** das usadas na selecao.

---

## Fluxo de dados

```
Corpus Folha de Sao Paulo (167.053 artigos)
    |
    v
build_train_val_test_split()           # datasets.py  (80/10/10, seed=42)
    |
    +---> Teste (10%)        ~16.629 linhas  (INTOCAVEL ate test_set)
    |
    +---> Pool train+val (90%)  ~150.288 linhas
              |
              +-- build_cv_folds(n=5, seed=42)        # cv_folds.json
              |
              v
        random_search_tfidf() / random_search_bert()  # hyperparameter_search.py
              |   (busca em train+val; TF-IDF: inner-CV seed=43; BERT: val unico)
              v
        best_params (binario + multiclasse, por modelo)
              |
              v
    +--- 3 regimes por modelo (com best_params)
              fixed_split  : train -> val
              cv_5fold     : 5 folds em cv_folds.json (variancia)
              test_set     : train+val -> test FIXO
              |
              v
        result_card.json (metricas + custo + hyperparameter_search)
              |
              v
    +--- majority_vote()                   # ensemble.py (E1, no test_set)
    +--- weighted_vote()                   # ensemble.py (E2)
    +--- train_stacking_classifier()       # ensemble.py (E3, treinado em val)
    +--- compute_contingency_table()       # ensemble.py (E4)
              |
              v
         compute_binary_metrics()          # evaluation.py
         compute_mcnemar_test()            # evaluation.py
         compute_roc_auc()                 # evaluation.py
```

### Principios do fluxo

1. **Estratificacao preservada**: cada split mantem ~12.5% de `mercado` (tolerancia +/-0.5pp).
2. **Sem balanceamento no fluxo padrao**: val e teste preservam a distribuicao real. `build_balanced_training_frame` permanece para reproduzir resultados legados.
3. **Split de teste intocavel**: o split de teste so aparece no regime `test_set`.
4. **Busca antes da avaliacao**: cada modelo passa por `RandomizedSearchCV` em train+val antes dos 3 regimes. Os mesmos `best_params` sao usados nos 3.
5. **Sem vazamento**: o stacking e treinado em predicoes do split de val (nunca do treino). O test set nunca aparece na busca.
6. **Independencia de folds**: a busca TF-IDF usa inner `cv_seed=43`, o `cv_folds.json` usa `seed=42` — o regime `cv_5fold` reportado e sobre folds independentes dos usados na selecao.
7. **Determinismo**: seed=42 em todos os pontos de aleatoriedade garante reprodutibilidade.

---

## Estrutura de artefatos

Cada execucao de treino ou avaliacao gera um diretorio de run com metadados completos:

```
artifacts/
    splits/                           # Particoes do dataset (geradas uma unica vez)
        split_metadata.json           #   seed, proporcoes, contagens, hash do corpus
        train_indices.csv             #   indices do treino
        val_indices.csv               #   indices da validacao
        test_indices.csv              #   indices do teste
    runs/
        {timestamp}-{stage}-{slug}/   # Um diretorio por execucao
            run_metadata.json         #   metadados completos do run
            model/                    #   artefatos do modelo (quando aplicavel)
            predictions.csv           #   predicoes no split avaliado
            metrics.json              #   metricas computadas
            figures/                  #   imagens geradas (PNG + PDF)
    latest/
        {stage}/
            {manifest}.json           # Ponteiro estavel para o run mais recente
```

### Formato do run_metadata.json

| Campo | Descricao |
|-------|-----------|
| `run_id` | Identificador unico com timestamp (nome do diretorio) |
| `stage` | Tipo de operacao (ex: `tfidf-training`, `bert-training`, `evaluation`, `ensemble`) |
| `git_commit` | Hash curto do commit no momento da execucao |
| `generated_at` | Timestamp UTC em ISO-8601 |
| `parameters` | Todos os hiperparametros usados |
| `inputs` | Caminhos e proveniencia dos dados de entrada |
| `outputs` | Caminhos dos artefatos gerados |
| `summary` | Metricas resumidas e contagens |
| `timing` | Tempo de treino e inferencia em segundos |

### Formato padrao de predicoes (predictions.csv)

Todos os metodos produzem predicoes no mesmo formato CSV:

| Coluna | Tipo | Descricao |
|--------|------|-----------|
| `index` | int | Indice original do DataFrame do corpus |
| `y_true` | int (0/1) | Label real |
| `y_pred` | int (0/1) | Predicao binaria |
| `y_score` | float [0,1] | Score continuo (probabilidade ou score normalizado) |
| `method` | str | Identificador do metodo (ex: `logreg`, `linearsvc`, `nb`, `bertimbau`, `finbert_ptbr`, `deb3rta_base`) |

---

## Rastreabilidade de linhagem

Cada artefato de saida aponta para seus artefatos de entrada, permitindo reconstruir a cadeia completa:

```
splits --> treino M1 --> predicoes M1 --+
splits --> treino M2 --> predicoes M2 --+
splits --> treino M3 --> predicoes M3 --+-- ensemble --> avaliacao final
splits --> treino M4a --> predicoes M4a -+
splits --> treino M4b --> predicoes M4b -+
splits --> treino M4c --> predicoes M4c -+
```

---

## Saidas visuais

Toda figura e gerada por codigo e salva em dois formatos:

- **PNG** (300 DPI) para inclusao direta em documentos e visualizacao rapida.
- **PDF** vetorial para inclusao em LaTeX sem perda de qualidade.

Figuras obrigatorias por metodo individual:

| Figura | Descricao | Metodos |
|--------|-----------|---------|
| Matriz de confusao | Heatmap com contagens | Todos (M1-M4c) |
| Curva precision-recall | Precision vs recall | M1, M3, M4a, M4b, M4c |
| Curva ROC | TPR vs FPR com AUC | M1, M3, M4a, M4b, M4c |
| Top termos | Barplot dos 20 termos mais discriminativos | M1 |

Figuras comparativas:

| Figura | Descricao |
|--------|-----------|
| Barplot comparativo de F1 | Todos os metodos + ensembles lado a lado |
| Barplot de precision e recall | Barras agrupadas por metodo |
| Heatmap de concordancia | Cohen's Kappa entre cada par de classificadores |
| Tabela de contingencia | Nivel de concordancia (1/7 a 7/7) vs classe real |
| Curvas PR sobrepostas | Curvas de todos os metodos com `predict_proba` |

---

## Versionamento

| Item | Estrategia |
|------|-----------|
| Codigo | Git |
| Dependencias | `uv.lock` determinista, `pyproject.toml` com versoes minimas |
| Dados brutos | Nao versionados no git (`.gitignore`), documentados em manifests TOML |
| Artefatos de modelo | Nao versionados no git, rastreados por `run_metadata.json` |
| Splits | Persistidos em `artifacts/splits/`, hash do corpus garante consistencia |
| Seed global | `42` em todos os pontos de aleatoriedade |

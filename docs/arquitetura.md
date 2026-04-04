# Arquitetura do Projeto

## Visao geral

O economy-classifier e um pipeline de ML reprodutivel que constroi, avalia e seleciona um classificador binario de textos (`mercado` vs `outros`) para textos jornalisticos em portugues brasileiro. O corpus de referencia e a base da Folha de Sao Paulo com 167.053 artigos, onde a categoria editorial `mercado` serve como ground truth.

O repositorio nao e o produto final da pesquisa. Ele e a bancada experimental onde multiplos metodos sao treinados sob condicoes identicas, comparados com metricas padronizadas e combinados em ensembles, ate que um modelo (ou combinacao) seja selecionado para aplicacao em dados externos.

---

## Modulos Python

O codigo-fonte esta organizado em 8 modulos dentro de `src/economy_classifier/`:

| Modulo | Responsabilidade |
|--------|-----------------|
| `project.py` | Gerenciamento de runs: criacao de diretorios, metadados, persistencia de artefatos |
| `datasets.py` | Split estratificado 3-way (64/16/20) e balanceamento por downsample |
| `evaluation.py` | Metricas binarias (precision, recall, F1, accuracy), teste de McNemar, AUC-ROC |
| `heuristics.py` | Lexico ponderado com 195 termos, 7 temas, penalidades contextuais |
| `tfidf.py` | 3 classificadores lineares: LogisticRegression, CalibratedClassifierCV(LinearSVC), MultinomialNB |
| `bert.py` | Fine-tuning de 3 checkpoints BERT: BERTimbau, FinBERT, FinBERT-PT-BR |
| `ensemble.py` | 4 estrategias de ensemble: votacao majoritaria, votacao ponderada, stacking, concordancia |
| `visualization.py` | Geracao de figuras em PNG (300 DPI) e PDF vetorial |

---

## Inventario de metodos

### Metodos individuais (M1-M5)

| ID | Metodo | Familia | Requer treino |
|----|--------|---------|---------------|
| M1 | TF-IDF + Regressao Logistica | Linear, discriminativo | Sim |
| M2 | TF-IDF + LinearSVC (calibrado) | Linear, margem maxima | Sim |
| M3 | TF-IDF + Multinomial Naive Bayes | Probabilistico, generativo | Sim |
| M4a | BERT fine-tuned (BERTimbau) | Neural, Transformer | Sim |
| M4b | BERT fine-tuned (FinBERT) | Neural, Transformer | Sim |
| M4c | BERT fine-tuned (FinBERT-PT-BR) | Neural, Transformer | Sim |
| M5 | Heuristica lexical ponderada | Baseado em regras | Nao |

Os seis primeiros metodos sao supervisionados e compartilham o mesmo split de treino balanceado. O M5 e baseado em regras e nao requer treino.

### Estrategias de ensemble (E1-E4)

| ID | Estrategia | Entrada | Calibracao |
|----|-----------|---------|------------|
| E1 | Votacao majoritaria | Predicoes binarias (0/1) | Limiares: >=4/7, >=5/7, >=6/7 |
| E2 | Votacao ponderada | Scores continuos [0,1] | Pesos = F1 de cada metodo na validacao |
| E3 | Stacking (meta-classificador) | Scores continuos [0,1] | LogisticRegression treinado na validacao |
| E4 | Concordancia como confianca | Predicoes binarias (0/1) | Sem limiar — produz nivel de confianca |

As estrategias de ensemble sao calibradas exclusivamente no split de validacao. O meta-classificador do E3 e treinado nas predicoes que os modelos base fizeram sobre a validacao, sem vazamento de dados. O E4 nao produz uma classificacao final, mas uma tabela de contingencia (nivel de concordancia x classe real).

---

## Fluxo de dados

```
Corpus Folha de Sao Paulo (167.053 artigos)
    |
    v
build_train_val_test_split()           # datasets.py
    |
    +---> Teste (20%)  ~33.411 linhas  (INTOCAVEL ate avaliacao final)
    |
    +---> Validacao (16%)  ~26.728 linhas
    |
    +---> Treino bruto (64%)  ~106.914 linhas
              |
              v
         build_balanced_training_frame()    # datasets.py
              |
              v
         Treino balanceado  ~33.552 linhas (50/50)
              |
              v
    +--- train_tfidf_classifier()          # tfidf.py  (M1, M2, M3)
    +--- train_bert_classifier()           # bert.py   (M4a, M4b, M4c)
    +--- score_text() + classify_score()   # heuristics.py (M5)
              |
              v
         Predicoes no split de validacao
              |
              v
    +--- majority_vote()                   # ensemble.py (E1)
    +--- weighted_vote()                   # ensemble.py (E2)
    +--- train_stacking_classifier()       # ensemble.py (E3)
    +--- compute_contingency_table()       # ensemble.py (E4)
              |
              v
         Avaliacao final no split de teste
              |
              v
         compute_binary_metrics()          # evaluation.py
         compute_mcnemar_test()            # evaluation.py
         compute_roc_auc()                 # evaluation.py
```

### Principios do fluxo

1. **Estratificacao preservada**: cada split mantem ~12.5% de `mercado` (tolerancia +/-0.5pp).
2. **Balanceamento isolado**: apenas o split de treino e balanceado. Validacao e teste nunca sao balanceados.
3. **Split de teste intocavel**: o split de teste so e utilizado na avaliacao final comparativa.
4. **Sem vazamento**: o meta-classificador do stacking e treinado em predicoes do split de validacao, nunca do treino.
5. **Determinismo**: seed=42 em todos os pontos de aleatoriedade garante reprodutibilidade.

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
| `method` | str | Identificador do metodo (ex: `logreg`, `linearsvc`, `nb`, `bertimbau`, `finbert`, `finbert_ptbr`, `heuristic_strict`) |

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
           heuristica --> predicoes M5 --+
```

---

## Saidas visuais

Toda figura e gerada por codigo e salva em dois formatos:

- **PNG** (300 DPI) para inclusao direta em documentos e visualizacao rapida.
- **PDF** vetorial para inclusao em LaTeX sem perda de qualidade.

Figuras obrigatorias por metodo individual:

| Figura | Descricao | Metodos |
|--------|-----------|---------|
| Matriz de confusao | Heatmap com contagens | Todos (M1-M5) |
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

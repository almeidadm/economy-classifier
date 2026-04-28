# Economy Classifier

Pipeline reprodutivel para classificacao binaria de textos jornalisticos em portugues brasileiro: **mercado** vs **outros**.

O corpus de referencia e a base da [Folha de Sao Paulo](https://www.kaggle.com/datasets/marlesson/news-of-the-site-folhauol) (167 mil artigos), onde a categoria editorial `mercado` serve como ground truth. Sete metodos de classificacao sao treinados sob condicoes identicas, comparados com metricas padronizadas e combinados em estrategias de ensemble.

Este repositorio e a bancada experimental de uma **dissertacao de mestrado** sobre classificacao de noticias economicas e financeiras.

## Metodos avaliados

| ID | Metodo | Familia |
|----|--------|---------|
| M1 | TF-IDF + Regressao Logistica | Linear, discriminativo |
| M2 | TF-IDF + LinearSVC | Linear, margem maxima |
| M3 | TF-IDF + Multinomial Naive Bayes | Probabilistico, generativo |
| M4a | BERT fine-tuned (BERTimbau) | Neural, Transformer |
| M4b | BERT fine-tuned (FinBERT-PT-BR) | Neural, Transformer |
| M4c | DeBERTa fine-tuned (DeB3RTa-base) | Neural, Transformer |

## Estrutura do repositorio

```
economy-classifier/
    src/economy_classifier/     # Modulos reutilizaveis
        datasets.py             #   Splits 80/10/10 + StratifiedKFold(5) sobre o pool train+val
        tfidf.py                #   Pipeline TF-IDF (LogReg, LinearSVC, NB) + variantes multiclasse
        bert.py                 #   Fine-tuning Transformer (BERTimbau, FinBERT-PT-BR, DeB3RTa)
        hyperparameter_search.py #   RandomizedSearchCV (TF-IDF) + busca custom (BERT)
        evaluation.py           #   Metricas binarias e multiclasse, McNemar, AUC-ROC
        ensemble.py             #   Votacao, stacking, concordancia
        project.py              #   Runs, artefatos, result_card
        visualization.py        #   Figuras em PNG 300 DPI + PDF
    tests/                      # 200 testes (pytest)
    notebooks/                  # Orquestracao do pipeline (numeracao por faixa)
        01_preparacao_dados     #   Dados: carga, binarizacao, splits 80/10/10, cv_folds.json
        11_tfidf_logreg         #   TF-IDF M1: search + 6 regimes (binario/multi x fixed/cv/test)
        12_tfidf_linearsvc      #   TF-IDF M2: search + 6 regimes
        13_tfidf_multinomialnb  #   TF-IDF M3: search + 6 regimes
        21_bert                 #   BERT M4a/M4b/M4c: 6 regimes (Google Colab L4/A100)
        31_llm_hf               #   LLMs Qwen/Mistral: zero-shot + few-shot, binario + multi
        41_eda_resultados       #   EDA dos result_cards e predictions
        42_comparacao           #   (reservado) Tabela final do artigo + McNemar
        43_ensemble             #   Voting + stacking sobre os 4 modelos base
        91_smoke_multiclasse_tfidf   # Smoke da Fase 1 (auditoria) — TF-IDF native vs OvR
    scripts/                    # Utilidades
        colab_pack.py           #   Empacotar splits para upload ao Colab
        colab_unpack.py         #   Integrar resultados do Colab localmente
    docs/                       # Documentacao
        arquitetura.md          #   Estrutura de modulos e fluxo de dados
        estimativa_recursos_computacionais.md
        guia_colab.md           #   Passo a passo para treino BERT no Colab
    REQUIREMENTS.md             # Especificacao tecnica completa
    AGENTS.md                   # Personas para auditoria metodologica
```

## Quickstart

### Requisitos

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/) (gerenciador de pacotes)
- GPU com CUDA (opcional para BERT local; recomendado Google Colab)

### Instalacao

```bash
git clone git@github.com:almeidadm/economy-classifier.git
cd economy-classifier
uv sync --all-extras
```

### Testes

```bash
uv run pytest                   # todos (96 testes)
uv run pytest -m "not slow"     # somente rapidos
```

### Pipeline local (TF-IDF)

1. Obtenha o dataset da Folha de Sao Paulo e coloque em `data/news-of-the-site-folhauol/articles.csv`
2. Execute os notebooks na ordem:

```bash
# 01: gerar splits
# 11-13: treinar M1, M2, M3 (TF-IDF)
```

### Pipeline BERT (Google Colab)

Os modelos BERT sao treinados no Google Colab com **GPU A100** (necessaria para o orcamento da busca de hiperparametros). Em T4 multiplique tempos por 3-5x ou reduza `N_ITER_BERT`/`MODELS`. Veja o [guia completo](docs/guia_colab.md).

```bash
# 1. Empacotar splits para upload
uv run python scripts/colab_pack.py

# 2. Upload colab_splits.zip para Google Drive
# 3. Abrir notebooks/21_bert.ipynb no Colab (Runtime > A100)
# 4. Executar (search + 6 regimes x 3 modelos ~ 1-3 dias em A100)

# 5. Baixar resultados e integrar
uv run python scripts/colab_unpack.py colab_bert_results.zip
```

## Protocolo experimental

- **Splits**: treino (80%), validacao (10%), teste (10%) — estratificados, seed=42. Test fixo de 10% **nunca** e usado para selecao.
- **Cross-validation**: `StratifiedKFold(5, seed=42)` sobre o pool train+val (90%), persistida em `artifacts/splits/cv_folds.json`.
- **Balanceamento**: nenhum no fluxo padrao (val e teste preservam ~12,5% de mercado).
- **Busca de hiperparametros**: `RandomizedSearchCV` por modelo, em train+val (90%), antes dos 3 regimes de avaliacao.
  - TF-IDF: 60 trials, inner `StratifiedKFold(5, seed=43)` (folds independentes do `cv_folds.json`).
  - BERT: 25 trials, val unico como inner (custo de inner-CV proibitivo). Asimetria declarada no `result_card`.
- **3 regimes por modelo**: `fixed_split` (treina train, avalia val), `cv_5fold` (5 folds com best_params), `test_set` (treina train+val, avalia teste FIXO).
- **Metrica primaria**: F1-score (binario) ou macro-F1 (multiclasse). Accuracy e enganosa.
- **Comparacao**: McNemar test entre pares no test_set.
- **Ensembles**: votacao majoritaria, votacao ponderada (F1), stacking (meta-LogReg), concordancia.
- **Result card**: cada regime emite `result_card.json` com metricas + custo (`train_seconds`, `inference_seconds`, `model_size_mb`, `n_parameters`, `hardware`) + payload da busca (`hyperparameter_search`).
- **Reproducibilidade**: seed=42 em todos os pontos, `uv.lock` determinista, artefatos com metadados.

## Documentacao

| Documento | Descricao |
|-----------|-----------|
| [REQUIREMENTS.md](REQUIREMENTS.md) | Especificacao tecnica completa (modelos, splits, metricas, artefatos) |
| [docs/arquitetura.md](docs/arquitetura.md) | Estrutura de modulos e fluxo de dados |
| [docs/estimativa_recursos_computacionais.md](docs/estimativa_recursos_computacionais.md) | Analise de consumo de CPU, RAM, GPU e tempo |
| [docs/guia_colab.md](docs/guia_colab.md) | Guia passo a passo para treino BERT no Colab |

## Licenca

Projeto academico — uso restrito a fins de pesquisa.

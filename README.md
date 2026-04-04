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
| M4b | BERT fine-tuned (FinBERT) | Neural, Transformer |
| M4c | BERT fine-tuned (FinBERT-PT-BR) | Neural, Transformer |
| M5 | Heuristica lexical ponderada | Baseado em regras |

## Estrutura do repositorio

```
economy-classifier/
    src/economy_classifier/     # Modulos reutilizaveis
        datasets.py             #   Splits 3-way estratificados + balanceamento
        tfidf.py                #   Pipeline TF-IDF (LogReg, LinearSVC, NB)
        bert.py                 #   Fine-tuning BERT (BERTimbau, FinBERT, FinBERT-PT-BR)
        heuristics.py           #   Scoring heuristico (195 termos, 7 temas)
        evaluation.py           #   Metricas, McNemar, AUC-ROC
        ensemble.py             #   Votacao, stacking, concordancia
        project.py              #   Runs, artefatos, metadados
        visualization.py        #   Figuras em PNG 300 DPI + PDF
    tests/                      # 96 testes (pytest)
    notebooks/                  # Orquestracao do pipeline
        01_preparacao_dados     #   Carga, binarizacao, splits, persistencia
        02_tfidf_logreg         #   M1 treino + avaliacao
        03_tfidf_linearsvc      #   M2 treino + avaliacao
        04_tfidf_multinomialnb  #   M3 treino + avaliacao
        05_bert_colab           #   M4a/M4b/M4c treino (Google Colab)
        06_heuristica           #   M5 avaliacao (estrito + leniente)
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

### Pipeline local (TF-IDF + heuristica)

1. Obtenha o dataset da Folha de Sao Paulo e coloque em `data/news-of-the-site-folhauol/articles.csv`
2. Execute os notebooks na ordem:

```bash
# 01: gerar splits
# 02-04: treinar M1, M2, M3
# 06: avaliar M5
```

### Pipeline BERT (Google Colab)

Os modelos BERT sao treinados no Google Colab (GPU T4) para viabilizar o treino com VRAM limitada localmente. Veja o [guia completo](docs/guia_colab.md).

```bash
# 1. Empacotar splits para upload
uv run python scripts/colab_pack.py

# 2. Upload colab_splits.zip para Google Drive
# 3. Abrir notebooks/05_bert_colab.ipynb no Colab
# 4. Executar treino (~2-4h com GPU T4)

# 5. Baixar resultados e integrar
uv run python scripts/colab_unpack.py colab_bert_results.zip
```

## Protocolo experimental

- **Splits**: treino (64%), validacao (16%), teste (20%) — estratificados, seed=42
- **Balanceamento**: downsample da classe majoritaria somente no treino
- **Metrica primaria**: F1-score (accuracy e enganosa com 87,5% de classe majoritaria)
- **Comparacao**: McNemar test entre pares de classificadores
- **Ensembles**: votacao majoritaria, votacao ponderada (F1), stacking (meta-LogReg), concordancia
- **Reproducibilidade**: seed=42 em todos os pontos de aleatoriedade, artefatos com metadados

## Documentacao

| Documento | Descricao |
|-----------|-----------|
| [REQUIREMENTS.md](REQUIREMENTS.md) | Especificacao tecnica completa (modelos, splits, metricas, artefatos) |
| [docs/arquitetura.md](docs/arquitetura.md) | Estrutura de modulos e fluxo de dados |
| [docs/estimativa_recursos_computacionais.md](docs/estimativa_recursos_computacionais.md) | Analise de consumo de CPU, RAM, GPU e tempo |
| [docs/guia_colab.md](docs/guia_colab.md) | Guia passo a passo para treino BERT no Colab |

## Licenca

Projeto academico — uso restrito a fins de pesquisa.

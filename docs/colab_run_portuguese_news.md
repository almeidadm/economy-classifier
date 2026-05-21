# Colab — Avaliação OOD no PortugueseNewsDataset

Roteiro de execução para a avaliação OOD do classificador binário `mercado` (Folha) sobre a categoria *"Economia e negócios"* do PortugueseNewsDataset (Klaifer/WikiNotícias, PLOS ONE 2024).

Esta avaliação **substitui parcialmente** a leitura de `Fake.Br` e `FakeRecogna` quando o objetivo é validar o construto de tópico — aqui a label OOD é diretamente comparável (não é proxy via veracidade ou portal).

## Pré-requisitos

- `colab_portuguese_news.zip` enviado para `<DRIVE>/economy-classifier/colab_portuguese_news.zip` (~11 MB). Gerar localmente com `uv run python scripts/colab_pack_portuguese_news.py`.
- Os 6 modelos binários treinados em `<DRIVE>/economy-classifier/runs/<model_id>_binary_test_set/model/`, conforme convenção dos NBs 21 / 11–13:
  - `bert_bertimbau_binary_test_set/model/`
  - `bert_finbert_ptbr_binary_test_set/model/`
  - `bert_deb3rta_base_binary_test_set/model/`
  - `tfidf_logreg_binary_test_set/model/`
  - `tfidf_linearsvc_binary_test_set/model/`
  - `tfidf_nb_binary_test_set/model/`
- Repositório clonado em `/content/economy-classifier/` (mesmo padrão do notebook 45).

## Passos no Colab

### 1. Setup do ambiente

```python
from google.colab import drive
drive.mount("/content/drive")

import os, sys, subprocess
from pathlib import Path

REPO_DIR = Path("/content/economy-classifier")
DRIVE_BASE = Path("/content/drive/MyDrive/economy-classifier")
RUNS_DIR_DRIVE = DRIVE_BASE / "runs"

# Clonar/atualizar repo (se ainda não estiver presente)
if not REPO_DIR.exists():
    subprocess.run(
        ["git", "clone", "https://github.com/<seu-usuario>/economy-classifier.git", str(REPO_DIR)],
        check=True,
    )
else:
    subprocess.run(["git", "-C", str(REPO_DIR), "pull"], check=True)

# Instalar deps mínimas (transformers + sklearn + pandas já vêm com o runtime Colab GPU)
subprocess.run(["pip", "install", "-q", "joblib", "mwparserfromhell"], check=True)

sys.path.insert(0, str(REPO_DIR / "src"))
sys.path.insert(0, str(REPO_DIR / "scripts"))
```

### 2. Extrair o dataset

```python
import zipfile

DATA_ROOT = Path("/content/ood_data/portuguese_news_wikinotices")
zip_path = DRIVE_BASE / "colab_portuguese_news.zip"
assert zip_path.exists(), f"Falta {zip_path}. Rode scripts/colab_pack_portuguese_news.py local e faça upload."

DATA_ROOT.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(zip_path) as zf:
    zf.extractall("/content/ood_data")

print(f"Extraído: {sorted(p.name for p in DATA_ROOT.iterdir())}")
```

### 3. Rodar inferência sobre os 6 modelos

```python
import evaluate_portuguese_news as pn
from economy_classifier.project import compute_artifact_size_mb

OUTPUT_ROOT = RUNS_DIR_DRIVE  # cards saem direto no Drive
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

MODELS = [
    ("bert_bertimbau",     "bert_bertimbau_binary_test_set"),
    ("bert_finbert_ptbr",  "bert_finbert_ptbr_binary_test_set"),
    ("bert_deb3rta_base",  "bert_deb3rta_base_binary_test_set"),
    ("tfidf_logreg",       "tfidf_logreg_binary_test_set"),
    ("tfidf_linearsvc",    "tfidf_linearsvc_binary_test_set"),
    ("tfidf_nb",           "tfidf_nb_binary_test_set"),
]

PARTITION = "full"  # 9135 docs; troque para "test" (914) para reproduzir paper Klaifer
df = pn.load_portuguese_news(DATA_ROOT, partition=PARTITION)
print(f"PortugueseNewsDataset partition={PARTITION}: n={len(df)}, "
      f"prevalência positiva={df['y_true_binary'].mean():.4f}")

for model_id, run_subdir in MODELS:
    model_dir = RUNS_DIR_DRIVE / run_subdir / "model"
    if not model_dir.exists():
        print(f"  SKIP {model_id}: {model_dir} não existe.")
        continue

    model_type = pn.detect_model_type(model_dir)
    print(f"  Avaliando {model_id} ({model_type})...")

    texts = df["text"].tolist()
    if model_type == "bert":
        probs, classes, inf_s, info = pn.predict_bert(
            texts, model_dir, batch_size=32, max_length=128,
        )
    else:
        probs, classes, inf_s, info = pn.predict_tfidf(texts, model_dir)

    out_dir = pn.evaluate_level1_binary(
        df=df, probs=probs, classes=classes,
        model_id=model_id, model_type=model_type,
        output_root=OUTPUT_ROOT,
        inference_seconds=inf_s,
        model_size_mb=round(compute_artifact_size_mb(model_dir), 3),
        max_length=128,
        n_parameters=info["n_parameters"],
        hardware=info["hardware"],
        partition=PARTITION,
    )
    print(f"    -> {out_dir}")
```

Tempo esperado em A100/T4: ~1–2 min/modelo BERT (9135 docs × 128 tokens) e <30s/modelo TF-IDF. Total ~7–10 min para os 6.

### 4. Empacotar resultados para baixar

Use o snippet existente em `scripts/colab_pack_results.py` (cole o corpo numa célula). Vai gerar `runs_cards_<timestamp>.zip` com `result_card.json` + `predictions.csv` dos 6 novos diretórios `*_binary_portuguese_news_wikinotices_ood/`. Baixe o zip pelo navegador.

### 5. Local — desempacotar

```bash
uv run python scripts/colab_unpack_streaming.py --delete-after
```

Os cards aparecerão em `artifacts/runs/<model_id>_binary_portuguese_news_wikinotices_ood/`.

## Output esperado

Cada um dos 6 diretórios terá:
- `result_card.json` — schema padrão do projeto, com `config.domain='portuguese_news_wikinotices'`, `config.level='1_binary_full_corpus'`, `config.partition='full'`.
- `predictions.csv` — 9135 linhas com `document_id`, `category`, `y_true`, `y_pred`, `y_score`, `method`.

## Reconstrução local (referência)

Para reconstruir os JSONs do zero (já feito uma vez):

```bash
cd /tmp && git clone https://github.com/Klaifer/PortugueseNewsDataset.git
cd PortugueseNewsDataset
mkdir -p content/raw && cd content/raw
curl -sL -o dump.xml.bz2 https://archive.org/download/ptwikinews-20220401/ptwikinews-20220401-pages-meta-current.xml.bz2
bunzip2 -k dump.xml.bz2
cd ../..
uv run --with beautifulsoup4 --with lxml --with mwparserfromhell python extractor.py \
    --input content/raw/dump.xml --output content/json/wikinews_full.json
python seletor.py --input content/json/wikinews_full.json \
    --output content/json/wikinews_categories.json \
    --categories 'Desporto' 'Crime, Direito e Justiça' 'Saúde' 'Economia e negócios' 'Política'
uv run --with scikit-learn python train_split.py \
    --input content/json/wikinews_categories.json \
    --splitfile content/json/split_ids.csv \
    --operation apply \
    --train content/json/wikinews_train.json \
    --test content/json/wikinews_test.json
# Copiar para data/portuguese_news_wikinotices/ do economy-classifier
```

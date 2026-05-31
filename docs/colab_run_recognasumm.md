# Colab — Avaliação OOD no RecognaSumm

Roteiro de execução para a avaliação OOD do classificador binário `mercado` (Folha) sobre a categoria *"Economia"* do RecognaSumm (Paiola et al., PROPOR 2024).

Diferencial em relação aos outros 3 OODs:
- **RecognaSumm** é jornalismo comercial multi-veículo (G1, CNN Brasil, UOL, Folha, etc.) — registro próximo ao treino e controla efeito único-veículo.
- **PortugueseNewsDataset** é WikiNotícias colaborativo (registro enciclopédico).
- **Fake.Br / FakeRecogna** são corpora de fake news (label proxy via veracidade ou portal).

## Pré-requisitos

- `colab_ood_data.zip` (zip unificado dos 4 corpora OOD) em `<DRIVE>/economy-classifier/`. Gerar com `uv run python scripts/colab_pack_ood_data.py` — inclui `recognasumm/test.jsonl` junto com FB+FR+PN.
- 6 modelos binários treinados em `<DRIVE>/economy-classifier/runs/<model_id>_binary_test_set/model/` (mesmos do `colab_run_portuguese_news.md`).
- Repositório clonado em `/content/economy-classifier/`.

## Passos no Colab

### 1. Setup (idem ao colab_run_portuguese_news.md)

```python
from google.colab import drive
drive.mount("/content/drive")

import os, sys, subprocess
from pathlib import Path

REPO_DIR = Path("/content/economy-classifier")
DRIVE_BASE = Path("/content/drive/MyDrive/economy-classifier")
RUNS_DIR_DRIVE = DRIVE_BASE / "runs"

if not REPO_DIR.exists():
    subprocess.run(
        ["git", "clone", "https://github.com/<seu-usuario>/economy-classifier.git", str(REPO_DIR)],
        check=True,
    )
else:
    subprocess.run(["git", "-C", str(REPO_DIR), "pull"], check=True)

subprocess.run(["pip", "install", "-q", "joblib"], check=True)

sys.path.insert(0, str(REPO_DIR / "src"))
sys.path.insert(0, str(REPO_DIR / "scripts"))
```

### 2. Extrair o RecognaSumm

```python
import zipfile

DATA_ROOT = Path("/content/ood_data/recognasumm")
zip_path = DRIVE_BASE / "colab_ood_data.zip"
assert zip_path.exists(), f"Falta {zip_path}. Rode scripts/colab_pack_ood_data.py local e faça upload."

DATA_ROOT.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(zip_path) as zf:
    zf.extractall("/content/ood_data")

print(f"Extraído: {sorted(p.name for p in DATA_ROOT.iterdir())}")
```

> Em uso normal você não precisa desse passo isolado — o notebook `45_ood_evaluation.ipynb` já extrai `colab_ood_data.zip` na seção 3 e disponibiliza os 4 corpora simultaneamente. Esta seção existe só para auditar RecognaSumm isoladamente fora do notebook.

### 3. Rodar inferência sobre os 6 modelos

```python
import evaluate_recognasumm as rs
from economy_classifier.project import compute_artifact_size_mb

OUTPUT_ROOT = RUNS_DIR_DRIVE
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

MODELS = [
    ("bert_bertimbau",     "bert_bertimbau_binary_test_set"),
    ("bert_finbert_ptbr",  "bert_finbert_ptbr_binary_test_set"),
    ("bert_deb3rta_base",  "bert_deb3rta_base_binary_test_set"),
    ("tfidf_logreg",       "tfidf_logreg_binary_test_set"),
    ("tfidf_linearsvc",    "tfidf_linearsvc_binary_test_set"),
    ("tfidf_nb",           "tfidf_nb_binary_test_set"),
]

PARTITION = "test"  # 27.055 docs. validation tambem ~95MB; train ~289MB.
df = rs.load_recognasumm(DATA_ROOT, partition=PARTITION)
print(f"RecognaSumm partition={PARTITION}: n={len(df)}, "
      f"prevalência positiva={df['y_true_binary'].mean():.4f}")

for model_id, run_subdir in MODELS:
    model_dir = RUNS_DIR_DRIVE / run_subdir / "model"
    if not model_dir.exists():
        print(f"  SKIP {model_id}: {model_dir} não existe.")
        continue

    model_type = rs.detect_model_type(model_dir)
    print(f"  Avaliando {model_id} ({model_type})...")

    texts = df["text"].tolist()
    if model_type == "bert":
        probs, classes, inf_s, info = rs.predict_bert(
            texts, model_dir, batch_size=32, max_length=128,
        )
    else:
        probs, classes, inf_s, info = rs.predict_tfidf(texts, model_dir)

    out_dir = rs.evaluate_level1_binary(
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

Tempo esperado em A100/T4: ~3–4 min/modelo BERT (27k docs × 128 tokens) e <1min/modelo TF-IDF. Total ~15–20 min para os 6.

### 4. Empacotar resultados (idem)

Use `scripts/colab_pack_results.py` para gerar `runs_cards_<timestamp>.zip`. Os 6 novos diretórios serão `*_binary_recognasumm_ood/`.

### 5. Local — desempacotar

```bash
uv run python scripts/colab_unpack_streaming.py --delete-after
```

## Output esperado

Cada um dos 6 diretórios terá:
- `result_card.json` — schema padrão, com `config.domain='recognasumm_propor2024'`, `config.level='1_binary_full_corpus'`, `config.partition='test'`.
- `predictions.csv` — 27.055 linhas com `document_id`, `category`, `autor`, `y_true`, `y_pred`, `y_score`, `method`.

## Reconstrução local (referência)

```bash
mkdir -p data/recognasumm && cd data/recognasumm
curl -sL -o test.jsonl https://huggingface.co/datasets/recogna-nlp/recognasumm/resolve/main/test.jsonl
# (opcional) train.jsonl, validation.jsonl — 95-289 MB cada
```

## Combinando os 4 OODs no relatório

Após desempacotar, ambos `data/portuguese_news_wikinotices/` e `data/recognasumm/` estarão prontos, e `artifacts/runs/` terá os 6 × 4 = 24 cards OOD (Fake.Br, FakeRecogna, PortugueseNewsDataset, RecognaSumm). O notebook `45_ood_evaluation.ipynb` agrega cards por `config.domain` automaticamente.

Comparações cross-corpus interessantes:
| Corpus | Prevalência `mercado`/`Economia` | Registro | Veracidade |
|---|---:|---|---|
| Folha (treino) | 12.5% | Jornalismo comercial | Verdadeiro |
| PortugueseNewsDataset | 12.60% | Wiki colaborativo | Verdadeiro |
| **RecognaSumm** | **9.30%** | Jornalismo multi-veículo | Verdadeiro |
| FakeRecogna UOL balanced | ~50% | Multi-portal UOL | Misto |
| FakeRecogna full | 2.63% | Multi-portal | Misto |
| Fake.Br full | 0.61% | Multi-veículo | Misto |

A linha que mais valor agrega para a Manchete D do paper é **RecognaSumm**: prevalência próxima ao treino, registro idem, multi-publisher. Se FinBERT mantiver a vantagem aqui, a tese "domínio do pretraining ajuda quando o downstream é jornalismo financeiro" ganha sua melhor evidência.

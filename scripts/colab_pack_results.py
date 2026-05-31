"""Cell para colar em qualquer notebook Colab e baixar so os artefatos pequenos.

Empacota apenas o whitelist usado por scripts/colab_unpack_streaming.py
(result_card.json, predictions.csv, etc.) varrendo artifacts/runs/ no Drive
e gera um unico zip baixado pelo navegador. Pesos de BERT, checkpoints e
tokenizers ficam de fora.

Como usar:
    1. Ajuste RUNS_DIR para apontar para a pasta runs/ no seu Drive.
    2. Cole o corpo abaixo em uma celula e execute.
    3. Mande o zip resultante para o repositorio local.
"""

# === COLE A PARTIR DAQUI EM UMA CELULA DO NOTEBOOK ===
import zipfile
from datetime import datetime
from pathlib import Path

RUNS_DIR = Path("/content/drive/MyDrive/economy-classifier/artifacts/runs")
OUTPUT_ZIP = Path("/content") / f"runs_cards_{datetime.now():%Y%m%dT%H%M%S}.zip"

WHITELIST = {
    "result_card.json",
    "predictions.csv",
    "confusion_matrix.csv",
    "run_metadata.json",
    "search_result.json",
    "examples_binary.json",
    "examples_multiclass.json",
    "tfidf_config.json",
    "meta_classifier.joblib",
    "meta_classifier_meta.json",
    "agreement_matrix.csv",
    "fleiss_kappa.json",
    "contingency_table.csv",
}
BLACKLIST = {"predictions_checkpoint.csv"}

if not RUNS_DIR.exists():
    raise SystemExit(f"RUNS_DIR nao existe: {RUNS_DIR}")

n_files = n_runs = total_bytes = 0
with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
    for run_dir in sorted(p for p in RUNS_DIR.iterdir() if p.is_dir()):
        added = 0
        for path in run_dir.rglob("*"):
            if not path.is_file():
                continue
            if path.name in BLACKLIST or path.name not in WHITELIST:
                continue
            arcname = f"runs/{run_dir.name}/{path.relative_to(run_dir)}"
            zf.write(path, arcname)
            n_files += 1
            total_bytes += path.stat().st_size
            added += 1
        if added:
            n_runs += 1

print(f"Runs empacotados: {n_runs}")
print(f"Arquivos:         {n_files}  ({total_bytes / 1e6:.1f} MB brutos)")
print(f"Zip gerado:       {OUTPUT_ZIP}  ({OUTPUT_ZIP.stat().st_size / 1e6:.1f} MB)")

try:
    from google.colab import files
    files.download(str(OUTPUT_ZIP))
except ImportError:
    print("(fora do Colab: copie o zip manualmente)")
# === ATE AQUI ===

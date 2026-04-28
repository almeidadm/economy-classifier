#!/usr/bin/env python3
"""Unpack BERT training results from Google Colab into the local artifacts tree.

Usage:
    uv run python scripts/colab_unpack.py colab_bert_results.zip

Extracts run directories into ``artifacts/runs/``, preserving the structure
created by the Colab notebook:

    artifacts/runs/
        {timestamp}-bert-bertimbau/
        {timestamp}-bert-finbert-ptbr/
        {timestamp}-bert-deb3rta-base/

Each directory contains: predictions_val.csv, predictions_test.csv,
metrics.json, run_metadata.json, and optionally model/.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "artifacts" / "runs"


def main() -> None:
    if len(sys.argv) < 2:
        print("Uso: uv run python scripts/colab_unpack.py <caminho_do_zip>")
        sys.exit(1)

    zip_path = Path(sys.argv[1]).resolve()
    if not zip_path.exists():
        print(f"ERRO: Arquivo nao encontrado: {zip_path}")
        sys.exit(1)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        entries = zf.namelist()
        run_dirs = {e.split("/")[0] for e in entries if "/" in e}

        print(f"Runs encontrados no zip: {len(run_dirs)}")
        for run_dir in sorted(run_dirs):
            print(f"  - {run_dir}")

        zf.extractall(RUNS_DIR)

    print(f"\nArtefatos extraidos em: {RUNS_DIR}")
    print("\nConteudo integrado. Prossiga com o notebook 43_ensemble.")


if __name__ == "__main__":
    main()

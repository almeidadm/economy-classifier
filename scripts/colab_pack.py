#!/usr/bin/env python3
"""Package split artifacts for upload to Google Colab.

Usage:
    uv run python scripts/colab_pack.py

Produces ``colab_splits.zip`` in the project root, containing the parquet
splits and metadata needed to train BERT models on Colab.

Prerequisites:
    Run notebook 01_preparacao_dados.ipynb first to generate the splits.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR = PROJECT_ROOT / "artifacts" / "splits"
OUTPUT_ZIP = PROJECT_ROOT / "colab_splits.zip"

REQUIRED_FILES = [
    "train.parquet",
    "val.parquet",
    "test.parquet",
    "cv_folds.json",
    "split_metadata.json",
    "train_indices.csv",
    "val_indices.csv",
    "test_indices.csv",
]


def main() -> None:
    missing = [f for f in REQUIRED_FILES if not (SPLITS_DIR / f).exists()]
    if missing:
        print("ERRO: Arquivos nao encontrados em artifacts/splits/:")
        for f in missing:
            print(f"  - {f}")
        print("\nExecute o notebook 01_preparacao_dados.ipynb primeiro.")
        sys.exit(1)

    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for filename in REQUIRED_FILES:
            filepath = SPLITS_DIR / filename
            zf.write(filepath, f"splits/{filename}")
            size_mb = filepath.stat().st_size / 1e6
            print(f"  + {filename} ({size_mb:.1f} MB)")

    total_mb = OUTPUT_ZIP.stat().st_size / 1e6
    print(f"\nArquivo gerado: {OUTPUT_ZIP}")
    print(f"Tamanho total: {total_mb:.1f} MB")
    print("\nProximo passo: faça upload deste arquivo para o Google Drive")
    print("e abra o notebook 05_bert_colab.ipynb no Colab.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Empacota o RecognaSumm (test split) para upload no Google Colab.

Usage:
    uv run python scripts/colab_pack_recognasumm.py

Produz ``colab_recognasumm.zip`` na raiz do projeto, com a particao test
(27.055 docs, ~92 MB descompactado / ~38 MB compactado) baixada do HF.

Pre-requisito: ``data/recognasumm/test.jsonl`` baixado de
https://huggingface.co/datasets/recogna-nlp/recognasumm
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "recognasumm"
OUTPUT_ZIP = PROJECT_ROOT / "colab_recognasumm.zip"

REQUIRED_FILES = [
    "test.jsonl",
]


def main() -> None:
    missing = [f for f in REQUIRED_FILES if not (DATA_DIR / f).exists()]
    if missing:
        print("ERRO: Arquivos nao encontrados em data/recognasumm/:")
        for f in missing:
            print(f"  - {f}")
        print(
            "\nBaixe de https://huggingface.co/datasets/recogna-nlp/recognasumm "
            "ou veja docs/colab_run_recognasumm.md."
        )
        sys.exit(1)

    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for filename in REQUIRED_FILES:
            filepath = DATA_DIR / filename
            zf.write(filepath, f"recognasumm/{filename}")
            size_mb = filepath.stat().st_size / 1e6
            print(f"  + {filename} ({size_mb:.1f} MB)")

    total_mb = OUTPUT_ZIP.stat().st_size / 1e6
    print(f"\nArquivo gerado: {OUTPUT_ZIP}")
    print(f"Tamanho total: {total_mb:.1f} MB")
    print(
        "\nProximo passo: upload para o Google Drive em "
        "economy-classifier/colab_recognasumm.zip e siga "
        "docs/colab_run_recognasumm.md."
    )


if __name__ == "__main__":
    main()

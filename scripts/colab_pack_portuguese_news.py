#!/usr/bin/env python3
"""Empacota o PortugueseNewsDataset (WikiNoticias) para upload no Google Colab.

Usage:
    uv run python scripts/colab_pack_portuguese_news.py

Produz ``colab_portuguese_news.zip`` na raiz do projeto, com os arquivos JSON
reconstruidos via pipeline do repo Klaifer/PortugueseNewsDataset (PLOS ONE 2024).

Pre-requisito: reconstrucao concluida em
``data/portuguese_news_wikinotices/`` (extractor.py + seletor.py +
train_split.py do repo upstream — veja `docs/colab_run_portuguese_news.md`).
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "portuguese_news_wikinotices"
OUTPUT_ZIP = PROJECT_ROOT / "colab_portuguese_news.zip"

REQUIRED_FILES = [
    "wikinews_categories.json",
    "wikinews_train.json",
    "wikinews_test.json",
    "split_ids.csv",
]


def main() -> None:
    missing = [f for f in REQUIRED_FILES if not (DATA_DIR / f).exists()]
    if missing:
        print("ERRO: Arquivos nao encontrados em data/portuguese_news_wikinotices/:")
        for f in missing:
            print(f"  - {f}")
        print(
            "\nReconstrua o dataset primeiro (veja docs/colab_run_portuguese_news.md)."
        )
        sys.exit(1)

    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for filename in REQUIRED_FILES:
            filepath = DATA_DIR / filename
            zf.write(filepath, f"portuguese_news_wikinotices/{filename}")
            size_mb = filepath.stat().st_size / 1e6
            print(f"  + {filename} ({size_mb:.1f} MB)")

    total_mb = OUTPUT_ZIP.stat().st_size / 1e6
    print(f"\nArquivo gerado: {OUTPUT_ZIP}")
    print(f"Tamanho total: {total_mb:.1f} MB")
    print(
        "\nProximo passo: upload para o Google Drive em "
        "economy-classifier/colab_portuguese_news.zip e siga "
        "docs/colab_run_portuguese_news.md."
    )


if __name__ == "__main__":
    main()

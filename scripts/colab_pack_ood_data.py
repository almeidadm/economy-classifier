#!/usr/bin/env python3
"""Empacota os 4 datasets OOD num zip unico para upload no Google Colab.

Usage:
    uv run python scripts/colab_pack_ood_data.py

Produz ``colab_ood_data.zip`` na raiz do projeto com:

- ``fake_br/`` — Lira et al., copiado de ``fn-dataset-eda/data/raw/fake_br``
  (full_texts/{fake,true}/*.txt + metadata).
- ``fake_recogna/`` — copiado de ``fn-dataset-eda/data/raw/fake_recogna``
  (FakeRecogna_*.xlsx).
- ``portuguese_news_wikinotices/`` — Klaifer/WikiNoticias (PLOS ONE 2024),
  reconstruido em ``data/portuguese_news_wikinotices/`` via repo upstream.
- ``recognasumm/`` — RecognaSumm (Paiola et al., PROPOR 2024), baixado em
  ``data/recognasumm/test.jsonl``.

Estrutura final do zip casa exatamente com o que o notebook 45 espera:

    fake_br/
      full_texts/
        fake/<id>.txt
        true/<id>.txt
        fake-meta-information/<id>-meta.txt
        true-meta-information/<id>-meta.txt
    fake_recogna/
      FakeRecogna_*.xlsx
    portuguese_news_wikinotices/
      wikinews_categories.json
      wikinews_train.json
      wikinews_test.json
      split_ids.csv
    recognasumm/
      test.jsonl

Pre-requisitos:
- fn-dataset-eda clonado em ~/Documentos/repositorios/fn-dataset-eda/ (FB+FR).
- ``data/portuguese_news_wikinotices/`` reconstruido (ver docs/colab_run_portuguese_news.md).
- ``data/recognasumm/test.jsonl`` baixado de HF (ver docs/colab_run_recognasumm.md).
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ZIP = PROJECT_ROOT / "colab_ood_data.zip"

FN_DATASET_EDA_ROOT = Path.home() / "Documentos" / "repositorios" / "fn-dataset-eda" / "data" / "raw"
FAKE_BR_ROOT = FN_DATASET_EDA_ROOT / "fake_br"
FAKE_RECOGNA_ROOT = FN_DATASET_EDA_ROOT / "fake_recogna"

PN_ROOT = PROJECT_ROOT / "data" / "portuguese_news_wikinotices"
PN_FILES = [
    "wikinews_categories.json",
    "wikinews_train.json",
    "wikinews_test.json",
    "split_ids.csv",
]

RS_ROOT = PROJECT_ROOT / "data" / "recognasumm"
RS_FILES = ["test.jsonl", "train.jsonl", "validation.jsonl"]


def _validate_sources() -> None:
    errors: list[str] = []
    if not FAKE_BR_ROOT.exists():
        errors.append(
            f"Fake.Br ausente: {FAKE_BR_ROOT}\n"
            f"  Clone fn-dataset-eda em ~/Documentos/repositorios/."
        )
    if not FAKE_RECOGNA_ROOT.exists():
        errors.append(
            f"FakeRecogna ausente: {FAKE_RECOGNA_ROOT}\n"
            f"  Clone fn-dataset-eda em ~/Documentos/repositorios/."
        )
    pn_missing = [f for f in PN_FILES if not (PN_ROOT / f).exists()]
    if pn_missing:
        errors.append(
            f"PortugueseNewsDataset: faltam {pn_missing} em {PN_ROOT}\n"
            f"  Reconstrua via docs/colab_run_portuguese_news.md."
        )
    rs_missing = [f for f in RS_FILES if not (RS_ROOT / f).exists()]
    if rs_missing:
        errors.append(
            f"RecognaSumm: faltam {rs_missing} em {RS_ROOT}\n"
            f"  Baixe via docs/colab_run_recognasumm.md."
        )
    if errors:
        print("ERRO: fontes ausentes:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        sys.exit(1)


def _add_tree(zf: zipfile.ZipFile, src_root: Path, arc_prefix: str) -> tuple[int, int]:
    """Adiciona recursivamente src_root/* ao zip sob arc_prefix/."""
    n_files = 0
    total_bytes = 0
    for path in sorted(src_root.rglob("*")):
        if not path.is_file():
            continue
        arcname = f"{arc_prefix}/{path.relative_to(src_root).as_posix()}"
        zf.write(path, arcname)
        n_files += 1
        total_bytes += path.stat().st_size
    return n_files, total_bytes


def main() -> None:
    _validate_sources()

    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        n, b = _add_tree(zf, FAKE_BR_ROOT, "fake_br")
        print(f"  + fake_br/                       {n:>6d} files / {b/1e6:>7.1f} MB")
        n, b = _add_tree(zf, FAKE_RECOGNA_ROOT, "fake_recogna")
        print(f"  + fake_recogna/                  {n:>6d} files / {b/1e6:>7.1f} MB")

        n, b = 0, 0
        for fname in PN_FILES:
            src = PN_ROOT / fname
            zf.write(src, f"portuguese_news_wikinotices/{fname}")
            n += 1
            b += src.stat().st_size
        print(f"  + portuguese_news_wikinotices/   {n:>6d} files / {b/1e6:>7.1f} MB")

        n, b = 0, 0
        for fname in RS_FILES:
            src = RS_ROOT / fname
            zf.write(src, f"recognasumm/{fname}")
            n += 1
            b += src.stat().st_size
        print(f"  + recognasumm/                   {n:>6d} files / {b/1e6:>7.1f} MB")

    total_mb = OUTPUT_ZIP.stat().st_size / 1e6
    print(f"\nArquivo gerado: {OUTPUT_ZIP}")
    print(f"Tamanho total (compactado): {total_mb:.1f} MB")
    print(
        "\nProximo passo: upload para o Google Drive em "
        "economy-classifier/colab_ood_data.zip e execute o notebook 45."
    )


if __name__ == "__main__":
    main()
